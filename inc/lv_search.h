#ifndef _LV_SEARCH_H_INCLUDED
#define _LV_SEARCH_H_INCLUDED

#include <vector>
#include <iostream>
#include <nuSQuIDS/nuSQuIDS.h>
#include "nusquids_LV.h"
#include "get_eff_area.h"
#include <gsl/gsl_integration.h>

// chris weaver tools
#include <PhysTools/histogram.h>
#include "weighting.h"
#include "autodiff.h"
#include "likelihood.h"
// our stuff
#include "event.h"
#include "gsl_integrate_wrap.h"
#include "LV_to_flavour_function.h"
#include "chris_reads.h"

using namespace nusquids;

//==================== FITTER DEFINITIONS =========================//
//==================== FITTER DEFINITIONS =========================//

/// Simple fit result structure
struct fitResult {
  std::vector<double> params;
  double likelihood;
  unsigned int nEval, nGrad;
  bool succeeded;
};

/// A container to help with parsing index/value pairs for fixing likelihood parameters from the
/// commandline
struct paramFixSpec {
  struct singleParam : std::pair<unsigned int, double> {};
  std::vector<std::pair<unsigned int, double>> params;
};

/// Maximize a likelihood using the LBFGSB minimization algorithm
///\param likelihood The likelihood to maximize
///\param seed The seed values for all likelihood parameters
///\param indicesToFix The indices of likelihood parameters to hold constant at their seed va>
template <typename LikelihoodType>
fitResult doFitLBFGSB(LikelihoodType &likelihood, const std::vector<double> &seed,
                      std::vector<unsigned int> indicesToFix = {}) {
  using namespace likelihood;

  LBFGSB_Driver minimizer;
  // minimizer.addParameter(SEED, MAGIC_NUMBER, MINIMUM, MAXIMUM);
  minimizer.addParameter(seed[0], .001,
                         0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[1], .005);
  minimizer.addParameter(seed[2], .01, 0.0);
  minimizer.addParameter(seed[3], .001,
                         0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[4], .001,
                         0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[5], .005);

  for (auto idx : indicesToFix)
    minimizer.fixParameter(idx);

  minimizer.setChangeTolerance(1e-5);

  minimizer.setHistorySize(20);
  // std::cout << seed[0] << " " << seed[1] << " " << seed[2] << std::endl;

  fitResult result;
  result.succeeded = minimizer.minimize(BFGS_Function<LikelihoodType>(likelihood));
  result.likelihood = minimizer.minimumValue();
  result.params = minimizer.minimumPosition();
  result.nEval = minimizer.numberOfEvaluations();
  result.nGrad = minimizer.numberOfEvaluations();

  return (result);
}

//==================== NUISANCE PARAMETERS REWEIGHTERS =========================//
//==================== NUISANCE PARAMETERS REWEIGHTERS =========================//

template <typename T> struct powerlawWeighter : public GenericWeighter<powerlawWeighter<T>> {
private:
  T index;

public:
  using result_type = T;
  powerlawWeighter(T i) : index(i) {}

  template <typename Event> result_type operator()(const Event &e) const {
    return (pow((double)e.primaryEnergy, index));
  }
};

// Tilt a spectrum by an incremental powerlaw index about a precomputed median energy
template <typename Event, typename T>
struct powerlawTiltWeighter : public GenericWeighter<powerlawTiltWeighter<Event, T>> {
private:
  double medianEnergy;
  T deltaIndex;
  // typename Event::crTiltValues Event::* cachedData;
public:
  using result_type = T;
  powerlawTiltWeighter(double me, T dg /*, typename Event::crTiltValues Event::* c*/)
      : medianEnergy(me), deltaIndex(dg) /*,cachedData(c)*/ {}

  result_type operator()(const Event &e) const {
    // const typename Event::crTiltValues& cache=e.*cachedData;
    result_type weight = pow(e.energy_proxy / medianEnergy, -deltaIndex);
    return (weight);
  }
};

template <typename T, typename Event, typename U>
struct cachedValueWeighter : public GenericWeighter<cachedValueWeighter<T, Event, U>> {
private:
  U Event::*cachedPtr;

public:
  using result_type = T;
  cachedValueWeighter(U Event::*ptr) : cachedPtr(ptr) {}
  result_type operator()(const Event &e) const { return (result_type(e.*cachedPtr)); }
};

struct DiffuseFitWeighterMaker {
private:
  static constexpr double medianConvEnergy = 2020;   // GeV
  static constexpr double medianPromptEnergy = 7887; // GeV
  static constexpr double medianAstroEnergy = 1.0e5; // GeV
public:
  DiffuseFitWeighterMaker() {}

  template <typename DataType>
  std::function<DataType(const Event &)> operator()(const std::vector<DataType> &params) const {
    // check that we are getting the right number of nuisance parameters
    assert(params.size() == 6);
    // unpack things so we have legible names
    DataType convNorm = params[0];
    DataType CRDeltaGamma = params[1];
    DataType piKRatio = params[2];
    DataType promptNorm = params[3];
    DataType astroNorm = params[4];
    DataType astroDeltaGamma = params[5];

    using cachedWeighter = cachedValueWeighter<DataType, Event, double>;
    cachedWeighter convPionFlux(&Event::conv_pion_event); // we get the pion component
    cachedWeighter convKaonFlux(&Event::conv_kaon_event); // we get the kaon component
    cachedWeighter promptFlux(&Event::prompt_event);      // we get the prompt component
    cachedWeighter astroFlux(&Event::astro_event);        // we get the astro component

    auto conventionalComponent =
        convNorm * (convPionFlux + piKRatio * convKaonFlux) *
        powerlawTiltWeighter<Event, DataType>(
            medianConvEnergy, CRDeltaGamma); // we sum them upp according to some pi/k ratio.

    auto astroComponent =
        astroNorm * (astroFlux)*powerlawTiltWeighter<Event, DataType>(
                        medianAstroEnergy,
                        astroDeltaGamma); // constructing the astrophysical component with a tilt

    auto promptComponent = promptNorm * promptFlux *
                           powerlawTiltWeighter<Event, DataType>(medianPromptEnergy, CRDeltaGamma);

    return (conventionalComponent + promptComponent + astroComponent);
  }
};

//==================== AVERAGE FLUX CALCULATORS  =========================//
//==================== AVERAGE FLUX CALCULATORS  =========================//

double GetAveragedFlux(IntegrateWorkspace &ws, nuSQUIDSAtm<nuSQUIDSLV> *nus, PTypes flavor,
                       double costh, double enu_min, double enu_max) {
  if (enu_min > enu_max)
    throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");
  if (enu_min >= 1.0e6)
    return 0.;
  if (enu_max <= 1.0e2)
    return 0.;
  const double GeV = 1.0e9;
  const double integration_error = 1e-3;
  const unsigned int integration_iterations = 5000;
  if (flavor == NUMU) {
    return integrate(ws, [&](double enu) { return (nus->EvalFlavor(1, costh, enu * GeV, 0)); },
                     enu_min, enu_max, integration_error, integration_iterations);
  } else {
    return integrate(ws, [&](double enu) { return (nus->EvalFlavor(1, costh, enu * GeV, 1)); },
                     enu_min, enu_max, integration_error, integration_iterations);
  }
}

double GetAveragedFlux(IntegrateWorkspace &ws, nuSQUIDSAtm<nuSQUIDSLV> *nus, PTypes flavor,
                       double costh_min, double costh_max, double enu_min, double enu_max) {
  return (GetAveragedFlux(ws, nus, flavor, costh_max, enu_min, enu_max) +
          GetAveragedFlux(ws, nus, flavor, costh_min, enu_min, enu_max)) /
         2.;
}

double GetAvgPOsc(IntegrateWorkspace &ws, std::array<double, 3> params, double nlv,  PTypes flavor,
                  double costh_min, double costh_max, double enu_min, double enu_max) {
  if (enu_min > enu_max)
    throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");

  const double earth_diameter = 2 * 6371; // km
  const double integration_error = 1e-3;
  const unsigned int integration_iterations = 5000;

  double baseline_0 = -earth_diameter * costh_max, baseline_1 = -earth_diameter * costh_min;
  if (flavor == NUMU || flavor == NUMUBAR) {
    return integrate(ws,
                     [&](double enu) {
                       return LV::OscillationProbabilityTwoFlavorLV_intL(
                           enu, baseline_0, baseline_1, params[0], params[1], params[2], nlv);
                     },
                     enu_min, enu_max, integration_error, integration_iterations) /
           (baseline_1 - baseline_0) / (enu_max - enu_min);
  } else {
    throw std::logic_error("MNot implemented.");
  }
}

const int astro_gamma = -2;
const double N0 = 9.9e-9; // GeV^-1 cm^-2 sr^-1 s^-1

double GetAveragedAstroFlux(IntegrateWorkspace &ws, PTypes flavor, double costh, double enu_min,
                            double enu_max) {
  if (enu_min > enu_max)
    throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");
  if (enu_min >= 1.0e6)
    return 0.;
  if (enu_max <= 1.0e2)
    return 0.;

  if (astro_gamma != -1)
    return N0 * (pow(enu_min, astro_gamma + 1.) - pow(enu_max, astro_gamma + 1.)) /
           (astro_gamma + 1.);
  else
    return N0 * (log(enu_max) - log(enu_min));
}

double GetAveragedAstroFlux(IntegrateWorkspace &ws, PTypes flavor, double costh_min,
                            double costh_max, double enu_min, double enu_max) {
  // the astrophical flux is independent of cos(th)
  return GetAveragedAstroFlux(ws, flavor, costh_max, enu_min, enu_max);
}

double GetAvgAstroPOsc(IntegrateWorkspace &ws, std::array<double, 3> params, double nlv, PTypes flavor,
                       double costh_min, double costh_max, double enu_min, double enu_max) {
  if (enu_min > enu_max)
    throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");

  if (flavor == NUMU || flavor == NUMUBAR) {
    const double integration_error = 1e-3;
    const unsigned int integration_iterations = 5000;
    return integrate(ws, [&](double enu) {
             return LV::OscillationProbabilityTwoFlavorLV_Astro(enu, params[0], params[1],
                                                                params[2],nlv);
           }, enu_min, enu_max, integration_error, integration_iterations) / (enu_max - enu_min);
  } else {
    throw std::logic_error("MNot implemented.");
  }
}

//==================== MAIN CLASS  =========================//
//==================== MAIN CLASS  =========================//
//==================== MAIN CLASS  =========================//

using namespace phys_tools::histograms;
using HistType = histogram<3, likelihood::entryStoringBin<std::reference_wrapper<const Event>>>;
using phys_tools::histograms::amount;
auto binner = [](HistType &h, const Event &e) {
  h.add(e.energy_proxy, e.costh, e.year, amount(std::cref(e)));
};

template<typename ContainerType, typename HistType, typename BinnerType>
void bin(const ContainerType& data, HistType& hist, const BinnerType& binner){
  for(const Event& event : data)
    binner(hist,event);
}

class LVSearch {
private:
  double nlv= 1; // by default c-style constrains
  bool quiet = true;
  const double PI_CONSTANT = 3.141592;
  const double m2Tocm2 = 1.0e4;

  const unsigned int neutrino_energy_index = 0;
  const unsigned int coszenith_index = 1;
  const unsigned int proxy_energy_index = 2;
  const unsigned int neutrinoEnergyBins = 280;
  const unsigned int cosZenithBins = 11;
  const unsigned int energyProxyBins = 50;
  const unsigned int number_of_years = 2;
  const unsigned int histogramDims[3] = {neutrinoEnergyBins, cosZenithBins, energyProxyBins};
  const double minFitEnergy = 4.0e2;
  const double maxFitEnergy = 1.8e4;
  const double minCosth = -1;
  const double maxCosth = 0.2;

  // evaluation
  size_t evalThreads=1;
  std::vector<double> fitSeed {1.,0.,1.,1.,1.,0.};
protected:
  // gsl workspace
  IntegrateWorkspace ws;
  // events
  marray<double, 2> observed_data;
  std::deque<Event> observed_events;
  HistType data_hist;
  // effective areas
  AreaEdges edges;
  AreaArray areas;
  // lifetimes
  std::array<double, 2> livetime;
  // nusquids fluxes
  nuSQUIDSAtm<nuSQUIDSLV> *nus_kaon;
  nuSQUIDSAtm<nuSQUIDSLV> *nus_pion;
  nuSQUIDSAtm<nuSQUIDSLV> *nus_prompt;
  // prebinned fluxes
  multidim convPionAtmosFlux[2];
  multidim convKaonAtmosFlux[2];
  multidim promptAtmosFlux[2];
  // DOM efficiency correction
  multidim convDOMEffCorrection[2];
  // Simulation histogram
  HistType sim_hist;
  // Simulation events
  std::deque<Event> mc_events;
  // Expectation arrays
  nusquids::marray<double, 3> kaon_event_expectation;
  nusquids::marray<double, 3> pion_event_expectation;
  nusquids::marray<double, 3> prompt_event_expectation;
  nusquids::marray<double, 3> astro_event_expectation;

public:
  LVSearch(std::string effective_area_filename, std::string data_filename,
           std::string detector_correction_filename, std::string kaon_filename,
           std::string pion_filename, std::string prompt_filename, bool quiet = true):
    ws(IntegrateWorkspace(5000)),quiet(quiet)
   {

    livetime[y2010] = 2.7282e+07;  // in seconds
    livetime[y2011] = 2.96986e+07; // in seconds

    kaon_event_expectation.resize(
        std::vector<size_t>{number_of_years, cosZenithBins, energyProxyBins});
    pion_event_expectation.resize(
        std::vector<size_t>{number_of_years, cosZenithBins, energyProxyBins});
    prompt_event_expectation.resize(
        std::vector<size_t>{number_of_years, cosZenithBins, energyProxyBins});
    astro_event_expectation.resize(
        std::vector<size_t>{number_of_years, cosZenithBins, energyProxyBins});

    convPionAtmosFlux[0] = alloc_multi(2,histogramDims);
    convPionAtmosFlux[1] = alloc_multi(2,histogramDims);

    convKaonAtmosFlux[0] = alloc_multi(2,histogramDims);
    convKaonAtmosFlux[1] = alloc_multi(2,histogramDims);

    promptAtmosFlux[0] = alloc_multi(2,histogramDims);
    promptAtmosFlux[1] = alloc_multi(2,histogramDims);

    convDOMEffCorrection[0] = alloc_multi(3,histogramDims);
    convDOMEffCorrection[1] = alloc_multi(3,histogramDims);

    LoadData(data_filename);
    LoadEffectiveArea(effective_area_filename);
    LoadFluxes(kaon_filename, pion_filename, prompt_filename);
    LoadDetectorCorrection(detector_correction_filename);
  }

  void SetVerbose(bool quietness) { quiet = !quietness; }
  void SetEnergyExponent(double nlv_) { nlv = nlv_;}

protected:
  void LoadData(std::string data_filename) {
    if (!quiet)
      std::cout << "Loading data." << std::endl;
    observed_data = quickread(data_filename);
    observed_events.clear();
    for (unsigned int irow = 0; irow < observed_data.extent(0); irow++) {
      observed_events.push_back(Event(observed_data[irow][0], // energy proxy
                                      observed_data[irow][1], // costh
                                      static_cast<unsigned int>(observed_data[irow][2]))); // year
    }
    if (!quiet)
      std::cout << "Histograming data." << std::endl;
    // this magic number set the right bin edges
    data_hist = HistType(LogarithmicAxis(0, 0.1), LinearAxis(0, 0.1), LinearAxis(2010, 1));

    data_hist.getAxis(0)->setLowerLimit(minFitEnergy);
    data_hist.getAxis(0)->setUpperLimit(maxFitEnergy);
    data_hist.getAxis(1)->setLowerLimit(minCosth);
    data_hist.getAxis(1)->setUpperLimit(maxCosth);

    // fill in the histogram with the data
    bin(observed_events, data_hist, binner);

    size_t fill_bins = data_hist.getBinCount(0)*data_hist.getBinCount(1)*data_hist.getBinCount(2);
    size_t event_number = 0;
    for(auto it=data_hist.begin(); it!= data_hist.end(); it++){
      auto itc=static_cast<likelihood::entryStoringBin<std::reference_wrapper<const Event>>>(*it);
      event_number+=itc.size();
    }

    if(fill_bins == 0)
      throw std::runtime_error("No events loaded. Are you sure you are using the right event file/format?");

    if (!quiet)
      std::cout << "Data histogram has " << fill_bins << " bins filled. " << event_number << " events were loaded." << std::endl;
  }

  void LoadEffectiveArea(std::string effective_area_filename) {
    if (!quiet)
      std::cout << "Loading effective areas." << std::endl;
    areas = get_areas(effective_area_filename, edges);
  }

  void LoadFluxes(std::string kaon_filename, std::string pion_filename, std::string prompt_filename) {
    if (!quiet)
      std::cout << "Loading fluxes." << std::endl;

    // reading pion flux
    hid_t pion_file_id = H5Fopen(pion_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    readDataSet(pion_file_id, "/nu_mu/integrated_flux", convPionAtmosFlux[0].data);
    readDataSet(pion_file_id, "/nu_mu_bar/integrated_flux", convPionAtmosFlux[1].data);

    H5Fclose(pion_file_id);

    // reading kaon flux
    hid_t Kaon_file_id = H5Fopen(kaon_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    readDataSet(Kaon_file_id, "/nu_mu/integrated_flux", convKaonAtmosFlux[0].data);
    readDataSet(Kaon_file_id, "/nu_mu_bar/integrated_flux", convKaonAtmosFlux[1].data);

    H5Fclose(Kaon_file_id);

    // reading prompt flux
    hid_t prompt_file_id = H5Fopen(prompt_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    readDataSet(Kaon_file_id, "/nu_mu/integrated_flux", promptAtmosFlux[0].data);
    readDataSet(Kaon_file_id, "/nu_mu_bar/integrated_flux", promptAtmosFlux[1].data);

    H5Fclose(prompt_file_id);
  }

  void LoadDetectorCorrection(std::string detector_correction_filename) {

    // reading the DOM efficiency correction
    hid_t file_id = H5Fopen(detector_correction_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    readDataSet(file_id, "/detector_correction/2010", convDOMEffCorrection[0].data);
    readDataSet(file_id, "/detector_correction/2011", convDOMEffCorrection[1].data);
  }

protected:
  void MakeMCEvents(std::array<double, 3> &osc_params) {
    if (!quiet) {
      std::cout << "Calculating simulation weights." << std::endl;
    }

    for (auto it = kaon_event_expectation.begin(); it < kaon_event_expectation.end(); it++)
      *it = 0.;
    for (auto it = pion_event_expectation.begin(); it < pion_event_expectation.end(); it++)
      *it = 0.;
    for (auto it = prompt_event_expectation.begin(); it < prompt_event_expectation.end(); it++)
      *it = 0.;
    for (auto it = astro_event_expectation.begin(); it < astro_event_expectation.end(); it++)
      *it = 0.;

    unsigned int indices[3], p, y;

#define USE_CHRIS_FLUX
#ifndef USE_CHRIS_FLUX
    // read nusquids calculated flux
    if (!quiet) {
      std::cout << "Loading nuSQuIDS fluxes." << std::endl;
    }

    for (PTypes flavor : {NUMU, NUMUBAR}) {
      for (unsigned int ei = 0; ei < neutrinoEnergyBins; ei++) {
        for (unsigned int ci = 0; ci < cosZenithBins; ci++) {
          double p_osc =
              GetAvgPOsc(ws.get(), osc_params, nlv, flavor, edges[y2010][flavor][coszenith_index][ci],
                         edges[y2010][flavor][coszenith_index][ci + 1],
                         edges[y2010][flavor][neutrino_energy_index][ei],
                         edges[y2010][flavor][neutrino_energy_index][ei + 1]);
          double kaon_integrated_flux =
              GetAveragedFlux(ws, &nus_kaon, flavor, edges[y2010][flavor][coszenith_index][ci],
                              edges[y2010][flavor][coszenith_index][ci + 1],
                              edges[y2010][flavor][neutrino_energy_index][ei],
                              edges[y2010][flavor][neutrino_energy_index][ei + 1]) *
              p_osc;
          double pion_integrated_flux =
              GetAveragedFlux(ws, &nus_pion, flavor, edges[y2010][flavor][coszenith_index][ci],
                              edges[y2010][flavor][coszenith_index][ci + 1],
                              edges[y2010][flavor][neutrino_energy_index][ei],
                              edges[y2010][flavor][neutrino_energy_index][ei + 1]) *
              p_osc;

          double p_osc_astro =
              GetAvgAstroPOsc(ws, osc_params, nlv, flavor, edges[y2010][flavor][coszenith_index][ci],
                              edges[y2010][flavor][coszenith_index][ci + 1],
                              edges[y2010][flavor][neutrino_energy_index][ei],
                              edges[y2010][flavor][neutrino_energy_index][ei + 1]);
          double astro_integrated_flux =
              GetAveragedAstroFlux(ws, flavor, edges[y2010][flavor][coszenith_index][ci],
                                   edges[y2010][flavor][coszenith_index][ci + 1],
                                   edges[y2010][flavor][neutrino_energy_index][ei],
                                   edges[y2010][flavor][neutrino_energy_index][ei + 1]) *
              p_osc_astro;
          for (unsigned int pi = 0; pi < energyProxyBins; pi++) {
            for (Year year : {y2010, y2011}) {
              indices[0] = ei;
              indices[1] = ci;
              indices[2] = pi;
              p = (flavor == NUMU) ? 0 : 1;
              y = (year == y2010) ? 0 : 1;
              double solid_angle =
                  2. * PI_CONSTANT * (edges[year][flavor][coszenith_index][ci + 1] -
                                      edges[year][flavor][coszenith_index][ci]);
              double DOM_eff_correction =
                  1.; // this correction is flux dependent, we will need to fix this.
              // double DOM_eff_correction =*index_multi(*convDOMEffCorrection[y],indices);
              kaon_event_expectation[year][ci][pi] +=
                  DOM_eff_correction * solid_angle * m2Tocm2 * livetime[year] *
                  areas[year][flavor][ei][ci][pi] * kaon_integrated_flux;
              pion_event_expectation[year][ci][pi] +=
                  DOM_eff_correction * solid_angle * m2Tocm2 * livetime[year] *
                  areas[year][flavor][ei][ci][pi] * pion_integrated_flux;
              prompt_event_expectation[year][ci][pi] += 0.; // no nusquids flux yet for prompt
              astro_event_expectation[year][ci][pi] +=
                  DOM_eff_correction * solid_angle * m2Tocm2 * livetime[year] *
                  areas[year][flavor][ei][ci][pi] * astro_integrated_flux;
            }
          }
        }
      }
    }
#else
    for (PTypes flavor : {NUMU, NUMUBAR}) {
      for (unsigned int ci = 0; ci < cosZenithBins; ci++) {
        for (unsigned int ei = 0; ei < neutrinoEnergyBins; ei++) {
          double p_osc =
              GetAvgPOsc(ws, osc_params, nlv, flavor, edges[y2010][flavor][coszenith_index][ci],
                         edges[y2010][flavor][coszenith_index][ci + 1],
                         edges[y2010][flavor][neutrino_energy_index][ei],
                         edges[y2010][flavor][neutrino_energy_index][ei + 1]);
          double p_osc_astro =
              GetAvgAstroPOsc(ws, osc_params, nlv, flavor, edges[y2010][flavor][coszenith_index][ci],
                              edges[y2010][flavor][coszenith_index][ci + 1],
                              edges[y2010][flavor][neutrino_energy_index][ei],
                              edges[y2010][flavor][neutrino_energy_index][ei + 1]);
          double astro_integrated_flux =
              GetAveragedAstroFlux(ws, flavor, edges[y2010][flavor][coszenith_index][ci],
                                   edges[y2010][flavor][coszenith_index][ci + 1],
                                   edges[y2010][flavor][neutrino_energy_index][ei],
                                   edges[y2010][flavor][neutrino_energy_index][ei + 1]) *
              p_osc_astro;

          for (unsigned int pi = 0; pi < energyProxyBins; pi++) {
            for (Year year : {y2010, y2011}) {
              // std::cout << ci << " " << edges[year][flavor][coszenith_index][ci] << std::endl;
              indices[0] = ei;
              indices[1] = ci;
              indices[2] = pi;
              p = (flavor == NUMU) ? 0 : 1;
              y = (year == y2010) ? 0 : 1;
              double solid_angle =
                  2. * PI_CONSTANT * (edges[year][flavor][coszenith_index][ci + 1] -
                                      edges[year][flavor][coszenith_index][ci]);
              double pion_fluxIntegral = *index_multi(convPionAtmosFlux[p], indices);
              double kaon_fluxIntegral = *index_multi(convKaonAtmosFlux[p], indices);
              double prompt_fluxIntegral = *index_multi(promptAtmosFlux[p], indices);
              double DOM_eff_correction = *index_multi(convDOMEffCorrection[y], indices);
              // chris does not separate between pions and kaon components. Lets just drop it all in
              // the kaons.
              // also chris has already included the solid angle factor in the flux
              kaon_event_expectation[year][ci][pi] +=
                  m2Tocm2 * livetime[year] * areas[year][flavor][ei][ci][pi] * DOM_eff_correction *
                  kaon_fluxIntegral * p_osc;
              pion_event_expectation[year][ci][pi] +=
                  m2Tocm2 * livetime[year] * areas[year][flavor][ei][ci][pi] * DOM_eff_correction *
                  pion_fluxIntegral * p_osc;
              prompt_event_expectation[year][ci][pi] +=
                  m2Tocm2 * livetime[year] * areas[year][flavor][ei][ci][pi] * DOM_eff_correction *
                  prompt_fluxIntegral * p_osc;
              astro_event_expectation[year][ci][pi] +=
                  DOM_eff_correction * solid_angle * m2Tocm2 * livetime[year] *
                  areas[year][flavor][ei][ci][pi] * astro_integrated_flux;
              if (kaon_event_expectation[year][ci][pi] < 0)
                throw std::runtime_error("badness");
            }
          }
        }
      }
    }
#endif

    // to keep the code simple we are going to construct fake MC events
    // which weights are just the expectation value. The advantage of this is that
    // when we have the actual IceCube MC the following code can be reused. Love, CA.

    if (!quiet) {
      std::cout << "Making simulation events." << std::endl;
    }
    mc_events.clear();
    for (Year year : {y2010, y2011}) {
      for (unsigned int ci = 0; ci < cosZenithBins; ci++) {
        for (unsigned int pi = 0; pi < energyProxyBins; pi++) {
          // we are going to save the bin content and label it at the bin center
          mc_events.push_back(Event((edges[year][flavor][proxy_energy_index][pi] +
                                     edges[year][flavor][proxy_energy_index][pi + 1]) /
                                        2., // energy proxy bin center
                                    (edges[year][flavor][coszenith_index][ci] +
                                     edges[year][flavor][coszenith_index][ci + 1]) /
                                        2.,                                  // costh bin center
                                    year == y2010 ? 2010 : 2011,             // year
                                    kaon_event_expectation[year][ci][pi],    // amount of kaon
                                                                             // component events
                                    pion_event_expectation[year][ci][pi],    // amount of pion
                                                                             // component events
                                    prompt_event_expectation[year][ci][pi],  // amount of prompt
                                                                             // component events
                                    astro_event_expectation[year][ci][pi])); // amount of
                                                                             // astrophysical
                                                                             // component events
        }
      }
    }
  }

  void MakeSimulationHistogram(std::array<double, 3> &osc_params) {
    if (!quiet) {
      std::cout << "Constructing simulation histogram." << std::endl;
    }
    // create fake events according to hypothesis
    MakeMCEvents(osc_params);
    // create MC histogram with the same binning as the data
    sim_hist = makeEmptyHistogramCopy(data_hist);
    // fill in the histogram with the mc events
    bin(mc_events, sim_hist, binner);
  }

public:
  marray<double,3> GetDataDistribution(){

    marray<double,3> array {static_cast<size_t>(data_hist.getBinCount(2)),
                            static_cast<size_t>(data_hist.getBinCount(1)),
                            static_cast<size_t>(data_hist.getBinCount(0))};

    for(size_t iy=0; iy<data_hist.getBinCount(2); iy++){ // year
      for(size_t ic=0; ic<data_hist.getBinCount(1); ic++){ // zenith
        for(size_t ie=0; ie<data_hist.getBinCount(0); ie++){ // energy
          auto itc = static_cast<likelihood::entryStoringBin<std::reference_wrapper<const Event>>>(data_hist(ie,ic,iy));
          array[iy][ic][ie] = itc.size();
        }
      }
    }
    return array;
  }

  marray<double,3> GetExpectationDistribution(std::array<double, 9> & params){

    // get physics parameters
    std::array<double,3> osc_params {params[6],params[7],params[8]};
    MakeSimulationHistogram(osc_params);
    // get nuisance parameters
    std::vector<double> nuisance {params[0],params[1],params[2],params[3],params[4],params[5]};

    marray<double,3> array {static_cast<size_t>(sim_hist.getBinCount(2)),
                            static_cast<size_t>(sim_hist.getBinCount(1)),
                            static_cast<size_t>(sim_hist.getBinCount(0))};

    DiffuseFitWeighterMaker DFWM;
    auto weighter = DFWM(nuisance);
    for(size_t iy=0; iy<sim_hist.getBinCount(2); iy++){ // year
      for(size_t ic=0; ic<sim_hist.getBinCount(1); ic++){ // zenith
        for(size_t ie=0; ie<sim_hist.getBinCount(0); ie++){ // energy
          auto itc = static_cast<likelihood::entryStoringBin<std::reference_wrapper<const Event>>>(sim_hist(ie,ic,iy));
          double expectation=0;
          for(auto event : itc.entries()){
            expectation+=weighter(event);
          }
          array[iy][ic][ie] = expectation;
        }
      }
    }
    return array;
  }

  double llhFull(std::array<double, 9> & params){
    // get physics parameters
    std::array<double,3> osc_params {params[6],params[7],params[8]};
    MakeSimulationHistogram(osc_params);
    // get nuisance parameters
    std::vector<double> nuisance {params[0],params[1],params[2],params[3],params[4],params[5]};

    if (!quiet) {
      std::cout << "Defining priors." << std::endl;
    }
    // here we define the priors
    likelihood::UniformPrior positivePrior(0.0, std::numeric_limits<double>::infinity());
    likelihood::GaussianPrior normalizationPrior(1., 0.4); // 0.4
    likelihood::GaussianPrior crSlopePrior(0.0, 0.05);
    likelihood::GaussianPrior kaonPrior(1.0, 0.1);
    likelihood::UniformPrior prompt_norm(0.0, std::numeric_limits<double>::infinity());
    likelihood::UniformPrior astro_norm(0.0, std::numeric_limits<double>::infinity());
    likelihood::UniformPrior astro_gamma(-0.5, 0.5);

    auto priors = makePriorSet(normalizationPrior, crSlopePrior, kaonPrior, prompt_norm, astro_norm,
                               astro_gamma);
    // construct a MC event reweighter
    DiffuseFitWeighterMaker DFWM;
    // construct likelihood problem
    // there are two numbers here. The first number is the number of histogram
    // dimension, in this case 3.
    // The second number is the number of nuisance parameters, it is 3 for
    // conventional-only,
    // and 5 for conventional+astro, and 6 for conventional+astro+prompt.
    auto prob = likelihood::makeLikelihoodProblem<std::reference_wrapper<const Event>, 3, 6>(
        data_hist, {sim_hist}, priors, {1.0}, likelihood::simpleDataWeighter(), DFWM,
        likelihood::poissonLikelihood(), fitSeed);
    prob.setEvaluationThreadCount(evalThreads);

    return -prob.evaluateLikelihood(nuisance);
  }

  fitResult llh(std::array<double, 3> &osc_params) {
    MakeSimulationHistogram(osc_params);

    if (!quiet) {
      std::cout << "Defining priors." << std::endl;
    }
    // here we define the priors
    likelihood::UniformPrior positivePrior(0.0, std::numeric_limits<double>::infinity());
    likelihood::GaussianPrior normalizationPrior(1., 0.4); // 0.4
    likelihood::GaussianPrior crSlopePrior(0.0, 0.05);
    likelihood::GaussianPrior kaonPrior(1.0, 0.1);
    likelihood::UniformPrior prompt_norm(0.0, std::numeric_limits<double>::infinity());
    likelihood::UniformPrior astro_norm(0.0, std::numeric_limits<double>::infinity());
    likelihood::UniformPrior astro_gamma(-0.5, 0.5);

    auto priors = makePriorSet(normalizationPrior, crSlopePrior, kaonPrior, prompt_norm, astro_norm,
                               astro_gamma);
    // construct a MC event reweighter
    DiffuseFitWeighterMaker DFWM;
    if (!quiet) {
      std::cout << "Constructing likelihood problem." << std::endl;
    }
    // construct likelihood problem
    // there are two numbers here. The first number is the number of histogram
    // dimension, in this case 3.
    // The second number is the number of nuisance parameters, it is 3 for
    // conventional-only,
    // and 5 for conventional+astro, and 6 for conventional+astro+prompt.
    auto prob = likelihood::makeLikelihoodProblem<std::reference_wrapper<const Event>, 3, 6>(
        data_hist, {sim_hist}, priors, {1.0}, likelihood::simpleDataWeighter(), DFWM,
        likelihood::poissonLikelihood(), fitSeed);
    prob.setEvaluationThreadCount(evalThreads);

    std::vector<double> seed = prob.getSeed();
    std::vector<unsigned int> fixedIndices = {}; // nuisance parameters that will be fixed
    paramFixSpec fixedParams;
    for (const auto pf : fixedParams.params) {
      if (!quiet)
        std::cout << "Fitting with parameter " << pf.first << " fixed to " << pf.second
                  << std::endl;
      seed[pf.first] = pf.second;
      fixedIndices.push_back(pf.first);
    }
    if (!quiet) {
      std::cout << "Finding minima." << std::endl;
    }
    // minimize over the nuisance parameters
    fitResult fr = doFitLBFGSB(prob, seed, fixedIndices);
    return fr;
  }
};

#endif
