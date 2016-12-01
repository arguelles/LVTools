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
//#include "likelihood_cool.h"
#include "likelihood.h"
#include "event.h"
#include "gsl_integrate_wrap.h"
#include "LV_to_flavour_function.h"
#include "chris_reads.h"

using namespace nusquids;

///Simple fit result structure
struct fitResult{
  std::vector<double> params;
  double likelihood;
  unsigned int nEval, nGrad;
  bool succeeded;
};

///A container to help with parsing index/value pairs for fixing likelihood parameters from the commandline
struct paramFixSpec{
	struct singleParam : std::pair<unsigned int, double>{};
	std::vector<std::pair<unsigned int,double>> params;
};

///Maximize a likelihood using the LBFGSB minimization algorithm
///\param likelihood The likelihood to maximize
///\param seed The seed values for all likelihood parameters
///\param indicesToFix The indices of likelihood parameters to hold constant at their seed va>
template<typename LikelihoodType>
fitResult doFitLBFGSB(LikelihoodType& likelihood, const std::vector<double>& seed,
            std::vector<unsigned int> indicesToFix={}){
  using namespace likelihood;

  LBFGSB_Driver minimizer;
  //minimizer.addParameter(SEED, MAGIC_NUMBER, MINIMUM, MAXIMUM);
  minimizer.addParameter(seed[0],.001,0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[1],.005);
  minimizer.addParameter(seed[2],.01,0.0);
  minimizer.addParameter(seed[3],.001,0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[4],.001,0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[5],.005);

  for(auto idx : indicesToFix)
    minimizer.fixParameter(idx);

  minimizer.setChangeTolerance(1e-5);

  minimizer.setHistorySize(20);
  //std::cout << seed[0] << " " << seed[1] << " " << seed[2] << std::endl;

  fitResult result;
  result.succeeded=minimizer.minimize(BFGS_Function<LikelihoodType>(likelihood));
  result.likelihood=minimizer.minimumValue();
  result.params=minimizer.minimumPosition();
  result.nEval=minimizer.numberOfEvaluations();
  result.nGrad=minimizer.numberOfEvaluations();

  return(result);
}


template<typename ContainerType, typename HistType, typename BinnerType>
void bin(const ContainerType& data, HistType& hist, const BinnerType& binner){
  for(const Event& event : data)
    binner(hist,event);
}

const double integration_error = 1e-3;
const unsigned int integration_iterations = 5000;

double GetAveragedFlux(IntegrateWorkspace& ws, nuSQUIDSAtm<nuSQUIDSLV> * nus,PTypes flavor, double costh, double enu_min, double enu_max) {
    if(enu_min > enu_max)
      throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");
    if (enu_min >= 1.0e6)
      return 0.;
    if (enu_max <= 1.0e2)
      return 0.;
    double GeV = 1.0e9;
    if (flavor == NUMU){
        return integrate(ws, [&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,0));},enu_min,enu_max, integration_error, integration_iterations);
    } else {
        return integrate(ws, [&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,1));},enu_min,enu_max, integration_error, integration_iterations);
    }
}

double GetAveragedFlux(IntegrateWorkspace& ws, nuSQUIDSAtm<nuSQUIDSLV> * nus,PTypes flavor, double costh_min, double costh_max, double enu_min, double enu_max) {
  return (GetAveragedFlux(ws, nus,flavor,costh_max,enu_min,enu_max) + GetAveragedFlux(ws, nus,flavor,costh_min,enu_min,enu_max))/2.;
}

double GetAvgPOsc(IntegrateWorkspace& ws, std::array<double, 3> params, PTypes flavor, double costh_min, double costh_max, double enu_min, double enu_max) {
    if(enu_min > enu_max)
      throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");

    const double earth_diameter = 2*6371; // km

    double baseline_0 = -earth_diameter*costh_max, baseline_1 = -earth_diameter*costh_min ;
    if (flavor == NUMU || flavor == NUMUBAR){
        return integrate(ws, [&](double enu){return  LV::OscillationProbabilityTwoFlavorLV_intL(enu, baseline_0, baseline_1, params[0], params[1], params[2]); },enu_min,enu_max, integration_error, integration_iterations)/(baseline_1 - baseline_0)/(enu_max-enu_min);
    } else {
        throw std::logic_error("MNot implemented.");
    }
}

const int astro_gamma = -2;
const double N0 = 9.9e-9; // GeV^-1 cm^-2 sr^-1 s^-1

double GetAveragedAstroFlux(IntegrateWorkspace& ws, PTypes flavor, double costh, double enu_min, double enu_max) {
    if(enu_min > enu_max)
      throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");
    if (enu_min >= 1.0e6)
      return 0.;
    if (enu_max <= 1.0e2)
      return 0.;

    if(astro_gamma != -1)
      return N0*(pow(enu_min,astro_gamma+1.)-pow(enu_max,astro_gamma+1.))/(astro_gamma+1.);
    else
      return N0*(log(enu_max)-log(enu_min));
}

double GetAveragedAstroFlux(IntegrateWorkspace& ws, PTypes flavor, double costh_min, double costh_max, double enu_min, double enu_max) {
  // the astrophical flux is independent of cos(th)
  return GetAveragedAstroFlux(ws,flavor,costh_max,enu_min,enu_max);
}

double GetAvgAstroPOsc(IntegrateWorkspace& ws, std::array<double, 3> params, PTypes flavor, double costh_min, double costh_max, double enu_min, double enu_max) {
    if(enu_min > enu_max)
      throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");

    if (flavor == NUMU || flavor == NUMUBAR){
        return integrate(ws, [&](double enu){return  LV::OscillationProbabilityTwoFlavorLV_Astro(enu, params[0], params[1], params[2]); },enu_min,enu_max, integration_error, integration_iterations)/(enu_max-enu_min);
    } else {
        throw std::logic_error("MNot implemented.");
    }
}



//==================== NUISANCE PARAMETERS REWEIGHTERS =========================//
//==================== NUISANCE PARAMETERS REWEIGHTERS =========================//


template<typename T>
struct powerlawWeighter : public GenericWeighter<powerlawWeighter<T>>{
private:
	T index;
public:
	using result_type=T;
	powerlawWeighter(T i):index(i){}

	template<typename Event>
	result_type operator()(const Event& e) const{
		return(pow((double)e.primaryEnergy,index));
	}
};

//Tilt a spectrum by an incremental powerlaw index about a precomputed median energy
template<typename Event, typename T>
struct powerlawTiltWeighter : public GenericWeighter<powerlawTiltWeighter<Event,T>>{
private:
	double medianEnergy;
	T deltaIndex;
	//typename Event::crTiltValues Event::* cachedData;
public:
	using result_type=T;
	powerlawTiltWeighter(double me, T dg/*, typename Event::crTiltValues Event::* c*/):
	medianEnergy(me),deltaIndex(dg)/*,cachedData(c)*/{}

	result_type operator()(const Event& e) const{
		//const typename Event::crTiltValues& cache=e.*cachedData;
		result_type weight=pow(e.energy_proxy/medianEnergy,-deltaIndex);
		return(weight);
	}
};

template<typename T, typename Event, typename U>
struct cachedValueWeighter : public GenericWeighter<cachedValueWeighter<T,Event,U>>{
private:
	U Event::* cachedPtr;
public:
	using result_type=T;
	cachedValueWeighter(U Event::* ptr):cachedPtr(ptr){}
	result_type operator()(const Event& e) const{
		return(result_type(e.*cachedPtr));
	}
};

struct DiffuseFitWeighterMaker{
private:
	static constexpr double medianConvEnergy=2020; // GeV
	static constexpr double medianPromptEnergy=7887; // GeV
	static constexpr double medianAstroEnergy=1.0e5; // GeV
public:
	DiffuseFitWeighterMaker()
	{}

	template<typename DataType>
	std::function<DataType(const Event&)> operator()(const std::vector<DataType>& params) const{
    // check that we are getting the right number of nuisance parameters
		assert(params.size()==6);
		//unpack things so we have legible names
		DataType convNorm=params[0];
		DataType CRDeltaGamma=params[1];
		DataType piKRatio=params[2];
		DataType promptNorm=params[3];
		DataType astroNorm=params[4];
		DataType astroDeltaGamma=params[5];

		using cachedWeighter=cachedValueWeighter<DataType,Event,double>;
		cachedWeighter convPionFlux(&Event::conv_pion_event); // we get the pion component
		cachedWeighter convKaonFlux(&Event::conv_kaon_event); // we get the kaon component
		cachedWeighter promptFlux(&Event::prompt_event); // we get the prompt component
		cachedWeighter astroFlux(&Event::astro_event); // we get the astro component

		auto conventionalComponent = convNorm*(convPionFlux + piKRatio*convKaonFlux)
		                             *powerlawTiltWeighter<Event,DataType>(medianConvEnergy, CRDeltaGamma); // we sum them upp according to some pi/k ratio.

		auto astroComponent = astroNorm*(astroFlux)
		                             *powerlawTiltWeighter<Event,DataType>(medianAstroEnergy, astroDeltaGamma); // constructing the astrophysical component with a tilt

		auto promptComponent = promptNorm*promptFlux
		                       *powerlawTiltWeighter<Event,DataType>(medianPromptEnergy, CRDeltaGamma);

		return (conventionalComponent + promptComponent + astroComponent);
	}
};


//===================LLH function


const double PI_CONSTANT = 3.141592;
const double m2Tocm2 = 1.0e4;

const unsigned int neutrino_energy_index = 0;
const unsigned int coszenith_index = 1;
const unsigned int proxy_energy_index = 2;
const unsigned int neutrinoEnergyBins=280;
const unsigned int cosZenithBins=11;
const unsigned int energyProxyBins=50;
const unsigned int number_of_years = 2;

const double minFitEnergy = 4.0e2;
    //double maxFitEnergy = 2.0e4;
const double maxFitEnergy = 1.8e4;
const double minCosth = -1;
const double maxCosth = 0.2;

const bool quiet = false;
//const bool quiet = true;
const size_t evalThreads=1;
const std::vector<double> fitSeed {1.,0.,1.,1.,1.,0.};
// normalization, deltaGamma, pi/K ratio, prompt normalization, astrophysical normalization, astrophysical index

struct LLHWorkspace {
  std::deque<Event>& observed_events;

  nusquids::marray<double,3>& kaon_event_expectation;
  nusquids::marray<double,3>& pion_event_expectation;
  nusquids::marray<double,3>& prompt_event_expectation;
  nusquids::marray<double,3>&astro_event_expectation;

  IntegrateWorkspace& ws;

  AreaEdges& edges;
  AreaArray& areas;
  std::array<double, 2>& livetime;

  nuSQUIDSAtm<nuSQUIDSLV>* nus_kaon;
  nuSQUIDSAtm<nuSQUIDSLV>* nus_pion;
  nuSQUIDSAtm<nuSQUIDSLV>* nus_prompt;

  // legacy:
  multidim** convAtmosFlux;
  // fluxes use:
  multidim** convPionAtmosFlux;
  multidim** convKaonAtmosFlux;
  multidim** promptAtmosFlux;
  multidim** convDOMEffCorrection;
};

//not sure 2 --> 3
double llh(LLHWorkspace& ws, std::array<double, 3>& osc_params) {

    nusquids::marray<double,3>& kaon_event_expectation = ws.kaon_event_expectation;
    nusquids::marray<double,3>& pion_event_expectation = ws.pion_event_expectation;
    nusquids::marray<double,3>& prompt_event_expectation = ws.prompt_event_expectation;
    nusquids::marray<double,3>&astro_event_expectation = ws.astro_event_expectation;

    for(auto it = kaon_event_expectation.begin(); it < kaon_event_expectation.end(); it++)
        *it = 0.;
    for(auto it = pion_event_expectation.begin(); it < pion_event_expectation.end(); it++)
        *it = 0.;
    for(auto it = prompt_event_expectation.begin(); it < prompt_event_expectation.end(); it++)
        *it = 0.;
    for(auto it =astro_event_expectation.begin(); it <astro_event_expectation.end(); it++)
        *it = 0.;

    unsigned int indices[3],p,y;

#define USE_CHRIS_FLUX
#ifndef USE_CHRIS_FLUX
    // read nusquids calculated flux
    if(!quiet){
      std::cout << "Loading nuSQuIDS fluxes." << std::endl;
    }

    for(PTypes flavor : {NUMU,NUMUBAR}){
      for(unsigned int ei = 0; ei < neutrinoEnergyBins; ei++){
        for(unsigned int ci = 0; ci < cosZenithBins; ci++){
          double p_osc = GetAvgPOsc(ws.ws, osc_params, flavor,
                                                           ws.edges[y2010][flavor][coszenith_index][ci],
                                                           ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                           ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                           ws.edges[y2010][flavor][neutrino_energy_index][ei+1]);
          double kaon_integrated_flux = GetAveragedFlux(ws.ws, &ws.nus_kaon,flavor,
                                                         ws.edges[y2010][flavor][coszenith_index][ci],
                                                         ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                         ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                         ws.edges[y2010][flavor][neutrino_energy_index][ei+1]) * p_osc;
          double pion_integrated_flux = GetAveragedFlux(ws.ws, &ws.nus_pion,flavor,
                                                         ws.edges[y2010][flavor][coszenith_index][ci],
                                                         ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                         ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                         ws.edges[y2010][flavor][neutrino_energy_index][ei+1]) * p_osc;

          double p_osc_astro = GetAvgAstroPOsc(ws.ws, osc_params, flavor,
                                                           ws.edges[y2010][flavor][coszenith_index][ci],
                                                           ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                           ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                           ws.edges[y2010][flavor][neutrino_energy_index][ei+1]);
          double astro_integrated_flux = GetAveragedAstroFlux(ws.ws,flavor,
                                                         ws.edges[y2010][flavor][coszenith_index][ci],
                                                         ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                         ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                         ws.edges[y2010][flavor][neutrino_energy_index][ei+1]) * p_osc_astro;
          for(unsigned int pi = 0; pi < energyProxyBins; pi++){
            for(Year year : {y2010,y2011}){
                indices[0]=ei;
                indices[1]=ci;
                indices[2]=pi;
                p = (flavor == NUMU) ? 0 : 1;
                y = (year == y2010) ? 0 : 1;
                double solid_angle = 2.*PI_CONSTANT*(ws.edges[year][flavor][coszenith_index][ci+1]-ws.edges[year][flavor][coszenith_index][ci]);
                double DOM_eff_correction = 1.; // this correction is flux dependent, we will need to fix this.
                // double DOM_eff_correction =*index_multi(*convDOMEffCorrection[y],indices);
                kaon_event_expectation[year][ci][pi] += DOM_eff_correction*solid_angle*m2Tocm2*ws.livetime[year]*ws.areas[year][flavor][ei][ci][pi]*kaon_integrated_flux;
                pion_event_expectation[year][ci][pi] += DOM_eff_correction*solid_angle*m2Tocm2*ws.livetime[year]*ws.areas[year][flavor][ei][ci][pi]*pion_integrated_flux;
                prompt_event_expectation[year][ci][pi] += 0.; // no nusquids flux yet for prompt
                astro_event_expectation[year][ci][pi] += DOM_eff_correction*solid_angle*m2Tocm2*ws.livetime[year]*ws.areas[year][flavor][ei][ci][pi]*astro_integrated_flux;
            }
          }
        }
      }
    }
#else
        for(PTypes flavor : {NUMU,NUMUBAR}){
            for(unsigned int ci = 0; ci < cosZenithBins; ci++){
                for(unsigned int ei = 0; ei < neutrinoEnergyBins; ei++){
                  double p_osc = GetAvgPOsc(ws.ws, osc_params, flavor,
                                                           ws.edges[y2010][flavor][coszenith_index][ci],
                                                           ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                           ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                           ws.edges[y2010][flavor][neutrino_energy_index][ei+1]);
                  double p_osc_astro = GetAvgAstroPOsc(ws.ws, osc_params, flavor,
                                                                   ws.edges[y2010][flavor][coszenith_index][ci],
                                                                   ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                                   ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                                   ws.edges[y2010][flavor][neutrino_energy_index][ei+1]);
                  double astro_integrated_flux = GetAveragedAstroFlux(ws.ws,flavor,
                                                                 ws.edges[y2010][flavor][coszenith_index][ci],
                                                                 ws.edges[y2010][flavor][coszenith_index][ci+1],
                                                                 ws.edges[y2010][flavor][neutrino_energy_index][ei],
                                                                 ws.edges[y2010][flavor][neutrino_energy_index][ei+1]) * p_osc_astro;

                    for(unsigned int pi = 0; pi < energyProxyBins; pi++){
                      for(Year year : {y2010,y2011}){
                        //std::cout << ci << " " << edges[year][flavor][coszenith_index][ci] << std::endl;
                        indices[0]=ei;
                        indices[1]=ci;
                        indices[2]=pi;
                        p = (flavor == NUMU) ? 0 : 1;
                        y = (year == y2010) ? 0 : 1;
                        double solid_angle = 2.*PI_CONSTANT*(ws.edges[year][flavor][coszenith_index][ci+1]-ws.edges[year][flavor][coszenith_index][ci]);
                        double pion_fluxIntegral=*index_multi(*ws.convPionAtmosFlux[p],indices);
                        double kaon_fluxIntegral=*index_multi(*ws.convKaonAtmosFlux[p],indices);
                        double prompt_fluxIntegral=*index_multi(*ws.promptAtmosFlux[p],indices);
                        double DOM_eff_correction =*index_multi(*ws.convDOMEffCorrection[y],indices);
                        // chris does not separate between pions and kaon components. Lets just drop it all in the kaons.
                        // also chris has already included the solid angle factor in the flux
                        kaon_event_expectation[year][ci][pi] += m2Tocm2*ws.livetime[year]*ws.areas[year][flavor][ei][ci][pi]*DOM_eff_correction*kaon_fluxIntegral * p_osc;
                        pion_event_expectation[year][ci][pi] += m2Tocm2*ws.livetime[year]*ws.areas[year][flavor][ei][ci][pi]*DOM_eff_correction*pion_fluxIntegral * p_osc;
                        prompt_event_expectation[year][ci][pi] += m2Tocm2*ws.livetime[year]*ws.areas[year][flavor][ei][ci][pi]*DOM_eff_correction*prompt_fluxIntegral * p_osc;
                        astro_event_expectation[year][ci][pi] += DOM_eff_correction*solid_angle*m2Tocm2*ws.livetime[year]*ws.areas[year][flavor][ei][ci][pi]*astro_integrated_flux;
                        if (kaon_event_expectation[year][ci][pi] < 0) throw std::runtime_error("badness");
                      }
                  }
              }
          }
      }
#endif

    // to keep the code simple we are going to construct fake MC events
    // which weights are just the expectation value. The advantage of this is that
    // when we have the actual IceCube MC the following code can be reused. Love, CA.

    std::deque<Event> mc_events;

    for(Year year : {y2010,y2011}){
      for(unsigned int ci = 0; ci < cosZenithBins; ci++){
        for(unsigned int pi = 0; pi < energyProxyBins; pi++){
          // we are going to save the bin content and label it at the bin center
          mc_events.push_back(Event((ws.edges[year][flavor][proxy_energy_index][pi]+ws.edges[year][flavor][proxy_energy_index][pi+1])/2., // energy proxy bin center
                                    (ws.edges[year][flavor][coszenith_index][ci]+ws.edges[year][flavor][coszenith_index][ci+1])/2., // costh bin center
                                    year == y2010 ? 2010 : 2011, // year
                                    ws.kaon_event_expectation[year][ci][pi],  // amount of kaon component events
                                    ws.pion_event_expectation[year][ci][pi],  // amount of pion component events
	                                  ws.prompt_event_expectation[year][ci][pi],// amount of prompt component events
	                                  ws.astro_event_expectation[year][ci][pi]));// amount of astrophysical component events
        }
      }
    }
    //============================= end calculating event expectation  =============================//

    //============================= begin making histograms  =============================//
    if(!quiet){
      std::cout << "Making histograms." << std::endl;
    }
    // LLH problem and histogram
    // lets construct to weaver histograms
    using namespace phys_tools::histograms;
    using HistType = histogram<3,likelihood::entryStoringBin<std::reference_wrapper<const Event>>>;
    using phys_tools::histograms::amount;
    auto binner = [](HistType& h, const Event& e){
      h.add(e.energy_proxy,e.costh,e.year,amount(std::cref(e)));
    };

    /*{ // old axis construction. There is a bug when using FixedUserDefinedAxis and the LLH class. Ticket sent to CW.
    // set up data histogram // note that the energy proxy and costh bins are the same for both years
    marray<double,1> zenith_bin_edges = edges[y2010][flavor][coszenith_index];
    marray<double,1> energy_proxy_bin_edges = edges[y2010][flavor][proxy_energy_index];

    auto costh_ub_it = std::lower_bound(zenith_bin_edges.begin(),zenith_bin_edges.end(),maxCosth);
    auto costh_lb_it = std::upper_bound(zenith_bin_edges.begin(),zenith_bin_edges.end(),minCosth);
    auto ep_ub_it = std::lower_bound(energy_proxy_bin_edges.begin(),energy_proxy_bin_edges.end(),maxFitEnergy);
    auto ep_lb_it = std::upper_bound(energy_proxy_bin_edges.begin(),energy_proxy_bin_edges.end(),minFitEnergy);

    //std::cout << *costh_ub_it << " " << *costh_lb_it << std::endl;
    //std::cout << *ep_ub_it << " " << *ep_lb_it << std::endl;
    HistType data_hist(FixedUserDefinedAxis(costh_lb_it,costh_ub_it),
                       FixedUserDefinedAxis(ep_lb_it,ep_ub_it),
                       LinearAxis(2010,1));
    }*/

    // this magic number set the right bin edges
    HistType data_hist(LogarithmicAxis(0,0.1),LinearAxis(0,0.1),LinearAxis(2010,1));

    data_hist.getAxis(0)->setLowerLimit(minFitEnergy);
    data_hist.getAxis(0)->setUpperLimit(maxFitEnergy);
    data_hist.getAxis(1)->setLowerLimit(minCosth);
    data_hist.getAxis(1)->setUpperLimit(maxCosth);

    // fill in the histogram with the data
    bin(ws.observed_events,data_hist,binner);

    /*
    { // print axis edges
      std::cout << data_hist.range(0).first << " " << data_hist.range(0).second << std::endl;
      for(unsigned int i = 0; i < data_hist.getBinCount(0); i++)
        std::cout << data_hist.getBinEdge(0, i) << ' ';
      std::cout << std::endl;
      std::cout << data_hist.range(1).first << " " << data_hist.range(1).second << std::endl;
      for(unsigned int i = 0; i < data_hist.getBinCount(1); i++)
        std::cout << data_hist.getBinEdge(1, i) << ' ';
      std::cout << std::endl;
    }
    */

    // create MC histogram with the same binning as the data
    HistType sim_hist = makeEmptyHistogramCopy(data_hist);
    // fill in the histogram with the mc events
    bin(mc_events,sim_hist,binner);

    //============================= end making histograms  =============================//

    //============================= likelihood problem begins  =============================//
    // priors on the nuisance parameters
    if(!quiet){
      std::cout << "Constructing likelihood problem." << std::endl;
    }

    likelihood::UniformPrior positivePrior(0.0,std::numeric_limits<double>::infinity());
    likelihood::GaussianPrior normalizationPrior(1.,0.4);//0.4
    likelihood::GaussianPrior crSlopePrior(0.0,0.05);
    likelihood::GaussianPrior kaonPrior(1.0,0.1);
    likelihood::UniformPrior prompt_norm(0.0,std::numeric_limits<double>::infinity());
    likelihood::UniformPrior astro_norm(0.0,std::numeric_limits<double>::infinity());
    likelihood::UniformPrior astro_gamma(-0.5,0.5);

    auto priors=makePriorSet(normalizationPrior,crSlopePrior,kaonPrior,prompt_norm,astro_norm,astro_gamma);
    // construct a MC event reweighter
    DiffuseFitWeighterMaker DFWM;
    // construct likelihood problem
    // there are two numbers here. The first number is the number of histogram dimension, in this case 3.
    // The second number is the number of nuisance parameters, it is 3 for conventional-only,
    // and 5 for conventional+astro, and 6 for conventional+astro+prompt.
    auto prob=likelihood::makeLikelihoodProblem<std::reference_wrapper<const Event>,3,6>(data_hist, {sim_hist}, priors, {1.0}, likelihood::simpleDataWeighter(), DFWM, likelihood::poissonLikelihood(), fitSeed);
    prob.setEvaluationThreadCount(evalThreads);

    std::vector<double> seed=prob.getSeed();
    std::vector<unsigned int> fixedIndices = {}; // nuisance parameters that will be fixed
    paramFixSpec fixedParams;
    for(const auto pf : fixedParams.params){
      if(!quiet)
        std::cout << "Fitting with parameter " << pf.first << " fixed to " << pf.second << std::endl;
      seed[pf.first]=pf.second;
      fixedIndices.push_back(pf.first);
    }
    if(!quiet){
      std::cout << "Finding minima." << std::endl;
    }

    //std::cout << "evalllh " << std::endl;
    //std::cout << prob.evaluateLikelihood(seed) << std::endl;

    // minimize over the nuisance parameters
    fitResult fr = doFitLBFGSB(prob,seed,fixedIndices);

    if(!quiet){
      std::cout << "Fitted Hypothesis: ";
    for(unsigned int i=0; i<fr.params.size(); i++)
      std::cout << fr.params[i] << ' ';
    std::cout << std::setprecision(10) << fr.likelihood << std::setprecision(6) << ' ' << (fr.succeeded?"succeeded":"failed") << std::endl;
    }
    return fr.likelihood;//fr.params[2];//
}

//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//


int main(int argc, char** argv)
{
    if(not (argc == 7 or argc ==8)){
        std::cout << "Invalid number of arguments. The arguments should be given as follows: \n"
                     "1) Path to the effective area hdf5.\n"
                     "2) Path to the observed events file.\n"
                     "3) Path to Chris flux file with DOM efficiency correction.\n"
                     "4) Path to the kaon component flux.\n"
                     "5) Path to the pion component flux.\n"
                     "6) Path to the prompt component flux.\n"
                     "7) [optional] Path to output the event expectations.\n"
                     << std::endl;
        exit(1);
    }

    // default hypothesis
    std::array<double, 3> osc_params = {1.e-30, 1.0e-30, 1.0e-30}; // close to null
    //std::array<double, 3> osc_params = {1e-25, 1e-25, 1e-25};

    IntegrateWorkspace ws(5000);
    // paths
    std::string effective_area_filename = std::string(argv[1]);
    std::string data_filename = std::string(argv[2]);
    std::string chris_flux_filename = std::string(argv[3]); // also contains DOM efficiency correction
    std::string kaon_filename = std::string(argv[4]);
    std::string pion_filename = std::string(argv[5]);
    std::string prompt_filename = std::string(argv[6]);

    std::string output_file_str;
    if(argc==8) output_file_str = std::string(argv[7]);

    //============================== SETTINGS ===================================//
    //============================== SETTINGS ===================================//


    //============================== SETTINGS ===================================//
    //============================== SETTINGS ===================================//

    //============================= begin read data  =============================//
    if(!quiet){
      std::cout << "Reading events from data file." << std::endl;
    }
    marray<double,2> observed_data = quickread(data_filename);
    std::deque<Event> observed_events;
    for(unsigned int irow = 0; irow < observed_data.extent(0); irow++){
      observed_events.push_back(Event(observed_data[irow][0],// energy proxy
                                      observed_data[irow][1],// costh
                                      static_cast<unsigned int>(observed_data[irow][2])));// year
    }
    //============================= end read data  =============================//

    //============================= begin calculating event expectation  =============================//
    if(!quiet){
      std::cout << "Calculating event expectation using effective area." << std::endl;
    }

    AreaEdges edges;
    AreaArray areas = get_areas(std::string(effective_area_filename), edges);

    //std::cout << "Printing out the number of edges. Just a sanity check." << std::endl;
    //std::cout << edges[y2010][NUMU][neutrino_energy_index].extent(0) << std::endl;
    //std::cout << edges[y2010][NUMU][coszenith_index].extent(0) << std::endl;
    //std::cout << edges[y2010][NUMU][proxy_energy_index].extent(0) << std::endl;
    std::array<double, 2> livetime;
    livetime[y2010] = 2.7282e+07; // in seconds
    livetime[y2011] = 2.96986e+07; // in seconds

    nusquids::marray<double,3> kaon_event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    nusquids::marray<double,3> pion_event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    nusquids::marray<double,3> prompt_event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    nusquids::marray<double,3>astro_event_expectation{number_of_years,cosZenithBins,energyProxyBins};


    //not sure astro_event fit here
    LLHWorkspace llh_ws = {observed_events, kaon_event_expectation, pion_event_expectation, prompt_event_expectation, astro_event_expectation, ws, edges, areas, livetime, nullptr, nullptr, nullptr, nullptr};

#ifndef USE_CHRIS_FLUX
    nuSQUIDSAtm<nuSQUIDSLV> nus_kaon(kaon_filename);
    nuSQUIDSAtm<nuSQUIDSLV> nus_pion(pion_filename);
    nuSQUIDSAtm<nuSQUIDSLV> nus_prompt(prompt_filename);
    llh_ws.nus_kaon = &nus_kaon;
    llh_ws.nus_pion = &nus_pion;
    llh_ws.nus_prompt = &nus_prompt;
#else
    //these share the same binning in the first two dimensions
    const unsigned int histogramDims[3]={neutrinoEnergyBins,cosZenithBins,energyProxyBins};
    multidim convDOMEffCorrection2010=alloc_multi(3,histogramDims);
    multidim convDOMEffCorrection2011=alloc_multi(3,histogramDims);
    multidim* convDOMEffCorrection[2]={&convDOMEffCorrection2010,&convDOMEffCorrection2011};

    // reading the DOM efficiency correction
    hid_t file_id = H5Fopen(chris_flux_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    readDataSet(file_id, "/detector_correction/2010", convDOMEffCorrection2010.data);
    readDataSet(file_id, "/detector_correction/2011", convDOMEffCorrection2011.data);

    // reading Chris Combined Flux
    multidim convAtmosNuMu=alloc_multi(2,histogramDims);
    multidim convAtmosNuMuBar=alloc_multi(2,histogramDims);
    multidim* convAtmosFlux[2]={&convAtmosNuMu,&convAtmosNuMuBar};

    readDataSet(file_id, "/nu_mu/integrated_flux", convAtmosNuMu.data);
    readDataSet(file_id, "/nu_mu_bar/integrated_flux", convAtmosNuMuBar.data);

    H5Fclose(file_id);

    // reading pion flux
    hid_t pion_file_id = H5Fopen(pion_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    multidim convPionAtmosNuMu=alloc_multi(2,histogramDims);
    multidim convPionAtmosNuMuBar=alloc_multi(2,histogramDims);
    multidim* convPionAtmosFlux[2]={&convPionAtmosNuMu,&convPionAtmosNuMuBar};

    readDataSet(pion_file_id, "/nu_mu/integrated_flux", convPionAtmosNuMu.data);
    readDataSet(pion_file_id, "/nu_mu_bar/integrated_flux", convPionAtmosNuMuBar.data);

    H5Fclose(pion_file_id);

    // reading kaon flux
    hid_t Kaon_file_id = H5Fopen(kaon_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    multidim convKaonAtmosNuMu=alloc_multi(2,histogramDims);
    multidim convKaonAtmosNuMuBar=alloc_multi(2,histogramDims);
    multidim* convKaonAtmosFlux[2]={&convKaonAtmosNuMu,&convKaonAtmosNuMuBar};

    readDataSet(Kaon_file_id, "/nu_mu/integrated_flux", convKaonAtmosNuMu.data);
    readDataSet(Kaon_file_id, "/nu_mu_bar/integrated_flux", convKaonAtmosNuMuBar.data);

    H5Fclose(Kaon_file_id);

    // reading prompt flux
    hid_t prompt_file_id = H5Fopen(prompt_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    multidim promptAtmosNuMu=alloc_multi(2,histogramDims);
    multidim promptAtmosNuMuBar=alloc_multi(2,histogramDims);
    multidim* promptAtmosFlux[2]={&promptAtmosNuMu,&promptAtmosNuMuBar};

    readDataSet(Kaon_file_id, "/nu_mu/integrated_flux", promptAtmosNuMu.data);
    readDataSet(Kaon_file_id, "/nu_mu_bar/integrated_flux", promptAtmosNuMuBar.data);

    H5Fclose(prompt_file_id);

    // put that in the box

    llh_ws.convDOMEffCorrection = convDOMEffCorrection;
    llh_ws.convAtmosFlux = convAtmosFlux;
    llh_ws.convPionAtmosFlux = convPionAtmosFlux;
    llh_ws.convKaonAtmosFlux = convKaonAtmosFlux;
    llh_ws.promptAtmosFlux = promptAtmosFlux;
#endif

//#######################

    // do only one parameter
    std::cout << llh(llh_ws,osc_params) << std::endl;

    /*
    // loop of death
    double llh_val[100][100][100] = {0};
    for (int i = 0; i < 100; i++) {
      for (int j = 0; j < 100; j++) {
        for (int k = 0; k < 100; k++) {
          osc_params[0] =  std::pow(10, -28 + 6.0*i/100.0);
          osc_params[1] =  std::pow(10, -28 + 6.0*j/100.0);
          osc_params[2] =  std::pow(10, -28 + 6.0*j/100.0);
          llh_val[i][j][k] = llh(llh_ws, osc_params);
          //std::cout << llh_val[i][j] << std::endl;
        }
      }
    }
    //std::cout << llh_val << std::endl;

    if (argc > 6) {
      std::ofstream output_file(output_file_str, std::ios::out | std::ios::binary);
      output_file.write((char*)llh_val, sizeof(llh_val));
      output_file.close();
    }
    */

    //============================= likelihood problem ends =============================//

    if(!quiet){
      std::cout << "Saving expectation." << std::endl;
    }

    return 0;
}
