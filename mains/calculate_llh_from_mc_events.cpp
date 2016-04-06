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
#include "event.h"

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
  minimizer.addParameter(seed[0],.01,0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[1],.01,0.0);
  minimizer.addParameter(seed[2],.01,0.0);

  for(auto idx : indicesToFix)
    minimizer.fixParameter(idx);

  minimizer.setChangeTolerance(1e-5);
  minimizer.setHistorySize(20);

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
	static constexpr double medianConvEnergy=2020;
	//static constexpr double medianPromptEnergy=7887;
public:
	DiffuseFitWeighterMaker()
	{}

	template<typename DataType>
	std::function<DataType(const Event&)> operator()(const std::vector<DataType>& params) const{
    // check that we are getting the right number of nuisance parameters
		assert(params.size()==3);
		//unpack things so we have legible names
		DataType convNorm=params[0];
		DataType CRDeltaGamma=params[1];
		DataType piKRatio=params[2];

		using cachedWeighter=cachedValueWeighter<DataType,Event,double>;
		cachedWeighter convPionFlux(&Event::conv_pion_event); // we get the pion component
		cachedWeighter convKaonFlux(&Event::conv_kaon_event); // we get the kaon component

		auto conventionalComponent = convNorm*(convPionFlux + piKRatio*convKaonFlux)
		                             *powerlawTiltWeighter<Event,DataType>(medianConvEnergy, CRDeltaGamma); // we sum them upp according to some pi/k ratio.

    /*
     * We will deal with the prompt and astrophysical components later. CA
		auto promptComponent = promptNorm*promptFlux
		                       *powerlawTiltWeighter<Event,DataType>(medianPromptEnergy, CRDeltaGamma);

    */

		return (conventionalComponent);
	}
};

//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//


int main(int argc, char** argv)
{
    if(argc < 4){
        std::cout << "Invalid number of arguments. The arguments should be given as follows: \n"
                     "1) Path to the effective area hdf5.\n"
                     "2) Path to the observed events file.\n"
                     "3) Path to expectation histogram.\n"
                     "4) [optional] Path to output the event expectations.\n"
                     << std::endl;
        exit(1);
    }

    //============================== SETTINGS ===================================//
    //============================== SETTINGS ===================================//

    bool quiet = false;
    size_t evalThreads=1;
    std::vector<double> fitSeed {1.,0.,1.}; // normalization, deltaGamma, pi/K ratio
    paramFixSpec fixedParams;
    double minFitEnergy = 4.0e2;
    double maxFitEnergy = 2.0e4;
    double minCosth = -1;
    double maxCosth = 0.2;

    //============================== SETTINGS ===================================//
    //============================== SETTINGS ===================================//

    // here we need the bin edges which are on the effective area file
    // we are going to use the edges to make the histograms
    const unsigned int neutrino_energy_index = 0;
    const unsigned int coszenith_index = 1;
    const unsigned int proxy_energy_index = 2;
    AreaEdges edges;
    AreaArray areas = get_areas(std::string(argv[1]), edges);

    //============================= begin read data  =============================//
    if(!quiet){
      std::cout << "Reading events from data file." << std::endl;
    }
    marray<double,2> observed_data = quickread(std::string(argv[2]));
    std::deque<Event> observed_events;
    for(unsigned int irow = 0; irow < observed_data.extent(0); irow++){
      observed_events.push_back(Event(observed_data[irow][0],// energy proxy
                                      observed_data[irow][1],// costh
                                      static_cast<unsigned int>(observed_data[irow][2])));// year
    }
    //============================= end read data  =============================//

    //============================= begin loading event expectation =============================//
    if(!quiet){
      std::cout << "Reading expectation from file." << std::endl;
    }
    marray<double,2> expectation_events = quickread(std::string(argv[3]));
    std::deque<Event> mc_events;

    for(unsigned int irow = 0; irow < observed_data.extent(0); irow++){
      mc_events.push_back(Event(expectation_events[irow][0], // energy proxy bin center
                                expectation_events[irow][1], // costh bin center
                                expectation_events[irow][2], // year
                                expectation_events[irow][3], // kaon component
                                expectation_events[irow][4])); // pion component
    }

    //============================= end loading event expectation  =============================//

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
    // set up data histogram // note that the energy proxy and costh bins are the same for both years
    marray<double,1> zenith_bin_edges = edges[y2010][flavor][coszenith_index];
    marray<double,1> energy_proxy_bin_edges = edges[y2010][flavor][proxy_energy_index];

    auto costh_ub_it = std::lower_bound(zenith_bin_edges.begin(),zenith_bin_edges.end(),maxCosth);
    auto costh_lb_it = std::upper_bound(zenith_bin_edges.begin(),zenith_bin_edges.end(),minCosth);
    auto ep_ub_it = std::lower_bound(energy_proxy_bin_edges.begin(),energy_proxy_bin_edges.end(),maxFitEnergy);
    auto ep_lb_it = std::upper_bound(energy_proxy_bin_edges.begin(),energy_proxy_bin_edges.end(),minFitEnergy);

    std::cout << *costh_ub_it << " " << *costh_lb_it << std::endl;
    std::cout << *ep_ub_it << " " << *ep_lb_it << std::endl;

    HistType data_hist(FixedUserDefinedAxis(costh_lb_it,costh_ub_it),
                       FixedUserDefinedAxis(ep_lb_it,ep_ub_it),
                       LinearAxis(2010,1));

    // fill in the histogram with the data
    bin(observed_events,data_hist,binner);

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
    likelihood::GaussianPrior normalizationPrior(1.,0.4);
    likelihood::GaussianPrior crSlopePrior(0.0,0.05);
    likelihood::GaussianPrior kaonPrior(1.0,0.1);

    auto priors=makePriorSet(normalizationPrior,crSlopePrior,kaonPrior);
    // construct a MC event reweighter
    DiffuseFitWeighterMaker DFWM;
    // construct likelihood problem
    // there are two numbers here. The first number is the number of histogram dimension
    // in this case 3. The second number is the number of nuisance parameters, also 3.
    auto prob=likelihood::makeLikelihoodProblem<std::reference_wrapper<const Event>,3,3>(data_hist, {sim_hist}, priors, {1.0}, likelihood::simpleDataWeighter(), DFWM, likelihood::poissonLikelihood(), fitSeed);
    prob.setEvaluationThreadCount(evalThreads);

    std::vector<double> seed=prob.getSeed();
    std::vector<unsigned int> fixedIndices;
    for(const auto pf : fixedParams.params){
      if(!quiet)
        std::cout << "Fitting with parameter " << pf.first << " fixed to " << pf.second << std::endl;
      seed[pf.first]=pf.second;
      fixedIndices.push_back(pf.first);
    }
    if(!quiet){
      std::cout << "Finding minima." << std::endl;
    }
    // minimize over the nuisance parameters
    fitResult fr = doFitLBFGSB(prob,seed,fixedIndices);

    if(!quiet)
      std::cout << "Fitted Hypothesis: ";
    for(unsigned int i=0; i<fr.params.size(); i++)
      std::cout << fr.params[i] << ' ';
    std::cout << std::setprecision(10) << fr.likelihood << std::setprecision(6) << ' ' << (fr.succeeded?"succeeded":"failed") << std::endl;

    //============================= likelihood problem ends =============================//

    if(!quiet){
      std::cout << "Saving expectation." << std::endl;
    }

    std::string output_file_str;;
    if(argc > 4)
      output_file_str=std::string(argv[4]);
    else
      output_file_str=std::string("./expectation.dat");

    std::ofstream output_file(output_file_str);
    auto weighter = DFWM(fr.params);
    for(auto event : mc_events){
      output_file << event.energy_proxy << " " << event.costh << " " << event.year << " " << event.conv_kaon_event << " " << event.conv_pion_event << " " << weighter(event) << std::endl;
    }
    output_file.close();

    return 0;
}
