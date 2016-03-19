#include <vector>
#include <iostream>
#include <nuSQuIDS/nuSQuIDS.h>
#include "nusquids_LV.h"
#include "get_eff_area.h"
#include <gsl/gsl_integration.h>

using namespace nusquids;

template<typename FunctionType>
double integrate(FunctionType f, double a, double b){
    double (*wrapper)(double,void*)=[](double x, void* params){
        FunctionType& f=*static_cast<FunctionType*>(params);
        return(f(x));
    };

    gsl_integration_workspace* ws=gsl_integration_workspace_alloc(5000);
    double result, error;
    gsl_function F;
    F.function = wrapper;
    F.params = &f;

    gsl_integration_qags(&F, a, b, 0, 1e-7, 5000, ws, &result, &error);
    gsl_integration_workspace_free(ws);

    return(result);
}

double GetAveragedFlux(nuSQUIDSAtm<nuSQUIDSLV> * nus,PTypes flavor, double costh, double enu_min, double enu_max) {
    if(enu_min > enu_max)
      throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");
    if (enu_min >= 1.0e6)
      return 0.;
    if (enu_max <= 1.0e2)
      return 0.;
    double GeV = 1.0e9;
    if (flavor == NUMU){
        return integrate([&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,0));},enu_min,enu_max)/(enu_max-enu_min);
    } else {
        return integrate([&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,1));},enu_min,enu_max)/(enu_max-enu_min);
    }
}

int main(int argc, char** argv)
{
    if(argc != 3){
        std::cout << "Invalid number of arguments. First argument\
                      should be effective are path. Second argument\
                      should be a nuSQuIDSAtm<nuSQuIDSLV> file." << std::endl;
        exit(1);
    }

    nuSQUIDSAtm<nuSQUIDSLV> nus((std::string(argv[2])));
    const unsigned int neutrino_energy_index = 0;
    const unsigned int coszenith_index = 1;
    const unsigned int proxy_energy_index = 2;
    AreaEdges edges;
    AreaArray areas = get_areas(std::string(argv[1]), edges);

    std::cout << "Printing out the number of edges. Just a sanity check." << std::endl;
    std::cout << edges[y2010][NUMU][neutrino_energy_index].extent(0) << std::endl;
    std::cout << edges[y2010][NUMU][coszenith_index].extent(0) << std::endl;
    std::cout << edges[y2010][NUMU][proxy_energy_index].extent(0) << std::endl;

    const unsigned int neutrinoEnergyBins=280;
    const unsigned int cosZenithBins=11;
    const unsigned int energyProxyBins=50;
    const unsigned int number_of_years = 2;
    const double pi = 3.141592;
    const double m2Tocm2 = 1.0e4;
    std::map<Year,double> livetime {{y2010,2.7282e+07},{y2011,2.96986e+07}}; // in seconds

    nusquids::marray<double,3> event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    for(auto it = event_expectation.begin(); it < event_expectation.end(); it++)
        *it = 0.;

    for(Year year : {y2010,y2011}){
        for(PTypes flavor : {NUMU,NUMUBAR}){
            for(unsigned int ci = 0; ci < cosZenithBins-1; ci++){
                for(unsigned int pi = 0; pi < energyProxyBins-1; pi++){
                    for(unsigned int ei = 0; ei < neutrinoEnergyBins-1; ei++){
                        double solid_angle = 2.*pi*(edges[year][flavor][coszenith_index][ci+1]-edges[year][flavor][coszenith_index][ci]);
                        std::cout << edges[year][flavor][coszenith_index][ci+1] << " " << edges[year][flavor][coszenith_index][ci] << std::endl;
                        std::cout << edges[year][flavor][neutrino_energy_index][ei+1] << " " << edges[year][flavor][neutrino_energy_index][ei] << std::endl;
                        event_expectation[year][ci][pi] += solid_angle*m2Tocm2*livetime[year]*areas[year][flavor][ei][ci][pi]*GetAveragedFlux(&nus,flavor,
                                                                                                                       edges[year][flavor][coszenith_index][ci],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei+1]);
                    }
                }
            }
        }
    }

    return 0;
}
