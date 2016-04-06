#include <vector>
#include <deque>
#include <iostream>
#include <nuSQuIDS/nuSQuIDS.h>
#include "nusquids_LV.h"
#include "get_eff_area.h"
#include <gsl/gsl_integration.h>

// chris weaver tools
#include <PhysTools/histogram.h>
#include "event.h"
#include "gsl_integrate_wrap.h"
#include "chris_reads.h"

//#define USE_CHRIS_FLUX

using namespace nusquids;

double GetAveragedFlux(nuSQUIDSAtm<nuSQUIDSLV> * nus,PTypes flavor, double costh, double enu_min, double enu_max) {
    if(enu_min > enu_max)
      throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");
    if (enu_min >= 1.0e6)
      return 0.;
    if (enu_max <= 1.0e2)
      return 0.;
    double GeV = 1.0e9;
    if (flavor == NUMU){
        return integrate([&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,0));},enu_min,enu_max);
    } else {
        return integrate([&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,1));},enu_min,enu_max);
    }
}

double GetAveragedFlux(nuSQUIDSAtm<nuSQUIDSLV> * nus,PTypes flavor, double costh_min, double costh_max, double enu_min, double enu_max) {
  return (GetAveragedFlux(nus,flavor,costh_max,enu_min,enu_max) + GetAveragedFlux(nus,flavor,costh_min,enu_min,enu_max))/2.;
  //return GetAveragedFlux(nus,flavor,(costh_max+costh_min)/2.,enu_min,enu_max);
}

//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//


int main(int argc, char** argv)
{
    if(argc < 4){
        std::cout << "Invalid number of arguments. The arguments should be given as follows: \n"
                     "1) Path to the effective area hdf5.\n"
                     "2) Path to the kaon component nusquids calculated flux.\n"
                     "3) Path to the pion component nusquids calculated flux.\n"
                     "4) [optional] Path to output the event expectations.\n"
                     << std::endl;
        exit(1);
    }

    // read nusquids calculated flux
    nuSQUIDSAtm<nuSQUIDSLV> nus_kaon((std::string(argv[2])));
    nuSQUIDSAtm<nuSQUIDSLV> nus_pion((std::string(argv[3])));

    //============================= begin calculating event expectation  =============================//


    const unsigned int neutrino_energy_index = 0;
    const unsigned int coszenith_index = 1;
    const unsigned int proxy_energy_index = 2;
    AreaEdges edges;
    AreaArray areas = get_areas(std::string(argv[1]), edges);

    //std::cout << "Printing out the number of edges. Just a sanity check." << std::endl;
    //std::cout << edges[y2010][NUMU][neutrino_energy_index].extent(0) << std::endl;
    //std::cout << edges[y2010][NUMU][coszenith_index].extent(0) << std::endl;
    //std::cout << edges[y2010][NUMU][proxy_energy_index].extent(0) << std::endl;

    const unsigned int neutrinoEnergyBins=280;
    const unsigned int cosZenithBins=11;
    const unsigned int energyProxyBins=50;

    const unsigned int number_of_years = 2;
    const double PI_CONSTANT = 3.141592;
    const double m2Tocm2 = 1.0e4;
    std::map<Year,double> livetime {{y2010,2.7282e+07},{y2011,2.96986e+07}}; // in seconds

    nusquids::marray<double,3> kaon_event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    for(auto it = kaon_event_expectation.begin(); it < kaon_event_expectation.end(); it++)
        *it = 0.;

    nusquids::marray<double,3> pion_event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    for(auto it = pion_event_expectation.begin(); it < pion_event_expectation.end(); it++)
        *it = 0.;

    const unsigned int histogramDims[3]={neutrinoEnergyBins,cosZenithBins,energyProxyBins};
    multidim convDOMEffCorrection2010=alloc_multi(3,histogramDims);
    multidim convDOMEffCorrection2011=alloc_multi(3,histogramDims);
    multidim* convDOMEffCorrection[2]={&convDOMEffCorrection2010,&convDOMEffCorrection2011};

    hid_t file_id = H5Fopen("conventional_flux.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    readDataSet(file_id, "/detector_correction/2010", convDOMEffCorrection2010.data);
    readDataSet(file_id, "/detector_correction/2011", convDOMEffCorrection2011.data);
    H5Fclose(file_id);

    unsigned int indices[3],p,y;
#ifndef USE_CHRIS_FLUX
    for(Year year : {y2010,y2011}){
        for(PTypes flavor : {NUMU,NUMUBAR}){
            for(unsigned int ci = 0; ci < cosZenithBins; ci++){
                for(unsigned int pi = 0; pi < energyProxyBins; pi++){
                    for(unsigned int ei = 0; ei < neutrinoEnergyBins; ei++){
                        indices[0]=ei;
                        indices[1]=ci;
                        indices[2]=pi;
                        p = (flavor == NUMU) ? 0 : 1;
                        y = (year == y2010) ? 0 : 1;
                        double solid_angle = 2.*PI_CONSTANT*(edges[year][flavor][coszenith_index][ci+1]-edges[year][flavor][coszenith_index][ci]);
                        double DOM_eff_correction = 1.; // this correction is flux dependent, we will need to fix this.
                        // double DOM_eff_correction =*index_multi(*convDOMEffCorrection[y],indices);
                        kaon_event_expectation[year][ci][pi] += DOM_eff_correction*solid_angle*m2Tocm2*livetime[year]*areas[year][flavor][ei][ci][pi]*GetAveragedFlux(&nus_kaon,flavor,
                                                                                                                       edges[year][flavor][coszenith_index][ci],
                                                                                                                       edges[year][flavor][coszenith_index][ci+1],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei+1]);
                        pion_event_expectation[year][ci][pi] += DOM_eff_correction*solid_angle*m2Tocm2*livetime[year]*areas[year][flavor][ei][ci][pi]*GetAveragedFlux(&nus_pion,flavor,
                                                                                                                       edges[year][flavor][coszenith_index][ci],
                                                                                                                       edges[year][flavor][coszenith_index][ci+1],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei+1]);
                    }
                }
            }
        }
    }
#else
    //these share the same binning in the first two dimensions
    multidim convAtmosNuMu=alloc_multi(2,histogramDims);
    multidim convAtmosNuMuBar=alloc_multi(2,histogramDims);
    multidim* convAtmosFlux[2]={&convAtmosNuMu,&convAtmosNuMuBar};

    file_id = H5Fopen("conventional_flux.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

    readDataSet(file_id, "/nu_mu/integrated_flux", convAtmosNuMu.data);
    readDataSet(file_id, "/nu_mu_bar/integrated_flux", convAtmosNuMuBar.data);

    H5Fclose(file_id);

    //unsigned int indices[3],p,y;
    for(Year year : {y2010,y2011}){
        for(PTypes flavor : {NUMU,NUMUBAR}){
            for(unsigned int ci = 0; ci < cosZenithBins; ci++){
                for(unsigned int pi = 0; pi < energyProxyBins; pi++){
                    for(unsigned int ei = 0; ei < neutrinoEnergyBins; ei++){
                        //std::cout << ci << " " << edges[year][flavor][coszenith_index][ci] << std::endl;
                        indices[0]=ei;
                        indices[1]=ci;
                        indices[2]=pi;
                        p = (flavor == NUMU) ? 0 : 1;
                        y = (year == y2010) ? 0 : 1;
                        double fluxIntegral=*index_multi(*convAtmosFlux[p],indices);
                        double DOM_eff_correction =*index_multi(*convDOMEffCorrection[y],indices);
                        //double DOM_eff_correction = 1.;
                        // chris does not separate between pions and kaon components. Lets just drop it all in the kaons.
                        // also chris has already included the solid angle factor in the flux
                        kaon_event_expectation[year][ci][pi] += m2Tocm2*livetime[year]*areas[year][flavor][ei][ci][pi]*DOM_eff_correction*fluxIntegral;
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
          mc_events.push_back(Event((edges[year][flavor][proxy_energy_index][pi]+edges[year][flavor][proxy_energy_index][pi+1])/2., // energy proxy bin center
                                    (edges[year][flavor][coszenith_index][ci]+edges[year][flavor][coszenith_index][ci+1])/2., // costh bin center
                                    year == y2010 ? 2010 : 2011, // year
                                    kaon_event_expectation[year][ci][pi], // amount of kaon component events
                                    pion_event_expectation[year][ci][pi])); // amount of pion component events
        }
      }
    }

    //============================= end calculating event expectation  =============================//
    std::string output_file_str;;
    if(argc > 4)
      output_file_str=std::string(argv[4]);
    else
      output_file_str=std::string("./expectation.dat");

    double pikr = 1.1;
    std::ofstream output_file(output_file_str);
    for(auto event : mc_events){
      //std::cout << event.conv_pion_event << " " << event.conv_kaon_event << std::endl;
      output_file << event.energy_proxy << " " << event.costh << " " << event.year << " " << event.conv_kaon_event << " " << event.conv_pion_event << " " << event.conv_pion_event + pikr*event.conv_kaon_event << std::endl;
    }
    output_file.close();

    return 0;
}
