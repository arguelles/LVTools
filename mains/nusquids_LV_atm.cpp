#include <iostream>
#include <vector>
#include <string>
#include <nuSQuIDS/marray.h>
#include <nuSQuIDS/nuSQuIDS.h>
#include <nuSQuIDS/tools.h>
#include "nusquids_LV.h"

using namespace nusquids;

int main()
{
  // atmospheric model name
  std::string modelname = "HondaGaisser";
  //std::string meson = "pion";
  std::string meson = "kaon";
  marray<double,2> input_flux = quickread(std::string("./flux_models/") + "initial_"+meson+"_atmopheric_" + modelname + ".dat");

  // number of neutrino flavors
  const unsigned int numneu = 3;
  squids::Const units;
  std::cout << "Begin creating nuSQuIDS Atmospheric Object" << std::endl;
  nuSQUIDSAtm<nuSQUIDSLV> nus(-1,0.2,40,1.e2*units.GeV,1.e6*units.GeV,150,numneu,both,true,true);
  std::cout << "End creating nuSQuIDS Atmospheric Object" << std::endl;

  // set mixing angles and masses
  nus.Set_MixingAngle(0,1,0.563942);
  nus.Set_MixingAngle(0,2,0.154085);
  nus.Set_MixingAngle(1,2,0.785398);

  nus.Set_SquareMassDifference(1,7.65e-05);
  nus.Set_SquareMassDifference(2,0.00247);

  nus.Set_TauRegeneration(true);
  // now we will set the LV parameters
  // this is c_emu only
  //LVParameters c_test {gsl_complex_rect(1.0e-27,0),GSL_COMPLEX_ZERO};
  // this is c_mutau only
  double c_mutau = 1.0e-27;
//  c_mutau = 0.;
  LVParameters c_parameters {GSL_COMPLEX_ZERO,gsl_complex_rect(c_mutau,0)};
  for(auto & ns : nus.GetnuSQuIDS()){
    ns.Set_LV_CMatrix(c_parameters);
  }

  // setup integration settings
  nus.Set_h_max( 100.0*units.km );
  nus.Set_rel_error(1.0e-20);
  nus.Set_abs_error(1.0e-20);

  std::cout << "Begin setting up initial flux" << std::endl;
  // construct the initial state
  marray<double,4> inistate {nus.GetNumCos(),nus.GetNumE(),2,numneu};
  std::fill(inistate.begin(),inistate.end(),0);

  marray<double,1> cos_range = nus.GetCosthRange();
  marray<double,1> e_range = nus.GetERange();

  for ( int ci = 0 ; ci < nus.GetNumCos(); ci++){
    for ( int ei = 0 ; ei < nus.GetNumE(); ei++){
      double enu = e_range[ei];
      double cth = cos_range[ci];
      // we set the muon components and assume the others to be cero
      // neutrino muon
      inistate[ci][ei][0][1] = input_flux[ci*e_range.size() + ei][2];
      // antineutrino muon
      inistate[ci][ei][1][1] = input_flux[ci*e_range.size() + ei][3];
    }
  }

  // set the initial state
  nus.Set_initial_state(inistate,flavor);
  std::cout << "End setting up initial flux" << std::endl;

  nus.Set_ProgressBar(true);
  std::cout << "Begin calculation" << std::endl;
  nus.EvolveState();
  std::cout << std::endl;
  std::cout << "End calculation" << std::endl;

  // we can save the current state in HDF5 format
  // for future use.
  nus.WriteStateHDF5("./nusquids_LV_"+meson+"_"+modelname+"_"+std::to_string(c_mutau)+".hdf5");

  return 0;
}
