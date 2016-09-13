#include <vector>
#include <iostream>
#include <nuSQuIDS/nuSQuIDS.h>
#include "nusquids_LV.h"

using namespace nusquids;

int main()
{
  // atmospheric model name
  std::string modelname = "HondaGaisser";
  std::string meson = "pion";
  marray<double,2> input_flux = quickread(std::string("./flux_models/") + "initial_"+meson+"_atmopheric_" + modelname + ".dat");

  // setting up nusquids object
  squids::Const units;
  std::cout << "Begin creating nuSQuIDS Atmospheric Object" << std::endl;
  nuSQUIDSLV nus(logspace(1.e2*units.GeV,1.e6*units.GeV,150),3,both,true);
  std::cout << "End creating nuSQuIDS Atmospheric Object" << std::endl;

  double phi = acos(-1.);
  std::shared_ptr<EarthAtm> earth_atm = std::make_shared<EarthAtm>();
  std::shared_ptr<EarthAtm::Track> track_atm = std::make_shared<EarthAtm::Track>(phi);

  nus.Set_Body(earth_atm);
  nus.Set_Track(track_atm);

  // set mixing angles and masses
  nus.Set_MixingAngle(0,1,0.563942);
  nus.Set_MixingAngle(0,2,0.154085);
  nus.Set_MixingAngle(1,2,0.785398);

  nus.Set_SquareMassDifference(1,7.65e-05);
  nus.Set_SquareMassDifference(2,0.00247);

  // activate tau regeneration
  nus.Set_TauRegeneration(true);
  // now we will set the LV parameters
  // this is c_emu only
  //LVParameters c_test {gsl_complex_rect(1.0e-27,0),GSL_COMPLEX_ZERO};
  // this is c_mutau only
  double c_mutau = 1.0e-27;
  //double c_mutau = 0.;
  LVParameters c_parameters {GSL_COMPLEX_ZERO,gsl_complex_rect(c_mutau,0)};

  nus.Set_LV_CMatrix(c_parameters);

  // setup integration settings
  nus.Set_h_max( 100.0*units.km );
  nus.Set_PositivityConstrain(true);
  nus.Set_PositivityConstrainStep(300.0*units.km);
  nus.Set_rel_error(1.0e-25);
  nus.Set_abs_error(1.0e-25);

  marray<double,1> E_range = nus.GetERange();

  // construct the initial state
  marray<double,3> inistate({150,2,3});
  // given the file structure ci = 0 is cos(th) = -1
  unsigned int ci = 0;
  for ( int ei = 0 ; ei < inistate.extent(0); ei++){
      for ( int j = 0; j < inistate.extent(1); j++){
        for ( int k = 0; k < inistate.extent(2); k++){
          // initialze muon state
          //inistate[i][j][k] = (k == 1) ? 1. : 0.0;
          inistate[ei][0][1] = input_flux[ci*E_range.size() + ei][2];
          inistate[ei][1][1] = input_flux[ci*E_range.size() + ei][3];
        }
      }
  }

  // set the initial state
  nus.Set_initial_state(inistate,flavor);

  nus.Set_ProgressBar(true);
  nus.EvolveState();
  std::cout << std::endl;

  // we can save the current state in HDF5 format
  // for future use.
  nus.WriteStateHDF5("./nusquids_LV_both.hdf5");
  nus.dump_probabilities();

  return 0;
}
