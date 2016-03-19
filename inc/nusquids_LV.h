#ifndef nusquidslv_H
#define nusquidslv_H

#include <vector>
#include <iostream>
#include <nuSQuIDS/nuSQuIDS.h>


namespace nusquids {

struct LVParameters {
  gsl_complex c_emu;
  gsl_complex c_mutau;
};

class nuSQUIDSLV: public nuSQUIDS {
  private:
    bool lv_parameters_set = false;
    LVParameters c_params;
    squids::SU_vector LVP;
    std::vector<squids::SU_vector> LVP_evol;

    void AddToPreDerive(double x){
      if(!lv_parameters_set)
        throw std::runtime_error("LV parameters not set");
      for(int ei = 0; ei < ne; ei++){
        // asumming same mass hamiltonian for neutrinos/antineutrinos
        squids::SU_vector h0 = H0(E_range[ei],0);
        LVP_evol[ei] = LVP.Evolve(h0,(x-Get_t_initial()));
      }
    }

    void AddToWriteHDF5(hid_t hdf5_loc_id) const {
      // here we write the new parameters to be saved in the HDF5 file
      hsize_t dim[1]{1};
      hid_t lv = H5LTmake_dataset(hdf5_loc_id,"c_values",1,dim,H5T_NATIVE_DOUBLE,0);
      //hid_t lv = H5Dcreate(hdf5_loc_id, "c_values", H5T_NATIVE_INT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      //hid_t lv = H5Gcreate(hdf5_loc_id, "c_values", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      double cmutaur = GSL_REAL(c_params.c_mutau);
      double cmutaui = GSL_IMAG(c_params.c_mutau);
      H5LTset_attribute_double(hdf5_loc_id,"c_values","c_mu_tau_real" ,&(cmutaur),1);
      H5LTset_attribute_double(hdf5_loc_id,"c_values","c_mu_tau_imag" ,&(cmutaui),1);
      double cemur = GSL_REAL(c_params.c_emu);
      double cemui = GSL_IMAG(c_params.c_emu);
      H5LTset_attribute_double(hdf5_loc_id,"c_values","c_e_mu_real" ,&(cemur),1);
      H5LTset_attribute_double(hdf5_loc_id,"c_values","c_e_mu_imag" ,&(cemui),1);
    }

    void AddToReadHDF5(hid_t hdf5_loc_id){
      double cmutaur,cmutaui,cemur,cemui;
      // here we read the new parameters now saved in the HDF5 file
      //hid_t lv = H5Gopen(hdf5_loc_id, "c_values", H5P_DEFAULT);
      //H5Gclose(lv);
      H5LTget_attribute_double(hdf5_loc_id,"c_values","c_mu_tau_real" ,&(cmutaur));
      H5LTget_attribute_double(hdf5_loc_id,"c_values","c_mu_tau_imag" ,&(cmutaui));
      H5LTget_attribute_double(hdf5_loc_id,"c_values","c_e_mu_real" ,&(cemur));
      H5LTget_attribute_double(hdf5_loc_id,"c_values","c_e_mu_imag" ,&(cemui));
      c_params = {gsl_complex_rect(cemur,cemui),gsl_complex_rect(cmutaur,cmutaui)};
      Set_LV_CMatrix(c_params);
    }


    squids:: SU_vector HI(unsigned int ie,unsigned int irho) const {
      double ye = body->ye(*track);
      double density = body->density(*track);

      double CC = params.sqrt2*params.GF*params.Na*pow(params.cm,-3)*density*ye;
      double NC;

      if (ye < 1.0e-10){
        NC = params.sqrt2*params.GF*params.Na*pow(params.cm,-3)*density;
      }
      else {
        NC = CC*(-0.5*(1.0-ye)/ye);
      }

      // construct standard potential in flavor basis
      squids::SU_vector potential = (CC+NC)*evol_b1_proj[irho][0][ie];
      potential += (NC)*(evol_b1_proj[irho][1][ie]);
      potential += (NC)*(evol_b1_proj[irho][2][ie]);

      if ((irho == 1 and NT==both) or NT==antineutrino){
          // antineutrino matter potential flips sign
          potential *= (-1.0);
      }

      // ================= HERE WE ADD THE NEW PHYSICS ===================
      potential += (E_range[ie])*LVP_evol[ie]; // <- super important line
      // ================= HERE WE ADD THE NEW PHYSICS ===================

      if (basis == mass){
        potential += H0_array[ie];
      }
      return potential;
    }
  public:
    nuSQUIDSLV() {};
    nuSQUIDSLV(double Emin_,double Emax_,int Esize_,int numneu_,NeutrinoType NT_,
         bool elogscale_,bool iinteraction_):
          nuSQUIDS(Emin_,Emax_,Esize_,numneu_,NT_,elogscale_,iinteraction_)
    {
      // just allocate some matrices
       LVP_evol.resize(ne);
       for(int ei = 0; ei < ne; ei++){
         LVP_evol[ei] = squids::SU_vector(nsun);
       }
    }

    void Set_LV_CMatrix(LVParameters & lv_params){
       // defining a complex matrix M which will contain our flavor
       // violating flavor structure.
       gsl_matrix_complex * M = gsl_matrix_complex_calloc(3,3);
       gsl_matrix_complex_set(M,1,0,lv_params.c_emu);
       gsl_matrix_complex_set(M,0,1,gsl_complex_conjugate(lv_params.c_emu));
       gsl_matrix_complex_set(M,2,1,lv_params.c_mutau);
       gsl_matrix_complex_set(M,1,2,gsl_complex_conjugate(lv_params.c_mutau));
       LVP = squids::SU_vector(M);
       // rotate from flavor to mass basis
       LVP.RotateToB1(params);
       // free allocated matrix
       gsl_matrix_complex_free(M);
       c_params = lv_params;
       lv_parameters_set = true;
    }

    void Set_LV_CMatrix(gsl_matrix_complex * cmatrix){
       // defining a complex matrix M which will contain our flavor
       // violating flavor structure.
       c_params = {gsl_matrix_complex_get(cmatrix,1,0),
                   gsl_matrix_complex_get(cmatrix,2,1)};
       LVP = squids::SU_vector(cmatrix);
       // rotate from flavor to mass basis
       LVP.RotateToB1(params);
       lv_parameters_set = true;
    }

    void dump_probabilities() const {
      for (int ie = 0; ie < ne; ie++){
        std::cout << E_range[ie] << ' ';
        for (int flv = 0; flv < numneu; flv++){
          if (NT == both){
            std::cout << EvalFlavorAtNode(flv,ie,0) << ' ';
            std::cout << EvalFlavorAtNode(flv,ie,1) << ' ';
          }
          else if ( NT == neutrino){
            std::cout << EvalFlavorAtNode(flv,ie,0) << ' ';
            std::cout << 0.0 << ' ';
          }
          else if ( NT == neutrino){
            std::cout << 0.0 << ' ';
            std::cout << EvalFlavorAtNode(flv,ie,1) << ' ';
          }
        }
        std::cout << '\n';
      }
    }

};

} // close nusquids namespace

#endif //nusquidslv_h

