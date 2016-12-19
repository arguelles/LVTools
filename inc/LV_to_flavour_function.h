#ifndef _LV_TWO_FLAVOR_FUNCTION_
#define _LV_TWO_FLAVOR_FUNCTION_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <ctime>
#include <iomanip>
#include <stdexcept>

#include <cmath>
#include <vector>
#include <numeric>
#include <functional>

//#include <vector>   //make code slower
//#include <algorithm>//make code slower

#include <stdlib.h>
#include <stdio.h>

///////////////////////////////////////////
//2016/06/09
//Gonzalez-Garcia et al., PRD71(2005)093010
///////////////////////////////////////////

namespace LV {
    // Teppeis' constants I have wrap into a namespace.

    //global constant
    double xpi   = 3.1415927; //3.1415926535
    double x2pi  = 6.2831853; //2*3.1415926535
    double xhbarc= 197.326968E-21;//MeV*fm=E-21GeV*kilometre
    double xGF   = 1.16637E-5;//GeV^-2
    double xAv   =6.022E23;   //Avogadro
    double xR    =6378.140;   //km
    double xDEG= 0.01745329252;//deg to rad;

    //neutrino constant
    double th13= 8.50*xDEG;
    double th23=42.30*xDEG;
    double th12=33.48*xDEG;
    double dCP = 0.00*xDEG;
    double c2t=cos(2.0*th23);
    double s2t=sin(2.0*th23);
    //mass
    double dm12=7.50e-5*1.0E-18;//GeV2
    double dm32=2.45e-3*1.0E-18;//GeV2

    // This is the function we will plug in
    double OscillationProbabilityTwoFlavorLV(double neutrino_energy /*GeV*/, double baseline, /*km*/
					     double cmutau_real,double cmutau_imag,double cmumu,double nlv){

        using namespace std;
        //Oscillation equation parameters
	// changes
	//1) xi is obtained from SME parameters including cmumu
	//2) typo, eta should be imaginary/real
	//3) typo, Dd is *2, not /2
	//4) typo, Rn is *4.0, not *2.0
        //double xi=xpi/4.0;
        //double eta=atan(cmutau_real/cmutau_imag);
        //double Dd=sqrt(cmutau_real*cmutau_real + cmutau_imag*cmutau_imag)/2.0;
        //double a0=sin(2.0*th23)*sin(2.0*xi)*cos(eta);
        //double a1=cos(2.0*th23)*cos(2.0*xi)+a0;
        //double a2=pow(sin(2.0*th23),2.0);
        //double a3=pow(sin(2.0*xi),2.0);
	//double Rn=2.0*Dd*neutrino_energy*neutrino_energy/dm32;
        //double R=sqrt(1.0+Rn*Rn+2.0*Rn*a1);
        //double sin22T=1.0/R/R*(a2+Rn*Rn*a3+2.0*Rn*a0);
        double Dd=sqrt(pow(cmumu,2.0)+pow(cmutau_real,2.0)+pow(cmutau_imag,2.0));
        double c2x=cmumu/Dd;
        double s2x=sqrt(pow(cmutau_real,2.0)+pow(cmutau_imag,2.0))/Dd;
        double cet=cmutau_real/sqrt(pow(cmutau_real,2.0)+pow(cmutau_imag,2.0));
        double Rn=4.0*Dd*pow(neutrino_energy,nlv+1)/dm32;
        double R=sqrt(1.0+Rn*Rn+2.0*Rn*(c2t*c2x+s2t*s2x*cet));
        double sin22T=1.0/R/R*(s2t*s2t+Rn*Rn*s2x*s2x+2.0*Rn*s2t*s2x*cet);
        //Oscillation equation itself
        double Posc=1.0-sin22T*pow(sin(dm32*baseline*R/4.0/neutrino_energy/xhbarc),2.0);

        return Posc;
    }

    // This is the function we will plug in
    double OscillationProbabilityTwoFlavorLV_intL(double neutrino_energy /*GeV*/, double baseline_0, double baseline_1, /*km*/
        double cmutau_real,double cmutau_imag,double cmumu,double nlv){

        using namespace std;
        //Oscillation equation parameters
        double Dd=sqrt(pow(cmumu,2.0)+pow(cmutau_real,2.0)+pow(cmutau_imag,2.0));
        double c2x=cmumu/Dd;
        double s2x=sqrt(pow(cmutau_real,2.0)+pow(cmutau_imag,2.0))/Dd;
        double cet=cmutau_real/sqrt(pow(cmutau_real,2.0)+pow(cmutau_imag,2.0));
        double Rn=4.0*Dd*pow(neutrino_energy,nlv+1)/dm32;
        double R=sqrt(1.0+Rn*Rn+2.0*Rn*(c2t*c2x+s2t*s2x*cet));
        double sin22T=1.0/R/R*(s2t*s2t+Rn*Rn*s2x*s2x+2.0*Rn*s2t*s2x*cet);
        //Oscillation equation itself

        double A = sin22T;
        double _2B = 2 * dm32*R/4.0/neutrino_energy/xhbarc;
        double delta_L = baseline_1 - baseline_0;
        double Posc_L= delta_L - (_2B*delta_L - (sin(_2B*baseline_1) - sin(_2B*baseline_0)) ) * (A/(2*_2B));

        return Posc_L;
    }

   double OscillationProbabilityTwoFlavorLV_Astro(double neutrino_energy /*GeV*/,
       double cmutau_real,double cmutau_imag,double cmumu,double nlv){

        using namespace std;
        //Oscillation equation parameters
        double Dd=sqrt(pow(cmumu,2.0)+pow(cmutau_real,2.0)+pow(cmutau_imag,2.0));
        double c2x=cmumu/Dd;
        double s2x=sqrt(pow(cmutau_real,2.0)+pow(cmutau_imag,2.0))/Dd;
        double cet=cmutau_real/sqrt(pow(cmutau_real,2.0)+pow(cmutau_imag,2.0));
        double Rn=4.0*Dd*pow(neutrino_energy,nlv+1)/dm32;
        double R=sqrt(1.0+Rn*Rn+2.0*Rn*(c2t*c2x+s2t*s2x*cet));
        double sin22T=1.0/R/R*(s2t*s2t+Rn*Rn*s2x*s2x+2.0*Rn*s2t*s2x*cet);
        //Oscillation equation itself

        double A = sin22T;
        double Posc_Astro= 1 - A/2;

        return Posc_Astro;
    }

} // end namespace

#endif
