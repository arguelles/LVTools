from __future__ import print_function
import numpy as np
import sys
import time
import pymultinest
from pymultinest.solve import solve
from pymultinest.watch import ProgressPrinter
import lvsearchpy as lv

#paths
central_data_path = '/home/carguelles/monadas/LVTools/'
effective_area_path= central_data_path + '/data/effective_area.h5'
events_path= central_data_path + '/data/simple_corrected_data_release.dat'
chris_flux_path= central_data_path + '/data/conventional_flux.h5'
kaon_flux_path= central_data_path + '/data/kaon_flux.h5'
pion_flux_path= central_data_path + '/data/pion_flux.h5'
prompt_flux_path= central_data_path + '/data/prompt_flux.h5'

# constructing object
lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
lvsearch.SetVerbose(False)
lvsearch.SetEnergyExponent(3)
#lvsearch.SetEnergyExponent(2)

batch_mode = False

if not batch_mode:
    if(len(sys.argv)!=4):
        print("wrong number of parameters")
        exit()
    RCmutau = float(sys.argv[1]);
    ICmutau = float(sys.argv[2]);
    Cmumu = float(sys.argv[3]);
else:
    if(len(sys.argv)!=2):
        print("wrong number of parameters")
        exit()
    puntillos = np.genfromtxt('/home/carguelles/monadas/LVTools/scr/lv6_blob_points')
    model_index = int(sys.argv[1])

    RCmutau = puntillos[model_index][1];
    ICmutau = puntillos[model_index][2];
    Cmumu = puntillos[model_index][3];

parameters = ["normalization", "cosmic_ray_slope", "pik", "prompt_norm", "astro_norm", "astro_gamma"]
#parameters_prior_ranges = [[0.5,3.], [-0.5, 0.5], [0., 2], [0, 10], [0, 10], [-4, 4]]
parameters_prior_ranges = [0.5,3, -0.3,0.3, 0.5,1.5, 0,10, 0,10,-1,1]
n_params = len(parameters)
theta = np.zeros(n_params)
prange = np.zeros(n_params)

def CubePrior(cube, ndim, nparams):
    # default are uniform priors
    return ;

def lnProb(cube, ndim, nparams):
    for i in range(ndim):
        prange[i] = parameters_prior_ranges[2*i+1] - parameters_prior_ranges[2*i]
        theta[i] = prange[i]*cube[i] + parameters_prior_ranges[2*i]
    theta_ = np.concatenate([theta,
                             [RCmutau,ICmutau,Cmumu]])
    output = -lvsearch.llhFull(theta_)
    return output

## Multinest business

#tt = time.time()
prefix = "mnrun_" + str(RCmutau) + "_" + str(ICmutau) + "_" + str(Cmumu) + "_"
#progress = ProgressPrinter(n_params = n_params, outputfiles_basename=prefix); progress.start()
print("begin running evidence calculation")
result = pymultinest.run(LogLikelihood=lnProb, Prior=CubePrior, n_dims=n_params,
                           outputfiles_basename=prefix, verbose=True)
                           #outputfiles_basename='/scratch/carguelles/'+prefix, verbose=False)
#progress.end()
print("end running evidence calculation")

print("begin analysis")
analyzer = pymultinest.Analyzer(outputfiles_basename=prefix, n_params=n_params)
a_lnZ = analyzer.get_stats()['global evidence']
print("end analysis")

if batch_mode:
    output_location = '/home/carguelles/monadas/LVTools/scr/metaresults/'
    np.savez(output_location+'lv_evidence_'+prefix+'.npy',RCmutau=RCmutau,ICmutau=ICmutau,Cmumu=Cmumu,a_lnZ=a_lnZ)
print("Evidence ", a_lnZ)

