from __future__ import print_function
import numpy as np
import sys
import time
import pymultinest
from pymultinest.solve import solve
from pymultinest.watch import ProgressPrinter
import lvsearchpy as lv

#paths
effective_area_path='../data/effective_area.h5'
events_path='../data/simple_corrected_data_release.dat'
chris_flux_path='../data/conventional_flux.h5'
kaon_flux_path='../data/kaon_flux.h5'
pion_flux_path='../data/pion_flux.h5'
prompt_flux_path='../data/prompt_flux.h5'

# constructing object
lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
lvsearch.SetVerbose(False)

logRCmutau = -40; logICmutau = -40; logCmumu = -40;

parameters = ["normalization", "cosmic_ray_slope", "pik", "prompt_norm", "astro_norm", "astro_gamma"]
#parameters_prior_ranges = [[0,10], [-0.5, 0.5], [0., 2], [0, 10], [0, 10], [-5, 5]]
parameters_prior_ranges = [0,10,-0.5, 0.5,0.,2,0,10,0,10,-5,5]
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
                             [np.power(10.,logRCmutau),np.power(10.,logICmutau),np.power(10.,logCmumu)]])
    output = -lvsearch.llhFull(theta_)
    return output

## Multinest business

tt = time.time()
prefix = "mnrun_" + str(logRCmutau) + "_" + str(logICmutau) + "_" + str(logCmumu)
#progress = ProgressPrinter(n_params = n_params, outputfiles_basename=prefix); progress.start()
result = pymultinest.run(LogLikelihood=lnProb, Prior=CubePrior, n_dims=n_params,
                           outputfiles_basename=prefix, verbose=False)
#progress.end()

analyzer = pymultinest.Analyzer(outputfiles_basename=prefix, n_params=n_params)

a_lnZ = analyzer.get_stats()['global evidence']

np.savez('lv_evidence_'+prefix+'.npy',logRCmutau=logRCmutau,logICmutau=logICmutau,logCmumu=logCmumu,a_lnZ=a_lnZ)
print("Evidence ", a_lnZ)

