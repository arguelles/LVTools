from __future__ import print_function
import numpy as np
import sys
from pymultinest.solve import solve
import lvsearchpy as lv

#paths
effective_area_path='/Users/carguelles/DropboxMIT/LVSearch/IC86OfficialRelease/effective_area_release/effective_area.h5'
events_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/simple_corrected_data_release.dat'
chris_flux_path='/Users/carguelles/DropboxMIT/LVSearch/IC86OfficialRelease/effective_area_release/conventional_flux.h5'
kaon_flux_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/kaon_flux.h5'
pion_flux_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/pion_flux.h5'
prompt_flux_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/prompt_flux.h5'

# constructing object
lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
lvsearch.SetVerbose(False)

logRCmutau = -30; logICmutau = -30; logCmumu = -30;

def CubePrior(theta):
    normalization, cosmic_ray_slope, pik, prompt_norm, astro_norm, astro_gamma = theta
    if(not( 0 < normalization < 10 and -0.5 < cosmic_ray_slope < 0.5 and 0. < pik < 2
        and 0 < prompt_norm < 10 and 0 < astro_norm < 10 and -5 < astro_gamma < 5)):
        return -np.inf
    return 0

def lnProb(theta):
    theta_ = theta[:]
    # physics parameter test points
    theta_.append(np.power(10.,logRCmutau))
    theta_.append(np.power(10.,logICmutau))
    theta_.append(np.power(10.,logCmumu))
    output = lvsearch.llhFull(theta_)
    return -output

## Multinest business

tt = time.time()
parameters = ["normalization", "cosmic_ray_slope", "pik", "prompt_norm", "astro_norm", "astro_gamma"]
prefix = str(logRCmutau) + "_" + str(logICmutau) + "_" + str(logCmumu)
n_params = len(parameters)
result = solve(LogLikelihood=lnProb, Prior=CubePrior, n_dims=n_params,
               outputfiles_basename=prefix, verbose=True)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()

