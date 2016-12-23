import numpy as np
import subprocess
import emcee
import lvsearchpy as lv

# paths
effective_area_path='/Users/carguelles/DropboxMIT/LVSearch/IC86OfficialRelease/effective_area_release/effective_area.h5'
events_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/simple_corrected_data_release.dat'
chris_flux_path='/Users/carguelles/DropboxMIT/LVSearch/IC86OfficialRelease/effective_area_release/conventional_flux.h5'
kaon_flux_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/kaon_flux.h5'
pion_flux_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/pion_flux.h5'
prompt_flux_path='/Users/carguelles/DropboxMIT/LVSearch/carlos/data/prompt_flux.h5'

# constructing object
lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
lvsearch.SetVerbose(False)
print lvsearch.GetExpectationDistribution([0.00000000e+00,3.42678594e-03,3.72106605e+00,0.00000000e00,0.00000000e+00,-4.08785930e-01,1.62975083e-26,9.77009957e-25,-1.26185688e-25])
print lvsearch.llhFull([0.00000000e+00,3.42678594e-03,3.72106605e+00,0.00000000e00,0.00000000e+00,-4.08785930e-01,1.62975083e-26,9.77009957e-25,-1.26185688e-25])

