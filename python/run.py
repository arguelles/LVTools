import numpy as np
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

#calculate likelihood from c++
def llhCPP(theta):
    output=lvsearch.llh(np.power(10.,theta))
    return output[-1]

def lnprior(theta):
    logRCmutau, logICmutau, logCmumu = theta
    if -30 < logRCmutau < -25 and -30 < logICmutau < -25 and -30 < logCmumu < -25:
        return 0.0
    return -np.inf

def lnprob(theta):
	lp = lnprior(theta)
	#print(lp)
	if not np.isfinite(lp):
		return -np.inf
	return lp + llhCPP(theta)

p0_base = [-28,-28,-28]
print lvsearch.llh(np.power(10.,p0_base))
print lnprob(p0_base)

