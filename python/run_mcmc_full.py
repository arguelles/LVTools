import numpy as np
import subprocess
import emcee
import time
import sys
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

#calculate likelihood from c++
def llhCPP(theta):
    theta[-3:] = np.power(10.,theta[-3:])
    output=lvsearch.llhFull(theta)
    return -output
 
def lnprior(theta):
    normalization, cosmic_ray_slope, pik, prompt_norm, astro_norm, astro_gamma, logRCmutau, logICmutau, logCmumu = theta
    if -30 < logRCmutau < -25 and -30 < logICmutau < -25 and -30 < logCmumu < -25 :
    #if -30 < logRCmutau < -25 and -30 < logICmutau < -25 and -30 < logCmumu < -25 \
    #        and 0.1 < normalization < 10 and -0.1 < cosmic_ray_slope < 0.1 and 0.1 < pik < 2.0 and 0 < prompt_norm < 10. and 0 < astro_norm < 10. and -0.5 <astro_gamma < 0.5:
        return 0.0
    return -np.inf

def lnprob(theta):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + llhCPP(theta)

## MCMC business

tt = time.time()
print("Initializing walkers")
ndim = 9
nwalkers = 200
p0_base = [1.,0.,1.,1.,1.,0.,-28,-28,-28]
p0 = [p0_base + np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Running burn-in")
pos, prob, state = sampler.run_mcmc(p0, 10000)
sampler.reset()
nsteps = 100000
width = 30
# sampler.run_mcmc(pos,500) #regular run
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    n = int((width+1) * float(i) / nsteps)
    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
sys.stdout.write("\n")
print("Time elapsed", time.time()-tt)

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

np.savetxt("chain_new_full.dat",samples)
import corner
#fig = corner.corner(samples, labels=["$log(ReC_{\mu\tau})$", "$log(ImagC_{\mu\tau})$", "$log(C_{\mu\mu})$"])
fig = corner.corner(samples)
fig.savefig("./triangle_full.png")

