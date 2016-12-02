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
    output=lvsearch.llh(np.power(10.,theta))
    return -output[-1]
 
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

## MCMC business

tt = time.time()
print("Initializing walkers")
ndim = 3
nwalkers = 50
p0_base = [-28,-28,-28]
p0 = [p0_base + np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Running burn-in")
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
nsteps = 10000
width = 30
# sampler.run_mcmc(pos,500) #regular run
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    n = int((width+1) * float(i) / nsteps)
    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
sys.stdout.write("\n")
print("Time elapsed", time.time()-tt)

samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

np.savetxt("chain_new.dat",samples)
import corner
fig = corner.corner(samples, labels=["$log(ReC_{\mu\tau})$", "$log(ImagC_{\mu\tau})$", "$log(C_{\mu\mu})$"])
#fig = corner.corner(samples)
fig.savefig("./triangle.png")

