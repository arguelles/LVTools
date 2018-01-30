import numpy as np

rho_range = np.logspace(-37, -29, 50)
costh_range = np.linspace(-1, 1, 50)

def SphereConvert(rho,costh,phi):
    sinth = np.sqrt(1-costh*costh)
    Cmumu = costh*rho
    CRmumu = (sinth*rho)*np.cos(phi)
    CImumu = (sinth*rho)*np.sin(phi)
    return CRmumu, CImumu, Cmumu

i = 0
for rho in rho_range:
    for costh in costh_range:
        crmumu,cimumu,cmumu = SphereConvert(rho,costh,0.)
        print i, crmumu, cimumu, cmumu
        i = i+1
