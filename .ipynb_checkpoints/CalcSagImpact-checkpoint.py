#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCULATE IMPACT OF SAGITTARIUS ON OUTER DISC
"""
import numpy as np
import os
import CalcImpulseImpact as cii
import pandas as pd
import CoordTrans as ct
import time

### USER-DEFINED PARAMETERS #########################################################
MOCKDFFILE          = "../results/impactofsag/GalacticPlane_new.txt"
TEST                = False
NTEST               = 16
PERTURBMASS         = 5.e10                    # solar masses
PERTURBRS           = 3.                       # kpc
SAGIMPACTWC         = np.array([15., 15., 10., -45.82, -93.79, 255.52])#np.array([26.270, -0.821, -1.397, -45.82, -93.79, 255.52]) # Impact coordinates 1.2 Gyr ago (Vasiliev et al., 2020)
TRAJSIZE            = 10
RUNNO               = 3
RUNFOLDER           = "../results/impactofsag/run"+str(RUNNO)
SAGIMPACTPARSFILE   = RUNFOLDER+"/sagImpactPars.txt"
VKICKSOUTPUTFILE    = RUNFOLDER+"/vKicks.csv"
EVOLVEDDFOUTPUTFILE = RUNFOLDER+"/evolvedDF.csv"
print("Galactocentric distance of impact = "+str(np.round(np.sqrt(SAGIMPACTWC[0]**2. + SAGIMPACTWC[1]**2. + SAGIMPACTWC[2]**2.)))+" kpc.")
print("Galactocentric velocity of impact = "+str(np.round(np.sqrt(SAGIMPACTWC[3]**2. + SAGIMPACTWC[4]**2. + SAGIMPACTWC[5]**2.)))+" km/s.")

#%% CREATE DIRECTORY AND WRITE RUN SPECS
if not os.path.exists(RUNFOLDER):
    os.makedirs(RUNFOLDER)
np.savetxt(SAGIMPACTPARSFILE,SAGIMPACTWC)

#%% READ DF MOCK SAMPLE
mockDF     = np.loadtxt(MOCKDFFILE)
wp         = mockDF[:,1:7]
wc         = ct.PolarToCartesian(wp)
if (TEST==True):
    wc = wc[0:NTEST,:]
nstars     = len(wc)
print(str(len(wc))+" stars in mock DF sample.")

#%% CALCULATE KICKS AND PERTURBED COORDINATES. SAVE KICKS TO FILE.
impulseImpact     = cii.CalcImpulseImpact()
vKicks, perturbWc = impulseImpact.perturbDF(wc,PERTURBMASS,PERTURBRS,SAGIMPACTWC)
vKicksDF = pd.DataFrame(vKicks,columns=["vx (km/s)","vy (km/s)","vz (km/s)"])
vKicksDF.to_csv(VKICKSOUTPUTFILE)

#%% EVOLVE PERTURBED DF FOR A FEW DYNAMICAL TIMESCALES

# Calculate a few dynamical timescales in Gyr at the outer disc
radius       = 16.    # kpc
enclosedMass = 1.e10  # Solar masses
avgDensity   = enclosedMass/((4.*np.pi*radius**3.)/3.)
grav         = 4.3e-6 #(kpc/msun (km/s)^2)
tDyn         = np.sqrt(3.*np.pi/(grav*avgDensity))
print("Dynamical timescale in outer disc = "+str(tDyn)+" Gyr.")
evolveTime   = 3.*tDyn
#%%
startTime = time.time() 
newWc     = impulseImpact.evolveDF(perturbWc,evolveTime,TRAJSIZE)
endTime   = time.time()
print("...DF evolution took "+str(endTime-startTime)+"s.")

#%% SAVE EVOLVED DF TO FILE
newWp = ct.CartesianToPolar(newWc)
if (TEST==True):
    sagImpactPolar = np.column_stack((newWp,mockDF[0:NTEST,7:]))
else:
    sagImpactPolar = np.column_stack((newWp,mockDF[:,7:]))
sagImpactPolarDF = pd.DataFrame(sagImpactPolar,columns=["R (kpc)","phi (rad)","z (kpc)","vr (km/s)","vphi (km/s)",\
                   "vz (km/s)","[Fe/H] (dex)","[a/Fe] (dex)","Age (Gyr)"])
sagImpactPolarDF.to_csv(EVOLVEDDFOUTPUTFILE)