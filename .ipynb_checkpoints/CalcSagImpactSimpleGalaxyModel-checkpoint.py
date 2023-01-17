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
DATAFILE         = "../data/LMRC-DR4-VF-SNR30-newnames-GaiaeDR3.csv"
IMOCK            = 6                                                           # Mock ID
DISCSETUP        = "DoubleDisc"      # Configuration of low-alpha thin disk
RESULTSDIR       = "../results/mocksamples/"+DISCSETUP+"/mock"+str(IMOCK)     
MOCKOBSFILE      = RESULTSDIR+"/mocksamples.txt" # Where to save mock samples
MOCKOBSFILE_NOSF = RESULTSDIR+"/mocksamples_nosf.txt"
USEMOCK          = True # Whether to us mock data generated from model or data itself
SOLPOS              = np.array([8.2,0.014,-8.6,13.9+232.8,7.1]) # McMillan 2017
IMPACTAGE           = 6
ZMIN                = 0.6 #kpc
ZMAX                = 1.  #kpc
TEST                = True
NTEST               = 1000
PERTURBMASS         = 2.e10                    # solar masses
PERTURBRS           = 3.                       # kpc
SAGIMPACTWC         = np.array([13., 0., -3.0, -45.82, -93.79, 255.52])#np.array([26.270, -0.821, -1.397, -45.82, -93.79, 255.52])
TRAJSIZE            = 1
POTFILE             = "../data/mcmillan17.ini"
RUNNO               = 16
RUNFOLDER           = "../results/impactofsag/run"+str(RUNNO)
SAGIMPACTPARSFILE   = RUNFOLDER+"/sagImpactPars.txt"
if (TEST):
    VKICKSOUTPUTFILE    = RUNFOLDER+"/vKicks_test"+str(NTEST)+".csv"
    EVOLVEDDFOUTPUTFILE = RUNFOLDER+"/evolvedDF_test"+str(NTEST)+".csv"
else:
    VKICKSOUTPUTFILE    = RUNFOLDER+"/vKicks.csv"
    EVOLVEDDFOUTPUTFILE = RUNFOLDER+"/evolvedDF.csv"
print("Galactocentric distance of impact = "+str(np.round(np.sqrt(SAGIMPACTWC[0]**2. + SAGIMPACTWC[1]**2. + SAGIMPACTWC[2]**2.)))+" kpc.")
print("Galactocentric velocity of impact = "+str(np.round(np.sqrt(SAGIMPACTWC[3]**2. + SAGIMPACTWC[4]**2. + SAGIMPACTWC[5]**2.)))+" km/s.")

#%% CREATE DIRECTORY AND WRITE RUN SPECS
if not os.path.exists(RUNFOLDER):
    os.makedirs(RUNFOLDER)
np.savetxt(SAGIMPACTPARSFILE,np.hstack([PERTURBMASS,PERTURBRS,SAGIMPACTWC]))


if (USEMOCK):
    trueSamples = np.loadtxt(MOCKOBSFILE_NOSF)
    nnoise_nosf = 1
    nobs        = 9
    trueSamples = np.reshape(trueSamples,[-1,nobs])
    feh         = trueSamples[:,0]
    afe         = trueSamples[:,1]
    age         = trueSamples[:,2]
    nstars      = len(trueSamples)

    # Convert coordinates to polar
    we        = trueSamples[:,0:6]
    wg        = ct.EquatorialToGalactic(we)
    wp        = ct.GalacticToPolar(wg,SOLPOS)
    wc        = ct.PolarToCartesian(wp)
else:    
    lamostgaia = pd.read_csv(DATAFILE)

    # Collect coordinates (ra/rad, dec/rad, s/kpc, vr/kms-1, mura/masyr-1, 
    # mudec/masyr-1, [Fe/H]/dex, [a/Fe]/dex, age/Gyr)
    Obs = np.column_stack((lamostgaia["ra"]/180.*np.pi,
                           lamostgaia["dec"]/180.*np.pi,
                           lamostgaia["s"],
                           lamostgaia["vr"],
                           lamostgaia["pmra"],
                           lamostgaia["pmdec"],
                           lamostgaia["feh"],
                           lamostgaia["afe"],
                           lamostgaia["age"]))

    feh = np.copy(lamostgaia["feh"])
    afe = np.copy(lamostgaia["afe"])
    age = np.copy(lamostgaia["age"])

    print("Read LAMOST DR4 red clump supplemented with Gaia eDR3 ...")

    # Replace stars with NaN proper motions with original UCAC5 proper motions
    index = (np.isnan(lamostgaia["pmra"])) | \
            (np.isnan(lamostgaia["pmdec"])) | \
                (np.isnan(lamostgaia["pmra_error"])) | \
                    (np.isnan(lamostgaia["pmdec_error"]))
    Obs[index,4]  = lamostgaia["mra"][index]
    Obs[index,5]  = lamostgaia["mdec"][index]
    print("Replaced "+str(np.sum(index))+" stars having NaN proper motions with original UCAC5 proper motions...")

    # Remove stars that are likely to be massive merged stars
    idx_merged_massive = (afe>0.12) & (age<5.)
    print("Removing "+str(np.sum(idx_merged_massive))+" stars that are likelly to be merged stars.")
    Obs  = Obs[~idx_merged_massive,:]
    feh  = feh[~idx_merged_massive]
    afe  = afe[~idx_merged_massive]
    age  = age[~idx_merged_massive]

    # Remove stars with ages older than the age of the Universe
    idx_older_universe = age>13.1
    print("Removing a further "+str(np.sum(idx_older_universe))+" stars that are older than the age of the Universe.")
    Obs  = Obs[~idx_older_universe,:]
    feh  = feh[~idx_older_universe]
    afe  = afe[~idx_older_universe]
    age  = age[~idx_older_universe]

    # Select low-alpha population
    slope         = -0.07
    yintercept    = 0.12
    afemax        = yintercept + slope*feh
    idx_lowalpha  = afemax > afe
    print("Removing a further "+str(len(feh) - np.sum(idx_lowalpha))+" stars that belong to the high-alpha disc.")
    Obs  = Obs[idx_lowalpha,:]
    feh  = feh[idx_lowalpha]
    afe  = afe[idx_lowalpha]
    age  = age[idx_lowalpha]
    
    # Number of stars
    nstars = len(afe)
    
    # Convert coordinates to Polar and Cartesian
    ra     = Obs[:,0]
    dec    = Obs[:,1]
    s      = Obs[:,2]
    vr     = Obs[:,3]
    mura   = Obs[:,4]
    mudec  = Obs[:,5]
    wg     = ct.EquatorialToGalactic(Obs)
    wp     = ct.GalacticToPolar(wg,SOLPOS)
    wc     = ct.PolarToCartesian(wg)

# Select stars younger than impact and above and below plane
index = (age<IMPACTAGE) & (np.abs(wp[:,2])>ZMIN) & (np.abs(wp[:,2])<ZMAX)
wc_young_beyondplane = wc[index,:]
feh_young_aboveplane = feh[index]
afe_young_aboveplane = afe[index]
age_young_aboveplane = age[index]

print(len(wc_young_beyondplane))

#%% CALCULATE KICKS AND PERTURBED COORDINATES. SAVE KICKS TO FILE.
impulseImpact     = cii.CalcImpulseImpact()
if (TEST):
    if (PERTURBMASS==0.): # i.e. no kicks
        perturbWc = np.copy(wc_young_beyondplane[0:NTEST,:])
    else:
        vKicks, perturbWc = impulseImpact.perturbDF(wc_young_beyondplane[0:NTEST,:],PERTURBMASS,PERTURBRS,SAGIMPACTWC)
        vKicksDF = pd.DataFrame(vKicks,columns=["vx (km/s)","vy (km/s)","vz (km/s)"])
        vKicksDF.to_csv(VKICKSOUTPUTFILE)
else:
    if (PERTURBMASS==0.):
        perturbWc = np.copy(wc_young_beyondplane)
    else:
        vKicks, perturbWc = impulseImpact.perturbDF(wc_young_beyondplane,PERTURBMASS,PERTURBRS,SAGIMPACTWC)
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
evolveTime   = 6.#3.*tDyn

#%%
startTime = time.time() 
newWc     = impulseImpact.evolveDF(perturbWc,evolveTime,TRAJSIZE,POTFILE)
endTime   = time.time()
print("...DF evolution took "+str(endTime-startTime)+"s.")

#%% SAVE EVOLVED DF TO FILE
newWp = ct.CartesianToPolar(newWc)
if (TEST==True):
    sagImpactPolar = np.column_stack((newWp,feh[0:NTEST],afe[0:NTEST],age[0:NTEST]))
else:
    sagImpactPolar = np.column_stack((newWp,feh,afe,age))
sagImpactPolarDF = pd.DataFrame(sagImpactPolar,columns=["R (kpc)","phi (rad)","z (kpc)","vr (km/s)","vphi (km/s)",\
                   "vz (km/s)","[Fe/H] (dex)","[a/Fe] (dex)","Age (Gyr)"])
sagImpactPolarDF.to_csv(EVOLVEDDFOUTPUTFILE)