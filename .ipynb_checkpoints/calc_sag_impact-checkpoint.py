#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCULATE IMPACT OF SAGITTARIUS ON OUTER DISC
"""
import numpy as np
import os
import calc_impulse_impact as cii
import pandas as pd
import CoordTrans as ct
import scipy.optimize as sciOpt
import matplotlib.pyplot as plt
import time

### USER-DEFINED PARAMETERS #########################################################
SOLPOS              = np.array([8.2,0.014,-8.6,13.9+232.8,7.1]) # McMillan 2017
NMOCK               = 100000
IMPACTAGE           = 6
ZMIN                = 0.6 #kpc
ZMAX                = 1.  #kpc
TEST                = True
NTEST               = 3000
PERTURBMASS         = 3.e10                    # solar masses
PERTURBRS           = 3.                       # kpc
SAGIMPACTWC         = np.array([12.0, 0., 3.0, -45.82, -93.79, 255.52])#np.array([26.270, -0.821, -1.397, -45.82, -93.79, 255.52])
TRAJSIZE            = 1
POTFILE             = "../potential_parameters/mcmillan17.ini"

RUNNO               = 10
RUNFOLDER           = "../results/impactofsag/run"+str(RUNNO)
PLOTFOLDER          = "../plots/impactofsag/run"+str(RUNNO)
SAGIMPACTPARSFILE   = RUNFOLDER+"/sagImpactPars.txt"
if (TEST):
    VKICKSOUTPUTFILE    = RUNFOLDER+"/vKicks_test"+str(NTEST)+".csv"
    EVOLVEDDFOUTPUTFILE = RUNFOLDER+"/evolvedDF_test"+str(NTEST)+".csv"
else:
    VKICKSOUTPUTFILE    = RUNFOLDER+"/vKicks.csv"
    EVOLVEDDFOUTPUTFILE = RUNFOLDER+"/evolvedDF.csv"
print("Galactocentric R of impact = "+str(np.round(SAGIMPACTWC[0]))+" kpc.")
print("Galactocentric z of impact = "+str(np.round(SAGIMPACTWC[2]))+" kpc.")
print("Galactocentric velocity of impact = "+str(np.round(np.sqrt(SAGIMPACTWC[3]**2. + SAGIMPACTWC[4]**2. + SAGIMPACTWC[5]**2.)))+" km/s.")

#%% CREATE DIRECTORIES AND WRITE RUN SPECS
if not os.path.exists(RUNFOLDER):
    os.makedirs(RUNFOLDER)
np.savetxt(SAGIMPACTPARSFILE,np.hstack([PERTURBMASS,PERTURBRS,SAGIMPACTWC,NTEST]))

if not os.path.exists(PLOTFOLDER):
    os.makedirs(PLOTFOLDER)

#%% CREATE MOCK MILKY WAY GALAXY

"""
# Set units for the agama library
agama.setUnits(mass=1,length=1,velocity=1)

# McMillan galaxy potential
gp  = agama.Potential(POTFILE) 

# Milky Way distribution function
# (FUNCTION CALCULATING MILKY WAY DF GIVEN OBSERVED COORDINATES. PARAMETERS FROM
# http://mnras.oxfordjournals.org/content/445/3/3133.full.pdf (Piffl et al. 2014)

R_d           = 2.68   
R_sigma_r     = 1.28*R_d # 2.* R_d
R_sigma_z     = 2.0*R_d # 2.* R_d                               
thin_disk_df_dict = dict(
                type       = 'QuasiIsothermal',
                Sigma0     = 1.0,        
                coefJr     = 0.0,
                coefJz     = 0.0,
                Rdisk      = R_d,     
                sigmar0    = 30.0 * np.exp(SOLPOS[0]/R_sigma_r),#34.0 * np.exp(SOLPOS[0]/R_sigma_r), 
                sigmaz0    = 22.0 * np.exp(SOLPOS[0]/R_sigma_z),#25.1 * np.exp(SOLPOS[0]/R_sigma_z),       
                Rsigmar    = R_sigma_r,        
                Rsigmaz    = R_sigma_z,       
                sigmamin   = 1.0,        
                Jphimin    = 0.0,
                potential  = gp)

# Create distribution function instance
thin_disk_df = agama.DistributionFunction(**thin_disk_df_dict)

# Create galaxy model and generate a mock sample
mwgm = agama.GalaxyModel(gp,thin_disk_df)

#%% PLOT MOCK VELOCITY DISPERSION PROFILE

# Create mock data
wc,error = mwgm.sample(NMOCK)

#%%

# Convert into olar coordinates
wp = ct.CartesianToPolar(wc)

# Select between ZMIN and ZMAX below and above plane.
index = (np.abs(wp[:,2]) > ZMIN ) & (np.abs(wp[:,2] < ZMAX))

wp_beyond_plane = wp[index,:]
wc_beyond_plane = wc[index,:]

print(len(wp_beyond_plane))

np.savetxt("../results/mock_mw_beyond_plane_wc.txt",wc_beyond_plane)
np.savetxt("../results/mock_mw_beyond_plane_wp.txt",wp_beyond_plane)
"""

#%% Generate robust velocity dispersion profile assuming 0.5 km/s error

# Read in obseved velocity dispersion profiles for young/old stars in outer disk
young_ages_df = pd.read_csv("../data/sigz_young_ages_beyondplane.csv")
old_ages_df   = pd.read_csv("../data/sigz_old_ages_beyondplane.csv")

# Read saved mock dispersion profile
wp_beyond_plane = np.loadtxt("../results/mock_mw_beyond_plane_wp.txt")
wc_beyond_plane = np.loadtxt("../results/mock_mw_beyond_plane_wc.txt")

# Define variables
R_min        = 6.
R_max        = 14.
n_R          = 10
R_edges      = np.logspace(np.log10(R_min),np.log10(R_max),n_R+1)
R_bin        = np.zeros(n_R)
sigz_bin     = np.zeros(n_R)
sigz_err_bin = np.zeros(n_R)

# Define objective function to maximize for estimation of robust velocity dispersion
# sigma - velocity dispersion
# vel   - vector of velocities
# eVel  - vector of velocity uncertainties
# mu    - mean velocity
def sigmaObjFunc(sigma,vel,eVel,mu):
    
    y = ((vel-mu)**2.)/((sigma**2.+eVel**2.)**2.) - (1./(sigma**2.+eVel**2.))
    
    return (np.sum(y))

# Number of sigma for clipping
f = 5.

for j_R in range(n_R):
        
    ## OBSERVATIONS
    # Define R region
    R_index = ((R_edges[j_R]<wp_beyond_plane[:,0]) & (wp_beyond_plane[:,0]<R_edges[j_R+1])) 
               
    # Calculate mean vz
    mean = np.mean(wp_beyond_plane[R_index,5])
                
    # Calculate initial guess for sigz
    sigma = np.std(wp_beyond_plane[R_index,5])
        
    # Clip velocities beyond three sigma
    clipped_index = (wp_beyond_plane[:,5] > (mean-f*sigma)) & (wp_beyond_plane[:,5] < (mean+f*sigma))
    index   = clipped_index * R_index
        
    # Calculate mean radii of clipped velocities
    R_bin[j_R] = np.mean(wp_beyond_plane[index,0])
        
    n = np.sum(index)
    print(n)
        
    # Recalculate mean
    mean = np.mean(wp_beyond_plane[index,5])
        
    # Maximise likelihood to find robust measure of observed velocity dispersion
    intervalMin = 0.
    intervalMax = 100.
    opt_sigz = sciOpt.brentq(sigmaObjFunc,intervalMin,intervalMax, 
                             args=(wp_beyond_plane[index,5],wp_beyond_plane[index,5]*0.+0.5,mean), 
                             xtol=2e-12, rtol=8.881784197001252e-16, 
                             maxiter=100, full_output=False)
    sigz_bin[j_R] = opt_sigz
    n = np.sum(index)
    sigz_err_bin[j_R] = opt_sigz * np.sqrt(1./(2.*n))

#%% CALCULATE KICKS AND PERTURBED COORDINATES. SAVE KICKS TO FILE.
impulseImpact     = cii.CalcImpulseImpact()
if (TEST):
    if (PERTURBMASS==0.): # i.e. no kicks
        perturbWc = np.copy(wc_beyond_plane[0:NTEST,:])
    else:
        vKicks, perturbWc = impulseImpact.perturbDF(wc_beyond_plane[0:NTEST,:],PERTURBMASS,PERTURBRS,SAGIMPACTWC)
        vKicksDF = pd.DataFrame(vKicks,columns=["vx (km/s)","vy (km/s)","vz (km/s)"])
        vKicksDF.to_csv(VKICKSOUTPUTFILE)
else:
    if (PERTURBMASS==0.):
        perturbWc = np.copy(wc_beyond_plane)
    else:
        vKicks, perturbWc = impulseImpact.perturbDF(wc_beyond_plane,PERTURBMASS,PERTURBRS,SAGIMPACTWC)
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
evolveTime   = 2.*tDyn

#%%
startTime  = time.time() 
wc_evolved = impulseImpact.evolveDF(perturbWc,evolveTime,TRAJSIZE,POTFILE)
endTime    = time.time()
print("...DF evolution took "+str(endTime-startTime)+"s.")

#%% RESELECT STARS BEYOND PLANE AND PLOT NEW VERTICAL VELOCITY DISPERSION PROFILE
wp_evolved = ct.CartesianToPolar(wc_evolved)
index = (np.abs(wp_evolved[:,2]) > ZMIN) & (np.abs(wp_evolved[:,2]) < ZMAX)
wp_evolved_beyond_plane = wp_evolved[index,:]
print(len(wp_evolved_beyond_plane))

R_evolved_bin        = np.zeros(n_R)
sigz_evolved_bin     = np.zeros(n_R)
sigz_err_evolved_bin = np.zeros(n_R)

for j_R in range(n_R):
        
    ## OBSERVATIONS
    # Define R region
    R_index = ((R_edges[j_R]<wp_evolved_beyond_plane[:,0]) & (wp_evolved_beyond_plane[:,0]<R_edges[j_R+1])) 
               
    # Calculate mean vz
    mean = np.mean(wp_evolved_beyond_plane[R_index,5])
                
    # Calculate initial guess for sigz
    sigma = np.std(wp_evolved_beyond_plane[R_index,5])
        
    # Clip velocities beyond three sigma
    clipped_index = (wp_evolved_beyond_plane[:,5] > (mean-f*sigma)) & (wp_evolved_beyond_plane[:,5] < (mean+f*sigma))
    index   = clipped_index * R_index
        
    n = np.sum(index)
    print(n)
    
    if (n==0):
        R_evolved_bin[j_R]        = np.nan
        sigz_evolved_bin[j_R]     = np.nan
        sigz_err_evolved_bin[j_R] = np.nan
    else:
        
        # Calculate mean radii of clipped velocities
        R_evolved_bin[j_R] = np.mean(wp_evolved_beyond_plane[index,0])
        
        # Recalculate mean
        mean = np.mean(wp_evolved_beyond_plane[index,5])
        
        # Maximise likelihood to find robust measure of observed velocity dispersion
        opt_sigz = sciOpt.brentq(sigmaObjFunc,intervalMin,intervalMax, 
                                 args=(wp_evolved_beyond_plane[index,5],wp_evolved_beyond_plane[index,5]*0.+0.5,mean), 
                                 xtol=2e-12, rtol=8.881784197001252e-16, 
                                 maxiter=100, full_output=False)
        sigz_evolved_bin[j_R] = opt_sigz

        sigz_err_evolved_bin[j_R] = opt_sigz * np.sqrt(1./(2.*n))
        
#%% CREATE PLOT
    
fig,ax = plt.subplots(1,1,figsize=(5,4))  
ax.errorbar(R_bin,sigz_bin,yerr=sigz_err_bin,linestyle="-",linewidth=2, color="black",label="Before impact")
ax.errorbar(young_ages_df["R"],young_ages_df["min_sigz"],linestyle="-",linewidth=2,color="gray")
ax.errorbar(young_ages_df["R"],young_ages_df["max_sigz"],linestyle="-",linewidth=2,color="gray")
ax.fill_between(young_ages_df["R"],young_ages_df["min_sigz"],young_ages_df["max_sigz"],color="gray",alpha=0.2)
ax.errorbar(old_ages_df["R"],old_ages_df["min_sigz"],linestyle="-",linewidth=2,color="gray")
ax.errorbar(old_ages_df["R"],old_ages_df["max_sigz"],linestyle="-",linewidth=2,color="gray")
ax.fill_between(old_ages_df["R"],old_ages_df["min_sigz"],old_ages_df["max_sigz"],color="gray",alpha=0.5)
ax.tick_params(axis="x",labelsize=12)
ax.tick_params(axis="y",labelsize=12)
ax.set_xlabel(r"$R$ (kpc)",fontsize=16)
ax.set_ylabel(r"$\sigma_z$ (km/s)",fontsize=16) 
ax.errorbar(R_evolved_bin,sigz_evolved_bin,yerr=sigz_err_evolved_bin,linestyle="-",linewidth=2, color="red",label="Post impact")
ax.set_xlim([6.4,13.8])
ax.set_ylim([10.,40.0])
ax.legend()
plt.tight_layout(pad=1)
plt.savefig(PLOTFOLDER+"sagImpact.png")
