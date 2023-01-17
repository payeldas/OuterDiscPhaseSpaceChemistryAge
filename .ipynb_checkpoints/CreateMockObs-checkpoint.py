#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CREATE MOCK LAMOST DR4-Gaia DR3 CATALOGUE
"""
import numpy as np
import pandas as pd
import CoordTrans as ct
import CreateMockCat as cmc
import warnings
warnings.filterwarnings("ignore")
import agama
import MilkyWayEDF 
import MilkyWaySF
import time
import dill

### USER-DEFINED PARAMETERS #########################################################
DISCSETUP         = "DoubleDisc"      # Configuration of thin disk
DATAMODELFILE   = "../results/fit/"+DISCSETUP+"/all/datamodel_nmc"+str(NMC)+".dill" # Where to pickled datamodel instance is stored
INITIALPARSFILE   = "../setupinfo/"+DISCSETUP+"/initial_pars_boundaries.csv"   # Initial parameters file
SFFILE            = "../results/selfunc/selfunc_with_redclumpsel_highres.dill" # File with red clump selection file
USESF             = True                                                       # Whether to use selection function
ADDOBSNOISE       = False                                                      # Whether to add observational noise
NMCMOCK           = 1                                                          # Number of Monte Carlo samples for mock sample 

if (USESF==True):
    SAMPLESFILE = "../results/mocksamples/"+DISCSETUP+"/abovePlane/emcee/mocksamples_"\
                    +FILEROOT+"_samples_nmc"+str(NMC)+"_nwalkers"+str(NWALKERS)+\
                    "_fstd"+str(FSTD)+"_stretch"+str(SMOVE)+"_de"+str(DEMOVE)+"_des"\
                    +str(DESMOVE)+"_fitall_newinitpars5_nmcMock"+str(NMCMOCK)+".txt" # Where to save mock samples
    else:
        SAMPLESFILE = "../results/mocksamples/"+DISCSETUP+"/abovePlane/emcee/mocksamples_"\
                    +FILEROOT+"_samples_nmc"+str(NMC)+"_nwalkers"+str(NWALKERS)+\
                    "_fstd"+str(FSTD)+"_stretch"+str(SMOVE)+"_de"+str(DEMOVE)+"_des"\
                    +str(DESMOVE)+"_fitall_newinitpars5_nmcMock"+str(NMCMOCK)+"_nosf.txt"

#####################################################################################
                
#%% Set agama units
agama.setUnits(mass=1,length=1,velocity=1)

# Read in pickled datamodel instance
print("Load datamodel instance, built with "+str(NMC)+" Monte Carlo samples...")
with open(DATAMODELFILE, 'rb') as dillfile:
    datamodel = dill.load(dillfile)
print("...done.")          
print(" ")

# Read initial edf and potential values within minimum and maximum
print("Create dictionaries with initial, minimum, and maximum parameter values...")

initialparsdict = {}
parsmindict     = {}
parsmaxdict     = {}
with open(INITIALPARSFILE) as f:
    for line in f:
        (key, val, vallo, valhi, description) = line.split(",")
        initialparsdict[key] = float(val)
        parsmindict[key]     = float(vallo)
        parsmaxdict[key]     = float(valhi)
print("...done.")          
print(" ")
npars = len(initialparsdict)

# Determine observational volume within which to extract samples 
minObs    = np.copy(datamodel.minObs)
maxObs    = np.copy(datamodel.maxObs)
minObs[0] = 0.
maxObs[0] = 2.*np.pi
minObs[1] = -np.pi/2.
maxObs[1] = np.pi/2.
minObs[2] = 0.01
maxObs[2] = 18. # This value is important for the samples converging
minObs[6] = -1.0
maxObs[6] = 0.6
minObs[7] = -0.25
maxObs[7] = 0.5
minObs[8] = 0.1
maxObs[8] = 13.1

nstars = np.copy(datamodel.nstars)
nobs   = len(minObs)

#%% CREATE LAMOST SELECTION FUNCTION INSTANCE
mwsf = MilkyWaySF.LamostRedClumpsLowAlphaDiscSF(SFFILE)
    
#%% GENERATE SAMPLES

# Solar position
solpos = np.array([8.2,0.014,-8.6,13.9+232.8,7.1]) # McMillan 2017
        
# Read fit parameter distributions into a dataframe
fitparsdistdf = pd.read_csv(FITPARSDISTFILE,index_col=0)
nfitpars      = len(fitparsdistdf)

#%%
class Posterior:
    
    def  __init__(self,edf,af):
        self.edf  = edf
        self.af   = af
        
    """ CLASS CONSTRUCTOR

    Arguments:
        edf    - action-based extended distribution function [object])
            
    Returns:
        Nothing
    """    
        
    def __call__(self,star):
        
        """ JACOBIAN DETERMINANT * SF * eDF * PRIOR
        
        Arguments:
           star  - ra(rad),dec(rad),s(kpc),vr(km/s),mura(mas/yr),mudec(mas/yr),
                   [Fe/H](dex),[a/Fe](dex),age(Gyr)  
            
        Returns:
            Nothing
        """    
        
        star  = np.atleast_2d(star)
                
        # Extract data
        ra     = star[:,0]
        dec    = star[:,1]
        s      = star[:,2]
        vr     = star[:,3]
        mura   = star[:,4]
        mudec  = star[:,5]
        feh    = star[:,6]
        afe    = star[:,7]
        age    = star[:,8]

        # Convert to Cartesian coordinates (depends on observed coordinates+errors)
        we = np.column_stack((ra,dec,s,vr,mura,mudec))
        wg = ct.EquatorialToGalactic(we)
        wc = ct.GalacticToCartesian(wg,solpos)  
        
        # Galactic position
        xe = np.column_stack((ra,dec,s))
        
        # Calculate actions
        acts = self.af(wc)

        # Chemical properties
        xi = np.column_stack((feh,afe,age))
        
        # Calculate Jacobian determinant for the Equatorial coordinates to 
        # Cartesian coordinates transformation
        jacdet = s**4. * np.cos(dec)
        
        # Calculate extended distribution function probability
        edfprob = self.edf(acts,xi)
        
        # Prior probability
        priorprob = self.edf.Prior()
        
        # Calculate total probability
        prob = jacdet * edfprob * priorprob
        
        if (USESF==True):
            # Calculate selection function probability
            sfprob = mwsf(xe,xi,MODELTYPE)
            prob  *= sfprob
                
        # Check nan/inf/neg
        index       = np.isnan(prob) | np.isinf(prob) | (prob<0.)
        prob[index] = 0.

        return(prob) 
        
#%% Define function to generate Lamost-Gaia samples
def genLamostGaiaSamples():

    # Create nmc realizations of mock samples according to parameter and observed errors
    import random
    random.seed(10)
    #np.random.seed(3)
    nsamp     = np.copy(nstars)
    parsdict  = initialparsdict.copy()
    parsMC    = np.zeros([NMCMOCK,npars])
    samplesMC = np.zeros([NMCMOCK,nsamp,nobs])
    for jmc in range(NMCMOCK):

        print("################################################################")
        print("MONTE CARLO REALIZATION: "+str(jmc+1)) 
        
        while True:
        
            # Generate random set of fitted parameters
            for jfitpars in range(nfitpars):
                if (EMCEE==True):
                    mu  = fitparsdistdf.loc[fitparsdistdf.index[jfitpars],"mu"]
                    std = (fitparsdistdf.loc[fitparsdistdf.index[jfitpars],"onesig_hi"]-
                           fitparsdistdf.loc[fitparsdistdf.index[jfitpars],"onesig_lo"])/2.
                    newpar = np.random.normal(mu,std)
                else:
                    newpar = fitparsdistdf.loc[fitparsdistdf.index[jfitpars],"best"]
                parsdict[fitparsdistdf.index[jfitpars]] = newpar
                print("     "+fitparsdistdf.index[jfitpars]+" = "+str(newpar))
            
            # Record Monte Carlo parameters
            parsMC[jmc,:] = np.fromiter(parsdict.values(),dtype=float)
     
                
            # Galaxy potential parameters
            thindisc_par   = dict(type='Disk', 
                                  surfaceDensity=8.96e+08,#5.70657e+08, 
                                  scaleRadius=2.5,#2.6824, 
                                  scaleHeight=0.3)#0.1960)
            thickdisc_par  = dict(type='Disk', 
                                  surfaceDensity=1.83e+08,#2.51034e+08, 
                                  scaleRadius=3.0,#2.6824, 
                                  scaleHeight=0.9)#0.7010)
            HIgasdisc_par  = dict(type='Disk', 
                                  surfaceDensity=5.31e+07,#9.45097e+07,
                                  scaleRadius=7.,#5.3649, 
                                  scaleHeight=-0.085,#0.04,
                                  innerCutoffRadius=4.)
            HIIgasdisc_par = dict(type='Disk', 
                                  surfaceDensity=2.18e+09,
                                  scaleRadius=1.5, 
                                  scaleHeight=-0.045,
                                  innerCutoffRadius=12.)
            bulge_par      = dict(type='Spheroid',     
                                  densityNorm=9.84e+10,#9.49e+10, 
                                  axisRatioZ=0.5, 
                                  gamma=0., 
                                  beta=1.8,
                                  scaleRadius=0.075,
                                  outerCutoffRadius=2.1)
            dmhalo_par     = dict(type='Spheroid', 
                                  densityNorm=parsdict["densityNorm_dh"],
                                  axisRatioZ=1.0, 
                                  gamma=1., 
                                  beta=3.,
                                  scaleRadius=parsdict["scaleRadius_dh"])
            galpot_par = [thindisc_par,thickdisc_par,HIgasdisc_par,HIIgasdisc_par,bulge_par,dmhalo_par]
            gp         = agama.Potential(galpot_par[0],galpot_par[1],galpot_par[2],galpot_par[3],galpot_par[4],galpot_par[5])
           
            # Create action finder instance
            af = agama.ActionFinder(gp)  
            
            # Create dictionary of global disc parameters
            globalDiscPar = dict(tausig       = parsdict["tausig"],
                                 Lz0          = parsdict["Lz0"])
                                          
            # Create dictionary of inner disc parameters
            innerDiscPar = dict(taumax       = parsdict["taumax_inner"],
                                sfrdecay     = parsdict["sfrdecay_inner"],
                                fehbirth     = parsdict["fehbirth_inner"],
                                tauenrich    = parsdict["tauenrich_inner"],
                                fehLzsolgrad = parsdict["fehLzsolgrad_inner"],
                                Lzsol        = parsdict["Lzsol_inner"],
                                fehsig       = parsdict["fehsig_inner"],
                                afemu        = parsdict["afemu_inner"],
                                afesig       = parsdict["afesig_inner"],
                                Rpmin        = parsdict["Rpmin_inner"],
                                Rpmax        = parsdict["Rpmax_inner"],
                                nfallmin     = parsdict["nfallmin_inner"],
                                nfallmax     = parsdict["nfallmax_inner"],
                                Rsigma       = parsdict["Rsigma_inner"],
                                sigmar0tau0  = parsdict["sigmar0tau0_inner"],
                                betar        = parsdict["betar_inner"],
                                sigmaz0tau0  = parsdict["sigmaz0tau0_inner"],
                                betaz        = parsdict["betaz_inner"])
            
            # Create dictionary of outer disc parameters
            outerDiscPar = dict(taumax       = parsdict["taumax_outer"],
                                sfrdecay     = parsdict["sfrdecay_outer"],
                                fehmu        = parsdict["fehmu_outer"],
                                fehsig       = parsdict["fehsig_outer"],
                                afemu        = parsdict["afemu_outer"],
                                afesig       = parsdict["afesig_outer"],
                                Rpmin        = parsdict["Rp_outer"],
                                Rpmax        = parsdict["Rp_outer"],
                                nfallmin     = parsdict["nfall_outer"],
                                nfallmax     = parsdict["nfall_outer"],
                                Rsigma       = parsdict["Rsigma_outer"],
                                sigmar0tau0  = parsdict["sigmar0tau0_outer"],
                                betar        = parsdict["betar_outer"],
                                sigmaz0tau0  = parsdict["sigmaz0tau0_outer"],
                                betaz        = parsdict["betaz_outer"])
            
            # Create Milky Way EDF instance
            mwedf = MilkyWayEDF.LowAlphaDiscEDF(gp,datamodel.solpos,
                                                parsdict["fOuterDisc"],
                                                globalDiscPar, 
                                                innerDiscPar,
                                                outerDiscPar)
            
            priorprob = mwedf.Prior()
            print("     Prior probability: "+str(priorprob))
            print(" ")
       
            if (priorprob==0.):
                print ("Parameter sample rejected. Trying again...")
                print(" ")
            else:
                break

        # Create Milky Way posterior instance
        mwpost  = Posterior(mwedf,af)
           
        # Create samples
        asp         = cmc.AnySkyPos(mwpost,minObs,maxObs)

        startTime   = time.time()
        samples     = asp.genSamples(nsamp)
        elapsedTime = time.time() - startTime
        print("     Took "+str(elapsedTime)+"s to generate samples.")
        print(" ")
        
        # Extract data
        ra     = samples[:,0]
        dec    = samples[:,1]
        s      = samples[:,2]
        vr     = samples[:,3]
        mura   = samples[:,4]
        mudec  = samples[:,5]
        feh    = samples[:,6]
        afe    = samples[:,7]
        age    = samples[:,8]

        # Randomly pick errors (percentage error for distance/age)
        Obs      = np.copy(datamodel.Obs)
        eObs     = np.copy(datamodel.eObs)
        era      = np.random.choice(eObs[:,0],nsamp)
        edec     = np.random.choice(eObs[:,1],nsamp)
        es       = np.random.choice(eObs[:,2]/Obs[:,2],nsamp)*s
        mulns    = np.log(s/np.sqrt(1.+es**2./s**2.))
        siglns   = np.sqrt(np.log(1.+es**2./s**2.))
        evr      = np.random.choice(eObs[:,3],nsamp)
        emura    = np.random.choice(eObs[:,4],nsamp)
        emudec   = np.random.choice(eObs[:,5],nsamp)
        efeh     = np.random.choice(eObs[:,6],nsamp)
        eafe     = np.random.choice(eObs[:,7],nsamp)
        eage     = np.random.choice(eObs[:,8]/Obs[:,8],nsamp)*age
        mulnage  = np.log(age/np.sqrt(1.+eage**2./age**2.))
        siglnage = np.sqrt(np.log(1.+eage**2./age**2.))
     
        # Add noise if desired
        if (ADDOBSNOISE):     
            ra    = np.random.normal(ra,era,size=nsamp)
            dec   = np.random.normal(dec,edec,size=nsamp)
            s     = np.random.lognormal(mulns,siglns,size=nsamp)
            vr    = np.random.normal(vr,evr,size=nsamp)
            mura  = np.random.normal(mura,emura,size=nsamp)
            mudec = np.random.normal(mudec,emudec,size=nsamp)
            feh   = np.random.normal(feh,efeh,size=nsamp)
            afe   = np.random.normal(afe,eafe,size=nsamp)
            age   = np.random.lognormal(mulnage,siglnage,size=nsamp)

        samplesMC[jmc,:,:] = np.column_stack((ra,dec,s,vr,mura,mudec,feh,afe,age))
        
    return parsMC, samplesMC
        
def main():
    
    parsMC, samplesMC = genLamostGaiaSamples()

    # Save to files
    np.savetxt(PARAMETERSFILE,parsMC)
    np.savetxt(SAMPLESFILE,samplesMC.flatten())
    
    return

#%% RUN MAIN  
if __name__ == "__main__":
    
    main()
    