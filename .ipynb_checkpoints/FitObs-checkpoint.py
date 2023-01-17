# coding: utf-8
# FIT LAMOST DATA, VARYING PARAMETERS OF THE EDF AND GRAVITATIONAL POTENTIAL

"""
DEFINES POSTERIORS AND FITS LAMOST DATA, VARYING PARAMETERS OF THE EDF AND GRAVITATIONAL POTENTIAL
"""

import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import dill
import MilkyWayEDF
from time import time
from multiprocessing import Pool
import agama
import emcee

RUNMODE         = "mp"              # serial, mp (multiprocessing)
NMC             = 10                # Number of Monte Carlo sample for error convolution integral
FSTD            = 0.1               # Fraction of initial values of parameters to use as standard deviation of Gaussian ball starting point
NWALKERS        = 28                # Number of walkers to use (at least double number of parameters)
RELERR          = 1.e-06            # Relative error in integration
MAXEVAL         = 1e06              # Maximum number of evaluations in integration 
NITER           = 20                # Number of iterations in integration
SMOVE           = 0.0               # Percentage of move mixture to be sampled with stretch move
DEMOVE          = 0.8               # Percentage of move mixture to be sampled with differential evolution move
DESMOVE         = 0.2               # Percentage of move mixture to be sampled with differential evolution snooker move
NRUN            = 500               # Maximum number of runs
NCHECK          = 10                # Number of steps after which to check convergence 
RESTART         = False             # Whether restarting run
DISCSETUP       = "DoubleDisc"      # Configuration of thin disk
FILEROOT        = "run6"            # Old fileroot which to run from
DATAMODELFILE   = "../results/fit/"+DISCSETUP+"/datamodel_nmc"+str(NMC)+".dill" # Where to store pickled datamodel isntance
INITIALPARSFILE = "../setupinfo/"+DISCSETUP+"/initial_pars_boundaries.csv"      # Initial parameters file
SAMPLESFILE     = "../results/fit/"+DISCSETUP+"/"+FILEROOT+"_samples_nmc"+str(NMC)+"_nwalkers"+str(NWALKERS)+\
                  "_fstd"+str(FSTD)+"_stretch"+str(SMOVE)+"_de"+str(DEMOVE)+"_des"+str(DESMOVE)+"_fitall_newinitpars2.h5" 
#####################################################################################

#%% PREPARE FOR RUN

# Check move fractions add up to 1
totalmove = SMOVE+DEMOVE+DESMOVE
if (totalmove != 1.):
    print("Check move fractions. They don't add up to one.")

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

# Set agama units
agama.setUnits(mass=1,length=1,velocity=1)
    
#%%
def lnPost(fitpars):   
                   
        """ LOG POSTERIOR

        Arguments:
            fitpars - parameters being fitted ([vector])
            
        Returns:
            Nothing
        """        
    
        parsdict = initialparsdict.copy()
        
        # Print model parameters
        print("##########################################################")
        print("NEW MODEL:")
        print("     densityNorm_dh     = "+str(np.round(fitpars[0],6)))
        print("     scaleRadius_dh     = "+str(np.round(fitpars[1],6)))
        print("     fOuterDisc         = "+str(np.round(fitpars[2],3)))
        print("     taumax_inner       = "+str(np.round(fitpars[3],3)))
        print("     sfrgrowth_inner    = "+str(np.round(fitpars[4],3)))
        print("     tauenrich_inner    = "+str(np.round(fitpars[5],3)))
        print("     fehLzsolgrad_inner = "+str(np.round(fitpars[6],3)))
        print("     Rpmin_inner        = "+str(np.round(fitpars[7],3)))
        print("     Rpmax_inner        = "+str(np.round(fitpars[8],3)))
        print("     nfallmin_inner     = "+str(np.round(fitpars[9],3)))
        print("     nfallmax_inner     = "+str(np.round(fitpars[10],3)))
        print("     sigmaz0tau0_inner  = "+str(np.round(fitpars[11],3)))
        print("     betaz_inner        = "+str(np.round(fitpars[12],3)))
        print("     taumax_outer       = "+str(np.round(fitpars[13],3)))
        print("     sfrgrowth_outer    = "+str(np.round(fitpars[14],3)))
        print("     fehmu_outer        = "+str(np.round(fitpars[15],3)))
        print("     fehsig_outer       = "+str(np.round(fitpars[16],3)))
        print("     Rp_outer           = "+str(np.round(fitpars[17],3)))
        print("     nfall_outer        = "+str(np.round(fitpars[18],3)))
        print("     sigmaz0tau0_outer  = "+str(np.round(fitpars[19],3)))
        print("     betaz_outer        = "+str(np.round(fitpars[20],3)))
        print(" ")
                      
        # Populate potential and edf parameter dictionaries
        parsdict["densityNorm_dh"]     = fitpars[0]
        parsdict["scaleRadius_dh"]     = fitpars[1]
        parsdict["fOuterDisc"]         = fitpars[2]
        parsdict["taumax_inner"]       = fitpars[3]
        parsdict["sfrgrowth_inner"]    = fitpars[4]   
        parsdict["tauenrich_inner"]    = fitpars[5]   
        parsdict["fehLzsolgrad_inner"] = fitpars[6]   
        parsdict["Rpmin_inner"]        = fitpars[7]  
        parsdict["Rpmax_inner"]        = fitpars[8]  
        parsdict["nfallmin_inner"]     = fitpars[9]  
        parsdict["nfallmax_inner"]     = fitpars[10]  
        parsdict["sigmaz0tau0_inner"]  = fitpars[11]  
        parsdict["betaz_inner"]        = fitpars[12]
        parsdict["taumax_outer"]       = fitpars[13]
        parsdict["sfrgrowth_outer"]    = fitpars[14]
        parsdict["fehmu_outer"]        = fitpars[15]
        parsdict["fehsig_outer"]       = fitpars[16]
        parsdict["Rp_outer"]           = fitpars[17]
        parsdict["nfall_outer"]        = fitpars[18]
        parsdict["sigmaz0tau0_outer"]  = fitpars[19]
        parsdict["betaz_outer"]        = fitpars[20]

        # Create vectors of all parameters
        npars   = len(parsdict)
        pars    = np.zeros(npars)
        parsmin = np.zeros(npars)
        parsmax = np.zeros(npars)
        jkey    = 0
        for key in parsdict:
            pars[jkey]    = parsdict[key]
            parsmin[jkey] = parsmindict[key]
            parsmax[jkey] = parsmaxdict[key]
            #print(key+","+str(pars[jkey])+","+str(parsmin[jkey])+","+str(parsmax[jkey]))
            jkey += 1
            
        # Calculate prior (my choice of boundaries of all parameters)
        index = (parsmin<=pars) & (pars<=parsmax)
        prior = np.prod(index)
 
        # Only proceed if prior is 1
        if (prior==1.):
        
            # Create galaxy potential instance
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
            
            # Create dictionary of global disc parameters
            globalDiscPar = dict(tausig       = parsdict["tausig"],
                                 Lz0          = parsdict["Lz0"])
                                          
            # Create dictionary of inner disc parameters
            innerDiscPar = dict(taumax       = parsdict["taumax_inner"],
                                sfrdecay     = parsdict["sfrdecay_inner"],
                                sfrgrowth    = parsdict["sfrgrowth_inner"],
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
                                sfrgrowth    = parsdict["sfrgrowth_outer"],
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
         
            # Multiply prior by intrinsic EDF prior
            prior *= mwedf.Prior()            
            if (prior==1.):
        
                timestart = time()
            
                # Create action finder instance
                af = agama.ActionFinder(gp)
            
                # Define ln prior
                lnprior = 0.
            
                # Calculate new actions
                datamodel.CalcActs(af)
                timeacts  = time()
                timetaken = timeacts - timestart
                print("     ...actions evaluation took "+ str(timetaken) + "s.")
                print(" ")

                # Calculate likelihood
                lnlh,nstars_nozerolh = datamodel.CalcLnLh(mwedf)
                timelh     = time()
                timetaken  = timelh-timeacts
                print("     ...ln-likelihood evaluation took "+ str(timetaken) + "s.")
                print(" ")
            
                # Model normalization probability
                try:
                    lnmnp = nstars_nozerolh*datamodel.CalcLnMnp(mwedf,af,NITER,MAXEVAL,RELERR)
                except ZeroDivisionError:
                    print("ZeroDivisionError in model normalization probability. Rejecting model.")
                    print(" ")
                    lnpost  = -np.inf
                    return(lnpost,lnprior) 
                except OverflowError:
                    print("OverflowError in model normalization probability. Rejecting model.")
                    print(" ")
                    lnpost  = -np.inf
                    return(lnpost,lnprior) 
                except ValueError:
                    print("ValueError in model normalization probability. Rejecting model.")
                    print(" ")
                    lnpost  = -np.inf
                    return(lnpost,lnprior) 
                
                timemnp   = time()
                timetaken = timemnp-timelh
                print("     ...ln-mnp evaluation took "+ str(timetaken) + "s")
                print("     Total log model normalization:" + str(lnmnp))
                print (" ")
            
                # Total log posterior
                lnpost = lnprior+lnlh+lnmnp
                print("     Total log posterior:" + str(lnpost))
                print(" ")
        
                timemodel = time()
                print("MODEL EVALUATION TOOK "+str(timemodel-timestart)+"s.")
                print(" ")
                
                return(lnpost,lnprior)

            else:
                print("Infinite prior.")
                print(" ")
                lnprior = -np.inf
                lnpost  = -np.inf
                return(lnpost,lnprior)     
            
        else:
            print("Infinite prior.")
            print(" ")
            lnprior = -np.inf
            lnpost  = -np.inf
            return(lnpost,lnprior)  
        
#%%
def main():
    
    # Set random number seed
    random.seed(a=10)
    
    timestart = time()
    
    # Initial values of the fit parameters
    pars0 = np.array([initialparsdict["densityNorm_dh"],
                      initialparsdict["scaleRadius_dh"],
                      initialparsdict["fOuterDisc"],
                      initialparsdict["taumax_inner"],
                      initialparsdict["sfrgrowth_inner"],
                      initialparsdict["tauenrich_inner"],
                      initialparsdict["fehLzsolgrad_inner"],
                      initialparsdict["Rpmin_inner"],       
                      initialparsdict["Rpmax_inner"],       
                      initialparsdict["nfallmin_inner"],    
                      initialparsdict["nfallmax_inner"],    
                      initialparsdict["sigmaz0tau0_inner"], 
                      initialparsdict["betaz_inner"],       
                      initialparsdict["taumax_outer"],      
                      initialparsdict["sfrgrowth_outer"],    
                      initialparsdict["fehmu_outer"],      
                      initialparsdict["fehsig_outer"],      
                      initialparsdict["Rp_outer"],          
                      initialparsdict["nfall_outer"],       
                      initialparsdict["sigmaz0tau0_outer"], 
                      initialparsdict["betaz_outer"]]) 
                                   
    # Number of fit parameters and walkers
    nfitpars = len(pars0)
    
    print("SETTING UP emcee RUN USING "+str(NWALKERS)+" WALKERS...")
    print(" ")
    
    # Define backend 
    if (RESTART):
        backend = emcee.backends.HDFBackend(SAMPLESFILE)          
    else:
        backend = emcee.backends.HDFBackend(SAMPLESFILE)
        backend.reset(NWALKERS,nfitpars)
     
    # Run accordint to selected mode
    if (RUNMODE=="serial"):
        
        print("...and running in serial.")
            
        # Initialize the sampler
        sampler   = emcee.EnsembleSampler(NWALKERS,        # the number of Goodman & Weare “walkers"
                                          nfitpars,        # number of dimensions in the parameter space
                                          lnPost,          # posterior function
                                          moves=[(emcee.moves.StretchMove(live_dangerously=True),SMOVE), 
                                                 (emcee.moves.DEMove(live_dangerously=True),DEMOVE),
                                                 (emcee.moves.DESnookerMove(live_dangerously=True),DESMOVE),],
                                          backend=backend)
        
        # Define starting point
        if (RESTART):
            start = sampler.get_last_sample()
        else:
            stds    = FSTD*pars0
            start   = emcee.utils.sample_ball(pars0,stds,size=NWALKERS)
            
        # Track average autocorrelation time to help track convergence
        index    = 0
        autocorr = np.empty(NRUN)
        oldTau   = np.inf
    
        # Sample for NRUN steps
        for sample in sampler.sample(start,iterations=NRUN,progress=True):
            # Check convergence every NCHECK steps
            if sampler.iteration % NCHECK:
                continue
            
            # Compute the autocorrelation time so far (using tol=0 means that we'll
            # always get an estimate even if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
            
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(oldTau - tau) / tau < 0.01)
            if converged:
                break
            oldTau = tau
            
    elif (RUNMODE=="mp"):
        
        print("...and running with multiprocessing.")
        
        # Use multiprocessing
        with Pool() as pool:
            
            # Initialize the sampler
            sampler   = emcee.EnsembleSampler(NWALKERS,        # the number of Goodman & Weare “walkers"
                                              nfitpars,        # number of dimensions in the parameter space
                                              lnPost,          # posterior function
                                              moves=[(emcee.moves.StretchMove(live_dangerously=True),SMOVE), 
                                                     (emcee.moves.DEMove(live_dangerously=True),DEMOVE),
                                                     (emcee.moves.DESnookerMove(live_dangerously=True),DESMOVE),],
                                              backend=backend,
                                              pool=pool) 
            
            # Define starting point
            if (RESTART):
                start = sampler.get_last_sample()
            else:
                stds    = FSTD*pars0
                start   = emcee.utils.sample_ball(pars0,stds,size=NWALKERS)
            
            # Track average autocorrelation time to help track convergence
            index    = 0
            autocorr = np.empty(NRUN)
            oldTau   = np.inf
    
            # Sample for NRUN steps
            for sample in sampler.sample(start,iterations=NRUN,progress=True):
                # Check convergence every NCHECK steps
                if sampler.iteration % NCHECK:
                    continue
            
                # Compute the autocorrelation time so far (using tol=0 means that we'll
                # always get an estimate even if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
            
                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(oldTau - tau) / tau < 0.01)
                if converged:
                    break
                oldTau = tau
                    
    else:
        print("RUNMODE not recognised.")
            
    # Print mean acceptance fraction
    print("Mean acceptance fraction: {0:.3f}"
          .format(np.mean(sampler.acceptance_fraction)))
    
    # Print total time for run
    timeend   = time()
    timetaken = (timeend - timestart)
    print("Total time taken for run: {0:.3f}"
          .format(timetaken))
    
  
#%%     
if __name__ == "__main__":
    main()