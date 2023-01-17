"""
CLASS CONSTRUCTING DATA SAMPLES FOR FITTING THE MODEL (i.e. EDF AND POTENTIAL). 
CALCULATES MC, JACOBIAN DETERMINANT, AND SF GIVEN OBSERVABLES.
CALCULATES POSTERIOR SAMPLES GIVEN OBSERVABLES AND MODEL
"""
import numpy as np
import CoordTrans as ct
import os 
import dill as dill
import vegas

class DataModel:
  
    def  __init__(self,Obs,eObs,mwsf,solpos):
   
        """ CLASS CONSTRUCTOR

        Arguments:
            Obs    - observations in Equatorial coordinates, [Fe/H], [a/Fe], age ([matrix])
            eObs   - error in observations in Equatorial coordinates, [Fe/H], [a/Fe], age ([matrix])
            mwsf   - selection function for sample
            solpos - solar position ([vector])           
        Returns:
            Nothing
        """ 
        
        # Define observational volume
        print("...defining observational volume...")
        print (" ")

        ve        = 1000.
        abspmmax  = 100.
        minObs    = np.min(Obs,0)
        maxObs    = np.max(Obs,0)
        minObs[0] = 0.
        maxObs[0] = 2.*np.pi
        minObs[1] = -np.pi/2.
        maxObs[1] = np.pi/2.
        minObs[2] = 0.01
        maxObs[2] = 26. 
        minObs[3] = -ve
        maxObs[3] = ve
        minObs[4] = -abspmmax
        maxObs[4] = abspmmax
        minObs[5] = -abspmmax
        maxObs[5] = abspmmax
        minObs[6] = -1.0
        maxObs[6] = 0.5
        minObs[7] = -0.2
        maxObs[7] = 0.2
        minObs[8] = 0.1
        maxObs[8] = 13.1     
        
        print ("Minima of observational volume:")
        print(minObs)
        print(" ")
        print ("Maxima of observational volume:")
        print(maxObs)
        print(" ")
        
        
        # Share variables and objects with class
        self.Obs    = np.copy(Obs)
        self.eObs   = np.copy(eObs)
        self.minObs = np.copy(minObs)
        self.maxObs = np.copy(maxObs)
        self.mwsf   = mwsf
        self.solpos = np.copy(solpos)
        self.nstars = len(Obs)
        
    def CalcSF(self):
        
        """ CALCULATE SELECTION FUNCTION FOR OBSERVABLES

        Arguments:
            Nothing
        Returns:
            Nothing
        """
        
        # Share variable and object
        Obs  = np.copy(self.Obs)
        mwsf = self.mwsf
        
        # Calculate selection function for each star
        xe    = np.copy(Obs[:,0:3])
        xi    = np.copy(Obs[:,6:])            
        obsSf = mwsf(xe,xi)
        
        # Share variables
        self.obsSf = np.copy(obsSf)
        
        return
                
    def CreateMcObs(self,nmc,mcfile):
        
        """ CREATE MONTE CARLO OBSERVABLES

        Arguments:
            nmc    - number of Monte Carlo samples ([scalar])
            mcfile - file with Monte Carlo data
        Returns:
            Nothing
        """
        
        # Make local copies of shared variables
        Obs    = np.copy(self.Obs)
        eObs   = np.copy(self.eObs)   
        nstars = np.copy(self.nstars)
        
        # If mcfile already exists, read in, otherwise calculate Monte Carlo observables
        if (os.path.isfile(mcfile)):
            print("     Monte Carlo samples already evaluated. Reading in...")
            mcObs = np.loadtxt(mcfile)
            mcObs = np.reshape(mcObs,[nstars,nmc,-1])
        else:
            # Ra, Dec, s, vlos,PMRa,PMDec,[M/H],[a/M],age
            print("     Monte Carlo samples being evaluated...")
            nstars,nobs = np.shape(Obs)  
            mcObs       = np.zeros([nstars,nmc,nobs])
            for jstars in range(nstars):
                for jobs in range(nobs):
                    # Assume lognormal for distance and age
                    if ((jobs==2)|(jobs==8)):
                        mulnobs              = np.log(Obs[jstars,jobs]/np.sqrt(1.+eObs[jstars,jobs]**2./Obs[jstars,jobs]**2.))
                        siglnobs             = np.sqrt(np.log(1.+eObs[jstars,jobs]**2./Obs[jstars,jobs]**2.))+1.e-06
                        mcObs[jstars,:,jobs] = np.random.lognormal(mulnobs,siglnobs,nmc)
                    else:
                        # Otherwise assume normal distribution (ensure non-zero dispersion)
                        mcObs[jstars,:,jobs] = np.random.normal(Obs[jstars,jobs],eObs[jstars,jobs]+1.e-06,nmc)
                         
            # Save to file
            np.savetxt(mcfile,np.reshape(mcObs,[nstars*nmc,nobs]),fmt='%15.5e') 	
                         
        # Share variables
        self.nmc    = np.copy(nmc)
        self.mcObs  = np.copy(mcObs)	
    
        return
        
    def CalcJacDet(self):
        
        """ CALCULATE JACOBIAN DETERMINANT FOR MONTE CARLO SAMPLES

        Arguments:
            nmc    - number of Monte Carlo samples ([scalar])
            mcfile - file with Monte Carlo data
        Returns:
            Nothing
        """
        
        # Share variables
        mcObs  = np.copy(self.mcObs)
        nstars = np.copy(self.nstars)
        nmc    = np.copy(self.nmc)
        
        mcJacdet = np.zeros([nstars,nmc])
        for jstars in range(nstars):
            
            we = mcObs[jstars,:,0:6]
            
            # Calculate Jacobian determinant and share
            mcJacdet[jstars,:] = we[:,2]**4. * np.cos(we[:,1])
            
        # Share variables
        self.mcJacdet = np.copy(mcJacdet)	
        
        return(mcJacdet)
        
    def CalcActs(self,af):
        
        """ CALCULATE ACTIONS FOR MONTE CARLO SAMPLES

        Arguments:
            af - action finder ([object])
        Returns:
            Nothing
        """
        
        # Share variables
        mcObs  = np.copy(self.mcObs)
        nstars = np.copy(self.nstars)
        nmc    = np.copy(self.nmc)
        solpos = np.copy(self.solpos)
        
        # Calculate actions        
        nacts  = 3
        mcActs = np.zeros([nstars,nmc,nacts])
        
        for jstars in range(self.nstars):
            
            # Calculate Cartesian coordinates of each star for the Monte Carlo samples
            we = mcObs[jstars,:,0:6]
            wg = ct.EquatorialToGalactic(we)
            wc = ct.GalacticToCartesian(wg,solpos)
            mcActs[jstars,:,:] = af(wc)
            
        # Share variables
        self.mcActs = np.copy(mcActs)
        
        return
        
    def CalcLnLh(self,mwedf):   
        
        """ CALCULATE LOG LIKELIHOOD FOR MODEL AND OBSERVABLES

        Arguments:
            Nothing
        Returns:
            lnlh            - log likelihood ([scalar])
            nstars_nozerolh - number of stars with non-zero likelihood ([scalar])
        """

        # Share variables
        mcObs    = np.copy(self.mcObs)  
        mcJacdet = np.copy(self.mcJacdet)
        mcActs   = np.copy(self.mcActs)   
        obsSf    = np.copy(self.obsSf)
        nstars   = np.copy(self.nstars)
        nmc      = np.copy(self.nmc)         

        # Flatten arrays
        mcObs_flatten    = np.reshape(mcObs,[nstars*nmc,-1])    
        mcJacdet_flatten = np.reshape(mcJacdet,nstars*nmc)
        mcActs_flatten   = np.reshape(mcActs,[nstars*nmc,-1])
        
        # Get chemical parameters
        xi_flatten = mcObs_flatten[:,6:]
            
        # Calculate DF for each Monte Carlo sample, setting to zero for nan actions or DF
        mcDf_flatten              = mcJacdet_flatten*mwedf(mcActs_flatten,xi_flatten)
        inanacts                  = np.isnan(mcActs_flatten)
        inanacts_1d               = np.any(inanacts==True,1)
        mcDf_flatten[inanacts_1d] = 0.
        ibaddf                    = np.isnan(mcDf_flatten) | np.isinf(mcDf_flatten)
        mcDf_flatten[ibaddf]      = 0.
            
        # Calculate likelihood for each star
        mcDf  = np.reshape(mcDf_flatten,[nstars,-1])    
        lh    = np.sum(np.multiply(np.transpose([obsSf,]*nmc),mcDf),1)
    
        # Remove stars with zero likelihood
        index  = lh>0.
        lh     = lh[index]
        nstars_nozerolh = np.sum(index)
        print("     Removed "+str(nstars-nstars_nozerolh)+" stars with zero likelihood.")

        # Calculate total ln-likelihood of all stars 
        lnlh = np.sum(np.log(lh))
        print("     Total log likelihood: " + str(lnlh))
                
        return(lnlh,nstars_nozerolh)
    
    def CalcLnMnp(self,mwedf,af,niter,neval,relerr): 
        
        """ CALCULATE LOG NORMALIZATION PROBABILITY FOR MODEL AND OBSERVABLES

        Arguments:
            mwedf   - Milky Way EDF instance ([object])
            af      - action finder instance ([object])
            niter   - maximum number of iterations in integral
            neval   - maximum number of evaluations in integral
            relerr  - acceptable relative error
        Returns:
            lnmnp  - log normalization probability ([scalar])
        """
        
        # Share variables and objects
        minObs = self.minObs
        maxObs = self.maxObs
        mwsf   = self.mwsf
        solpos = self.solpos
        
        # Define integrand for model normalization probability
        @vegas.batchintegrand
        def integrand(star): 
            star     = np.atleast_2d(star)
            we       = star[:,0:6]
            wg       = ct.EquatorialToGalactic(we)
            wc       = ct.GalacticToCartesian(wg,solpos)
            acts     = af(wc)
            ra       = star[:,0]
            dec      = star[:,1]
            s        = star[:,2]
            feh      = star[:,6]
            afe      = star[:,7]
            age      = star[:,8]
            xe       = np.column_stack((ra,dec,s))
            xi       = np.column_stack((feh,afe,age))
            jacdet   = s**4*np.cos(dec)
            sfprob   = mwsf(xe,xi) 
            edfprob  = mwedf(acts,xi)
            
            # Set any nans to zero (because of nan actions). No prior needed because assume that it's met.
            out                    = jacdet*sfprob*edfprob
            out[np.isnan(edfprob)] = 0.
            return(out)

        # Create integration object
        integ = vegas.Integrator([[minObs[0],maxObs[0]],
                                  [minObs[1],maxObs[1]],
                                  [minObs[2],maxObs[2]],
                                  [minObs[3],maxObs[3]],
                                  [minObs[4],maxObs[4]],
                                  [minObs[5],maxObs[5]],
                                  [minObs[6],maxObs[6]],
                                  [minObs[7],maxObs[7]],
                                  [minObs[8],maxObs[8]]])       
        
        # Train integration object
        integ(integrand,nitn=10,neval=1000)      
                
        # Calculate integration
        result = integ(integrand,nitn=niter,neval=neval)
        val    = result.mean
        err    = result.sdev
                
        # Dill integration instance
        with open("../results/mnp/integral.dill", "wb") as output:
            dill.dump(integ,output,dill.HIGHEST_PROTOCOL)  
                    
        pererr = err/val*100.
        if (pererr > relerr*100.):
            print("     mnp integral error is "+str(np.round(pererr,2))+"%.")
            
        # Calculate ln model normalization per star
        lnmnp = np.log(val)

        return(lnmnp) 