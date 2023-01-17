"""
CALCULATE EVOLUTION IN DISTRIBUTION FUNCTION AFTER IMPULSE IMPACT WITH A PERTURBER
This module calculates the SF probabilities.

Example:
    Initialize as sfdf = *****SF()
    Returns SF prob = sfdf(coords)
    
Author:
    Payel Das
    
To do:
    Nothing (I think).

"""
import numpy as np
import pathos.multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import agama as agama

# Constants
grav  = 4.3e-6 #(kpc/msun (km/s)^2)

class CalcImpulseImpact:
    
    def  __init__(self):
                      
        """ CLASS CONSTRUCTOR

        Arguments:
            None
            
        Returns:
            Nothing
        """        
    
        return
    
    def calcKicks(self,m0,rs,relWc):
        
        """ CALCULATE VELOCITY KICKS
        
        Arguments:
            m0    - mass of perturber [float]
            rs    - scale radius of perturber [float]
            relWc - relative velocity coordinates between stars and a perturber ((kpc, km/s), [matrix])
    
        Returns:
            vKicks - kicks to the velocity (km/s), [matrix])
        """
        
        # Calculate impact parameters and relative velocities
        bmag  = np.linalg.norm(relWc[:,0:3],axis=1)
        vrmag = np.linalg.norm(relWc[:,4:],axis=1)
    
        # Calculate kicks
        chi2  = bmag**2 + rs**2
        vKick = (2.*grav*m0*np.transpose(relWc[:,0:3]))/(vrmag*chi2)
    
        return(np.transpose(vKick))
    
    def perturbDF(self,dfSampleWc,perturbMass,perturbRs,perturbImpactWc):
        
        """ PERTURB DF 
        
        Arguments:
            dfSampleWc      - Cartesian coordinates for sample of stars before impact ((kpc, km/s), [matrix]))
            perturbMass     - mass of perturber (Msun)
            perturbRs       - scale radius of Plummer sphere describing perturber (kpc)
            perturbImpactWc - coordinates of perturber at impact ((kpc, km/s), [vector]))
    
        Returns:
            vKicks              - kicks to the velocity [3-element array]
            perturbedDfSampleWc - Cartesian coordinates for sample of stars after impact ((kpc, km/s), [matrix]))
            
        """
        # Get number of stars
        nstars = len(dfSampleWc)
        
        # Calculate relative coordinate vector between impact coordinates and sample stars
        relWc = dfSampleWc - perturbImpactWc
        
        vKicks = self.calcKicks(perturbMass,perturbRs,relWc)

        perturbedDfSampleWc      = np.copy(dfSampleWc)
        perturbedDfSampleWc[:,3] = dfSampleWc[:,3] + vKicks[:,0]
        perturbedDfSampleWc[:,4] = dfSampleWc[:,4] + vKicks[:,1]
        perturbedDfSampleWc[:,5] = dfSampleWc[:,5] + vKicks[:,2]
        
        # Share variables
        self.nstars              = np.copy(nstars)
    
        return(vKicks,perturbedDfSampleWc)
    
    def evolveDF(self,perturbedDfSampleWc,evolveTime,trajsize,ncores = None):
        
        """ EVOLVE PERTURBED DF ACCORDING TO SOME GRAVITATIONAL POTENTIAL FOR SOME TIME
        
        Arguments:
            perturbedDfSampleWc - Cartesian coordinates for sample of stars after impact ((kpc, km/s), [matrix]))
            gp         - AGAMA gravitational potential [object]
            evolveTime - time to evolve DF [Gyr]
            trajsize   - how many paints to record along trajectory
            ncores     - number of cores to use for multiprocessing [optional]
  
        Returns:
            newDF - Cartesian coordinates for sample of stars after evolution ((kpc, km/s), [matrix]))
            
        """

        nstars              = np.copy(self.nstars)
        
        # Evolve Df
        if (ncores == None):
            ncores = pathos.multiprocessing.cpu_count()
        print("Running orbit integrator using "+str(ncores)+" ncores...")

        a_pool = Pool(ncores)
        def orbitIntegrate(wc):
            
            # Set units for the agama library
            agama.setUnits(mass=1,length=1,velocity=1)

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
                                  densityNorm=1.81556e+07,
                                  axisRatioZ=1.0,
                                  gamma=1.,
                                  beta=3.,
                                  scaleRadius=14.4336)
            galpot_par = [thindisc_par,thickdisc_par,HIgasdisc_par,HIIgasdisc_par,bulge_par,dmhalo_par]
            gp         = agama.Potential(galpot_par[0],galpot_par[1],galpot_par[2],galpot_par[3],galpot_par[4],galpot_par[5]) 
            out = agama.orbit(potential=gp,ic=wc,time=evolveTime*1000,trajsize=trajsize)  
            return(out)
        
        result = a_pool.map(orbitIntegrate,perturbedDfSampleWc)
        
        newDfSampleWc = np.copy(perturbedDfSampleWc)
        for jstars in range(nstars):
            newDfSampleWc[jstars,:] = np.copy(result[jstars][1][trajsize-1])
            
        return(newDfSampleWc)
