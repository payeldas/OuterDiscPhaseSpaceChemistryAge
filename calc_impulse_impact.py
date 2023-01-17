"""
CALCULATE EVOLUTION IN DISTRIBUTION FUNCTION AFTER IMPULSE IMPACT WITH A PLUMMER SPHERE PERTURBER

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
    
        return(vKicks,perturbedDfSampleWc)
    
    def evolveDF(self,perturbedDfSampleWc,evolveTime,trajsize,potfile):
        
        """ EVOLVE PERTURBED DF ACCORDING TO SOME GRAVITATIONAL POTENTIAL FOR SOME TIME
        
        Arguments:
            perturbedDfSampleWc - Cartesian coordinates for sample of stars after impact ((kpc, km/s), [matrix]))
            evolveTime - time to evolve DF [Gyr]
            trajsize   - how many points to record along trajectory
            potfile    - name of potential file
        Returns:
            newDF - Cartesian coordinates for sample of stars after evolution ((kpc, km/s), [matrix]))
            
        """

        nstars              = len(perturbedDfSampleWc)
        
        # Evolve Df
            
        # Set units for the agama library
        agama.setUnits(mass=1,length=1,velocity=1)

        # McMillan galaxy potential
        gp  = agama.Potential(potfile) 
        
        # Evolve DF
        out = agama.orbit(potential=gp,ic=perturbedDfSampleWc,time=evolveTime*1000,trajsize=trajsize)  
        
        
        newDfSampleWc = np.copy(perturbedDfSampleWc)
        for jstars in range(nstars):
            newDfSampleWc[jstars,:] = np.copy(out[jstars][1][trajsize-1])
            
        return(newDfSampleWc)
