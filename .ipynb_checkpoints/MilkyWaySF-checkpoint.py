"""
MILKY WAY SELECTION FUNCTION
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
import dill as dill

class LamostRedClumpsLowAlphaDiscSF:
    
    def  __init__(self,sffile):
                      
        """ CLASS CONSTRUCTOR

        Arguments:
            Nothing
            
        Returns:
            Nothing
        """        
               
        # Load selection function            
        with open(sffile, "rb") as input:
            lamostsf = dill.load(input,encoding='latin1') 
            
        # Share object
        self.lamostsf = lamostsf
    
        return
        
    def __call__(self,xe,xi):
          
        """ SF PROBABILITY

        Arguments:
            xe  - Galactic position in Equatorial coordinates ([ra, dec, s], (rad, rad, kpc, [matrix]))
            xi  - chemistry parameters ([[Fe/H], [a/Fe], age], (dex, dex, Gyr, [matrix])
    
        Returns:
            SF probability [vector]
        """
                  
        # Galactic position
        xe  = np.atleast_2d(xe)
        ra  = xe[:,0]
        dec = xe[:,1]
        s   = xe[:,2]
         
        # Chemical parameters
        xi  = np.atleast_2d(xi)
        feh = xi[:,0]
        age = xi[:,2]
        
        sfprob = self.lamostsf((ra/np.pi*180.,dec/np.pi*180.,s,feh,age))
            
        # Get rid of merged massive stars
        #index = ((afe>0.12) & (age<5.))
        #sfprob[index] = 0.
        
        # Select low-alpha population
        #slope      = -0.07
        #yintercept = 0.12
        #afemax     = yintercept + slope*feh
        #index      = afemax > afe
        #sfprob[index]  = 0.
        
        return(sfprob)
                             