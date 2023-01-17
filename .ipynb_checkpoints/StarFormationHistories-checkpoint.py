#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:37:48 2020

@author: daspal
"""

import numpy as np

def StarFormationRateSingleExponential(age,taumax,sfrdecay):
            
    """ STAR FORMATION RATE (PIFFL ET AL. 2014)

    Arguments:            
        age       - age (Gyr, [vector])
        taumax    - maximum age (Gyr, [scalar])
        sfrdecay  - star formation rate decay constant (Gyr, [scalar])
            
    Returns:
        Star formation rate (1/Gyr, [vector])
    """
            
    sfr = np.exp(age/sfrdecay)/(sfrdecay*(np.exp(taumax/sfrdecay) - 1.))
    
    # If age greater than maximum age, set SFR to zero
    index      = age>taumax
    sfr[index] = 0.
            
    return(sfr)

def StarFormationRateDoubleExponential(age,taumax,sfrdecay,sfrgrowth):
            
    """ STAR FORMATION RATE (SANDERS & BINNEY 2015) THERE'S A MISTAKE. NOT PROPERLY NORMALIZED.

    Arguments:            
        age       - age (Gyr, [vector])
        taumax    - maximum age (Gyr, [scalar])
        sfrdecay  - star formation rate decay constant (Gyr, [scalar])
        sfrgrowth - early star formation growth timescale (Gyr, [scalar])     
            
    Returns:
        Star formation rate (1/Gyr, [vector])
    """
        
    sfr = np.zeros_like(age)
        
    # For ages close to the age of the Galaxy, let star formation rate be zero
    index      = (taumax-age) > 0.001
    sfr[index] = np.exp(age[index]/sfrdecay - sfrgrowth/(taumax-age[index]))
        
    return(sfr)