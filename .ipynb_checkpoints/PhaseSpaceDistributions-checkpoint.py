#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:41:47 2020

@author: daspal
"""

import numpy as np

def SurfaceBrightnessDFWeibull(Lz,age,Rc,frequencies,Lz0,taumax,Rpmin,Rpmax,nfallmin,nfallmax):
        
    """ SURFACE-BRIGHTNESS WEIBULL DISTRIBUTION FUNCTION GIVEN Lz, age
         
    Arguments:
        Lz          - z component of angular momentum (km/s kpc)
        age         - age of coeval population (Gyr, [vector])
        Rc          - circular radius with angular momentum Lz (kpc, [vector])
        frequencies - frequencies given potential (-, [matrix])
        Lz0         - scale angular momentum determining the suppression of retrograde orbits (kpc km/s, [scalar])
        taumax      - maximum age (Gyr, [scalar])
        Rpmin       - scale radius of oldest stars (kpc, [scalar])
        Rpmax       - scale radius of youngest stars (kpc, [scalar])
        nfallmin    - fall-off of oldest stars (-, [scalar])
        nfallmax    - fall-off of youngest stars (-, [scalar])
        
    Returns:    
        Surface-brightness DF probability.
    """
        
    # Frequencies
    kappa,nu,omega = frequencies
    
    # Surface brightness
    nfall    = nfallmax - (nfallmax-nfallmin)*(age/taumax)
    Rp       = Rpmin+(Rpmax-Rpmin)*(taumax-age)/taumax 
    Rpprime  = (nfall/(nfall-1))**(1/nfall)*Rp
    fsb      = omega/kappa**2. * (nfall/Rpprime) * \
               (Rc/Rpprime)**(nfall-1) * \
               np.exp(-(Rc/Rpprime)**nfall)

    # Rotation probability
    frot    = 1. + np.tanh(Lz/Lz0)
        
    return(fsb*frot) 
                
def JrDFExponential(Jr,age,Rc,frequencies,tausig,taumax,R0,sigmar0tau0,Rsigmar,betar):
         
    """ RADIAL ACTION EXPONENTIAL DISTRIBUTION FUNCTION GIVEN Jr, Lz, age
         
    Arguments:
        Jr          - radial action (km/s kpc, [vector])
        age         - age of coeval population (Gyr, [vector])
        Rc          - circular radius with angular momentum Lz (kpc, [vector])
        frequencies - frequencies given potential (-, [matrix])
        tausig      - parameter controlling velocity dispersions of stars born today (km/s, [scalar])
        taumax      - maximum age (Gyr, [scalar])
        R0          - solar radius (kpc, [scalar])
        sigmar0tau0 - radial velocity dispersion at solar radius (km/s, [scalar]) 
        Rsigmar     - R velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
        betar       - growth of radial velocity dispersions with age (-, [scalar])
 
    Returns:    
        Radial motion DF probability.
    """
        
    # Frequencies
    kappa,nu,omega = frequencies 
        
    # Velocity dispersion (heating with age)
    intermediate = ((age+tausig)/(taumax+tausig))
    sigmar0      = sigmar0tau0*intermediate**betar
    sigmar_sq    = (sigmar0*np.exp((R0-Rc)/Rsigmar))**2
        
    # Dispersion probability   
    fsigmar = kappa/(sigmar_sq)*np.exp(-kappa*Jr/sigmar_sq)
        
    return(fsigmar)

def JzDFExponential(Jz,age,Rc,frequencies,tausig,taumax,R0,sigmaz0tau0,Rsigmaz,betaz):
         
    """ VERTICAL ACTION EXPONENTIAL DISTRIBUTION FUNCTION GIVEN Jz, Lz, age
         
    Arguments:
        Jz          - vertical action (km/s kpc, [vector])
        age         - age of coeval population (Gyr, [vector])
        Rc          - circular radius with angular momentum Lz (kpc, [vector])
        frequencies - frequencies given potential (-, [matrix])
        tausig      - parameter controlling velocity dispersions of stars born today (km/s, [scalar])
        taumax      - maximum age (Gyr, [scalar])
        R0          - solar radius (kpc, [scalar])
        sigmaz0tau0 - vertical velocity dispersion at solar radius (km/s, [scalar]) 
        Rsigmaz     - R velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
        betaz       - growth of vertical velocity dispersions with age (-, [scalar])
 
    Returns:    
        Vertical motion DF probability.
    """
        
    # Frequencies
    kappa,nu,omega = frequencies 
        
    # Velocity dispersion (heating with age)
    intermediate = ((age+tausig)/(taumax+tausig))
    sigmaz0      = sigmaz0tau0*intermediate**betaz
    sigmaz_sq    = (sigmaz0*np.exp((R0-Rc)/Rsigmaz))**2
        
    # Dispersion probability   
    fsigmaz = nu/(sigmaz_sq)*np.exp(-nu*Jz/sigmaz_sq)
        
    return(fsigmaz)