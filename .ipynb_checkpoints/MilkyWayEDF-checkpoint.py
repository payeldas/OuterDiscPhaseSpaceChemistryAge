"""
MILKY WAY EDF
This module calculates the EDF probabilities.

Example:
    Initialize as mwedf = *****EDF()
    Returns EDF prob = mwedf(coords)
    
Author:
    Payel Das
    
To do:
    Nothing (I think).

"""
import numpy as np
import ChemicalDistributions as cd
import StarFormationHistories as sfh
import PhaseSpaceDistributions as psd

class LowAlphaDiscEDF:
    
    def  __init__(self,gp,solpos,fOuterDisc,globalDiscPar,innerDiscPar,outerDiscPar):
                      
        """ CLASS CONSTRUCTOR

        Arguments:
            gp           - Galactic potential instance [object]
            solpos       - Solar position and velocity (kpc,kpc,km/s,km/s,km/s, [vector])
            fOuterDisc   - weight on outer disc [-, scalar]
            globalDiscPar - parameters for both discs [dict]
                tausig       - parameter controlling velocity dispersions of stars born today (km/s, [scalar])
                Lz0          - scale angular momentum determining the suppression of retrograde orbits (kpc km/s, [scalar])        
            innerDiscPar - parameters for inner disc [dict]
                taumax       - age of disc (Gyr, [scalar])
                sfrdecay     - star formation rate decay constant (Gyr, [scalar])
                fehbirth	- [Fe/H] of oldest stars (dex, [scalar])
                tauenrich	- [Fe/H] enrichment timescale (Gyr, [scalar])
                fehLzsolgrad - [Fe/H]-Lz gradient at the Sun (dex/(km/s kpc), [scalar])
                Lzsol	     - z-component of angular momentum at the Sun (km/s kpc, [scalar])
                fehsig	     - dispersion in [Fe/H] at each Lz and age (dex, [scalar])     
                afemu        - mean [a/Fe] (dex, [scalar])
                afesig       - dispersion in [a/Fe] (dex, [scalar])
                Rpmin        - scale radius of oldest stars (kpc, [scalar])
                Rpmax        - scale radius of youngest stars (kpc, [scalar])
                nfallmin     - fall-off of oldest stars (-, [scalar])
                nfallmax     - fall-off of youngest stars (-, [scalar])
                Rsigma       - Velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
                sigmar0tau0  - radial velocity dispersion at solar radius (km/s, [scalar]) 
                betar        - growth of radial velocity dispersions with age
                sigmaz0tau0  - vertical velocity dispersion at solar radius (km/s, [scalar]) 
                betaz        - growth of vertical velocity dispersions with age
            outerDiscPar - parameters for outer disc [dict]
                taumax       - age of disc (Gyr, [scalar])
                sfrdecay     - star formation rate decay constant (Gyr, [scalar])
                fehmu        - mean [Fe/H] (dex, [scalar])
                fehsig       - dispersion in [Fe/H] (dex, [scalar])
                afemu        - mean [a/Fe] (dex, [scalar])
                afesig       - dispersion in [a/Fe] (dex, [scalar])
                Rpmin        - scale radius of oldest stars (kpc, [scalar])
                Rpmax        - scale radius of youngest stars (kpc, [scalar])
                nfallmin     - fall-off of oldest stars (-, [scalar])
                nfallmax     - fall-off of youngest stars (-, [scalar])
                Rsigma     -  velocity dispersion scale length of the youngest stars at the Galactic center (kpc, [scalar])
                sigmar0tau0 - radial velocity dispersion at solar radius (km/s, [scalar]) 
                betar       - growth of radial velocity dispersions with age
                sigmaz0tau0 - vertical velocity dispersion at solar radius (km/s, [scalar]) 
                betaz       - growth of vertical velocity dispersions with age

        Returns:
            Nothing
        """        
        
        # Share objects and variables
        self.gp            = gp
        self.solpos        = np.copy(solpos)
        self.fOuterDisc    = np.copy(fOuterDisc)
        self.globalDiscPar = globalDiscPar
        self.innerDiscPar  = innerDiscPar
        self.outerDiscPar  = outerDiscPar
               
        return
    
    def __call__(self,acts,xi):
          
        """ EDF PROBABILITY

        Arguments:
            acts         - actions in axisymmetric system ([Jr, Jz, Lz], (km/s kpc, km/s kpc, km/s kpc [matrix]))
            xi           - chemistry parameters ([[Fe/H], [a/Fe], age], (dex, dex, Gyr, [matrix])
        Returns:
            EDF probability [vector]
        """
        
        # Calculate guiding radius and frequencies
        Rc          = self.gp.Rcirc(L=np.atleast_2d(acts)[:,2])
        frequencies = self.CalcFreq(np.atleast_2d(acts)[:,2],Rc)
        
        edfprob = self.InnerDiscEDF(acts,xi,Rc,frequencies) + \
                  self.fOuterDisc*self.OuterDiscEDF(acts,xi,Rc,frequencies)
                          
        return(edfprob)
        
    def Prior(self):
        
        """ INTRINSIC EDF PRIOR PROBABILITY
        
        Arguments:
            None
                   
        Returns:
            Prior probability
        """
        
        prob = 1.
        if (self.globalDiscPar["tausig"] < 0.):
            prob = 0.
        if (self.globalDiscPar["Lz0"] <= 0.):
            prob = 0.
        if (self.innerDiscPar["taumax"] < 0.):
            prob = 0.
        if (self.innerDiscPar["sfrdecay"] < 0.):
            prob = 0.
        if (self.innerDiscPar["fehsig"] <= 0.):
            prob = 0.
        if (self.innerDiscPar["afesig"] <= 0.):
            prob = 0.
        if (self.innerDiscPar["Rpmin"] <=0. ):
            prob = 0.            
        if (self.innerDiscPar["Rpmax"] < self.innerDiscPar["Rpmin"]):
            prob = 0.
        if (self.innerDiscPar["nfallmin"] <=1. ):
            prob = 0.            
        if (self.innerDiscPar["nfallmax"] <=1. ):
            prob = 0. 
        if (self.innerDiscPar["nfallmax"] < self.innerDiscPar["nfallmin"]):
            prob = 0.
        if (self.innerDiscPar["Rsigma"] <=0. ):
            prob = 0. 
        if (self.innerDiscPar["sigmar0tau0"] <=0. ):
            prob = 0. 
        if (self.innerDiscPar["betar"] < 0. ):
            prob = 0. 
        if (self.innerDiscPar["sigmaz0tau0"] <=0. ):
            prob = 0. 
        if (self.innerDiscPar["betaz"] < 0. ):
            prob = 0. 
        if (self.outerDiscPar["taumax"] < 0.):
            prob = 0.
        if (self.outerDiscPar["sfrdecay"] < 0.):
            prob = 0.
        if (self.outerDiscPar["fehsig"] <= 0.):
            prob = 0.
        if (self.outerDiscPar["afesig"] <= 0.):
            prob = 0.        
        if (self.outerDiscPar["Rpmax"] < self.outerDiscPar["Rpmin"]):
            prob = 0.
        if (self.outerDiscPar["nfallmin"] <=1. ):
            prob = 0.            
        if (self.outerDiscPar["nfallmax"] <=1. ):
            prob = 0.
        if (self.outerDiscPar["nfallmax"] < self.outerDiscPar["nfallmin"]):
            prob = 0.
        if (self.outerDiscPar["Rsigma"] <=0. ):
            prob = 0.
        if (self.outerDiscPar["sigmar0tau0"] <=0. ):
            prob = 0. 
        if (self.outerDiscPar["betar"] < 0. ):
            prob = 0. 
        if (self.outerDiscPar["sigmaz0tau0"] <=0. ):
            prob = 0. 
        if (self.outerDiscPar["betaz"] < 0. ):
            prob = 0. 
            
        return(prob)
        
    def CalcFreq(self,Lz,Rc):
            
        """ CALCULATE FREQUENCIES OF CIRCULAR ORBIT WITH ANGULAR MOMENTUM Lz GIVEN POTENTIAL
            
        Arguments:
            Lz - z-component of angular momentum (km/s kpc, [vector])
                
        Returns:
            Kappa, nu, and omega frequencies (/s, [matrix])
        """
                
        xc           = np.column_stack((Rc,Rc*0.,Rc*0.))
        force, deriv = self.gp.forceDeriv(xc) # Returns force and force derivatives at (x, y, z)
        kappa        = np.sqrt(-deriv[:,0] - 3.*force[:,0]/xc[:,0])
        nu           = np.sqrt(-deriv[:,2])
        omega        = np.sqrt(-force[:,0]/xc[:,0])
        
        return(kappa,nu,omega)
        
    def InnerDiscEDF(self,acts,xi,Rc,frequencies):
                                        
        """ INNER DISC EDF PROBABILITY

        Arguments:
            acts        - actions in axisymmetric system ([Jr, Jz, Lz], (km/s kpc, km/s kpc, km/s kpc [matrix]))
            xi          - chemistry parameters ([[Fe/H], [a/Fe], age], (dex, dex, Gyr, [matrix])
            Rc          - circular radius with z component of angular momentum Lz (kpc, [vector])
            frequencies - potential frequencies (-, [matrix])
        Returns:
            Inner disc EDF probability [vector]
        """
                  
        # Galactic position
        acts = np.atleast_2d(acts)
        Jr   = acts[:,0]
        Jz   = acts[:,1]
        Lz   = acts[:,2]
         
        # Chemical parameters
        xi  = np.atleast_2d(xi)
        feh = xi[:,0]
        afe = xi[:,1]
        age = xi[:,2]
        
        # Calculate EDF probability
        sbprob = psd.SurfaceBrightnessDFWeibull(Lz,age,Rc,frequencies,
                                                self.globalDiscPar["Lz0"],
                                                self.innerDiscPar["taumax"],
                                                self.innerDiscPar["Rpmin"],
                                                self.innerDiscPar["Rpmax"],
                                                self.innerDiscPar["nfallmin"],
                                                self.innerDiscPar["nfallmax"])
        jrprob = psd.JrDFExponential(Jr,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.innerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.innerDiscPar["sigmar0tau0"],
                                     self.innerDiscPar["Rsigma"],
                                     self.innerDiscPar["betar"])
        jzprob = psd.JzDFExponential(Jz,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.innerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.innerDiscPar["sigmaz0tau0"],
                                     self.innerDiscPar["Rsigma"],
                                     self.innerDiscPar["betaz"])
        out = cd.ChemicalAbundanceDFSB(feh,Lz,age,
                                       self.innerDiscPar["taumax"],
                                       self.innerDiscPar["fehbirth"],
                                       self.innerDiscPar["tauenrich"],
                                       self.innerDiscPar["fehLzsolgrad"],
                                       self.innerDiscPar["Lzsol"],
                                       self.innerDiscPar["fehsig"])
        fehprob = out[1]
        afeprob = cd.ChemicalAbundanceDFNormal(afe,
                                               self.innerDiscPar["afemu"],
                                               self.innerDiscPar["afesig"])
        ageprob = sfh.StarFormationRateSingleExponential(age,
                                                         self.innerDiscPar["taumax"],
                                                         self.innerDiscPar["sfrdecay"])
        
        prob = sbprob*jrprob*jzprob*fehprob*afeprob*ageprob
        
        return(prob)
               
    def OuterDiscEDF(self,acts,xi,Rc,frequencies):
                  
        """ OUTER DISC EDF PROBABILITY

        Arguments:                            
            acts        - actions in axisymmetric system ([Jr, Jz, Lz], (km/s kpc, km/s kpc, km/s kpc [matrix]))
            xi          - chemistry parameters ([[Fe/H], [a/Fe], age], (dex, dex, Gyr, [matrix])
            Rc          - circular radius with z component of angular momentum Lz (kpc, [vector])
            frequencies - potential frequencies (-, [matrix])
    
        Returns:
            Outer disc EDF probability [vector]
        """
                  
        # Galactic position
        acts = np.atleast_2d(acts)
        Jr   = acts[:,0]
        Jz   = acts[:,1]
        Lz   = acts[:,2]
         
        # Chemical parameters
        xi  = np.atleast_2d(xi)
        feh = xi[:,0]
        afe = xi[:,1]
        age = xi[:,2]
        
        # Calculate EDF probability
        sbprob = psd.SurfaceBrightnessDFWeibull(Lz,age,Rc,frequencies,
                                                self.globalDiscPar["Lz0"],
                                                self.outerDiscPar["taumax"],
                                                self.outerDiscPar["Rpmin"],
                                                self.outerDiscPar["Rpmax"],
                                                self.outerDiscPar["nfallmin"],
                                                self.outerDiscPar["nfallmax"])
        jrprob = psd.JrDFExponential(Jr,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.outerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.outerDiscPar["sigmar0tau0"],
                                     self.outerDiscPar["Rsigma"],
                                     self.outerDiscPar["betar"])
        jzprob = psd.JzDFExponential(Jz,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.outerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.outerDiscPar["sigmaz0tau0"],
                                     self.outerDiscPar["Rsigma"],
                                     self.outerDiscPar["betaz"])
        fehprob = cd.ChemicalAbundanceDFNormal(feh,
                                               self.outerDiscPar["fehmu"],
                                               self.outerDiscPar["fehsig"])
        afeprob = cd.ChemicalAbundanceDFNormal(afe,
                                               self.outerDiscPar["afemu"],
                                               self.outerDiscPar["afesig"])
        ageprob = sfh.StarFormationRateSingleExponential(age,
                                                         self.outerDiscPar["taumax"],
                                                         self.outerDiscPar["sfrdecay"])
        
        prob = sbprob*jrprob*jzprob*fehprob*afeprob*ageprob
        
        return(prob)
    
class LowAlphaDiscEDF_2:
    
    def  __init__(self,gp,solpos,fOuterDisc,globalDiscPar,innerDiscPar,outerDiscPar):
                      
        """ CLASS CONSTRUCTOR

        Arguments:
            gp           - Galactic potential instance [object]
            solpos       - Solar position and velocity (kpc,kpc,km/s,km/s,km/s, [vector])
            fOuterDisc   - weight on outer disc [-, scalar]
            globalDiscPar - parameters for both discs [dict]
                tausig       - parameter controlling velocity dispersions of stars born today (km/s, [scalar])
                Lz0          - scale angular momentum determining the suppression of retrograde orbits (kpc km/s, [scalar])        
            innerDiscPar - parameters for stars dominating inner disc [dict]
                taumax       - age of disc (Gyr, [scalar])
                sfrdecay     - star formation rate decay constant (Gyr, [scalar])
                fehbirth     - [Fe/H] of oldest stars (dex, [scalar])
                tauenrich    - [Fe/H] enrichment timescale (Gyr, [scalar])
                fehLzsolgrad - [Fe/H]-Lz gradient at the Sun (dex/(km/s kpc), [scalar])
                Lzsol        - z-component of angular momentum at the Sun (km/s kpc, [scalar])
                fehsig       - dispersion in [Fe/H] at each Lz and age (dex, [scalar])     
                afemu        - mean [a/Fe] (dex, [scalar])
                afesig       - dispersion in [a/Fe] (dex, [scalar])
                Rpmin        - scale radius of oldest stars (kpc, [scalar])
                Rpmax        - scale radius of youngest stars (kpc, [scalar])
                nfallmin     - fall-off of oldest stars (-, [scalar])
                nfallmax     - fall-off of youngest stars (-, [scalar])
                Rsigma       - Velocity dispersion scale length of the youngest stars at the GC (kpc, [scalar])
                RsigmaOld    - Velocity dispersion scale length used for stars born before the outer disc and beyond the solar radius (kpc, [scalar])
                sigmar0tau0  - radial velocity dispersion at solar radius (km/s, [scalar]) 
                betar        - growth of radial velocity dispersions with age
                sigmaz0tau0  - vertical velocity dispersion at solar radius (km/s, [scalar]) 
                betaz        - growth of vertical velocity dispersions with age
            outerDiscPar - parameters for stars dominating outer disc [dict]
                taumax       - age of disc (Gyr, [scalar])
                sfrdecay     - star formation rate decay constant (Gyr, [scalar])
                fehmu        - mean [Fe/H] (dex, [scalar])
                fehsig       - dispersion in [Fe/H] (dex, [scalar])
                afemu        - mean [a/Fe] (dex, [scalar])
                afesig       - dispersion in [a/Fe] (dex, [scalar])
                Rpmin        - scale radius of oldest stars (kpc, [scalar])
                Rpmax        - scale radius of youngest stars (kpc, [scalar])
                nfallmin     - fall-off of oldest stars (-, [scalar])
                nfallmax     - fall-off of youngest stars (-, [scalar])
                Rsigma       - velocity dispersion scale length of the youngest stars at GC (kpc, [scalar])
                sigmar0tau0 - radial velocity dispersion at solar radius (km/s, [scalar]) 
                betar       - growth of radial velocity dispersions with age
                sigmaz0tau0 - vertical velocity dispersion at solar radius (km/s, [scalar]) 
                betaz       - growth of vertical velocity dispersions with age

        Returns:
            Nothing
        """        
        
        # Share objects and variables
        self.gp            = gp
        self.solpos        = np.copy(solpos)
        self.fOuterDisc    = np.copy(fOuterDisc)
        self.globalDiscPar = globalDiscPar
        self.innerDiscPar  = innerDiscPar
        self.outerDiscPar  = outerDiscPar
               
        return
    
    def __call__(self,acts,xi):
          
        """ EDF PROBABILITY

        Arguments:
            acts         - actions in axisymmetric system ([Jr, Jz, Lz], (km/s kpc, km/s kpc, km/s kpc [matrix]))
            xi           - chemistry parameters ([[Fe/H], [a/Fe], age], (dex, dex, Gyr, [matrix])
        Returns:
            EDF probability [vector]
        """
        
        # Calculate guiding radius and frequencies
        Rc          = self.gp.Rcirc(L=np.atleast_2d(acts)[:,2])
        frequencies = self.CalcFreq(np.atleast_2d(acts)[:,2],Rc)
        
        edfprob = self.InnerDiscEDF(acts,xi,Rc,frequencies) + \
                  self.fOuterDisc*self.OuterDiscEDF(acts,xi,Rc,frequencies)
                          
        return(edfprob)
        
    def Prior(self):
        
        """ INTRINSIC EDF PRIOR PROBABILITY
        
        Arguments:
            None
                   
        Returns:
            Prior probability
        """
        
        prob = 1.
        if (self.globalDiscPar["tausig"] < 0.):
            prob = 0.
        if (self.globalDiscPar["Lz0"] <= 0.):
            prob = 0.
        if (self.innerDiscPar["taumax"] < 0.):
            prob = 0.
        if (self.innerDiscPar["sfrdecay"] < 0.):
            prob = 0.
        if (self.innerDiscPar["fehsig"] <= 0.):
            prob = 0.
        if (self.innerDiscPar["afesig"] <= 0.):
            prob = 0.
        if (self.innerDiscPar["Rpmin"] <=0. ):
            prob = 0.            
        if (self.innerDiscPar["Rpmax"] < self.innerDiscPar["Rpmin"]):
            prob = 0.
        if (self.innerDiscPar["nfallmin"] <=1. ):
            prob = 0.            
        if (self.innerDiscPar["nfallmax"] <=1. ):
            prob = 0. 
        if (self.innerDiscPar["nfallmax"] < self.innerDiscPar["nfallmin"]):
            prob = 0.
        if (self.innerDiscPar["Rsigma"] <=0. ):
            prob = 0. 
        if (self.innerDiscPar["RsigmaOld"] <= self.innerDiscPar["Rsigma"] ):
            prob = 0. 
        if (self.innerDiscPar["sigmar0tau0"] <=0. ):
            prob = 0. 
        if (self.innerDiscPar["betar"] < 0. ):
            prob = 0. 
        if (self.innerDiscPar["sigmaz0tau0"] <=0. ):
            prob = 0. 
        if (self.innerDiscPar["betaz"] < 0. ):
            prob = 0. 
        if (self.outerDiscPar["taumax"] < 0.):
            prob = 0.
        if (self.outerDiscPar["taumax"] > 9.):
            prob = 0.
        if (self.outerDiscPar["sfrdecay"] < 0.):
            prob = 0.
        if (self.outerDiscPar["fehsig"] <= 0.):
            prob = 0.
        if (self.outerDiscPar["afesig"] <= 0.):
            prob = 0.        
        if (self.outerDiscPar["Rpmax"] < self.outerDiscPar["Rpmin"]):
            prob = 0.
        if (self.outerDiscPar["nfallmin"] <=1. ):
            prob = 0.            
        if (self.outerDiscPar["nfallmax"] <=1. ):
            prob = 0.
        if (self.outerDiscPar["nfallmax"] < self.outerDiscPar["nfallmin"]):
            prob = 0.
        if (self.outerDiscPar["Rsigma"] <=0. ):
            prob = 0.
        if (self.outerDiscPar["sigmar0tau0"] <=0. ):
            prob = 0. 
        if (self.outerDiscPar["betar"] < 0. ):
            prob = 0. 
        if (self.outerDiscPar["sigmaz0tau0"] <=0. ):
            prob = 0. 
        if (self.outerDiscPar["betaz"] < 0. ):
            prob = 0. 
            
        return(prob)
        
    def CalcFreq(self,Lz,Rc):
            
        """ CALCULATE FREQUENCIES OF CIRCULAR ORBIT WITH ANGULAR MOMENTUM Lz GIVEN POTENTIAL
            
        Arguments:
            Lz - z-component of angular momentum (km/s kpc, [vector])
                
        Returns:
            Kappa, nu, and omega frequencies (/s, [matrix])
        """
                
        xc           = np.column_stack((Rc,Rc*0.,Rc*0.))
        force, deriv = self.gp.forceDeriv(xc) # Returns force and force derivatives at (x, y, z)
        kappa        = np.sqrt(-deriv[:,0] - 3.*force[:,0]/xc[:,0])
        nu           = np.sqrt(-deriv[:,2])
        omega        = np.sqrt(-force[:,0]/xc[:,0])
        
        return(kappa,nu,omega)
        
    def InnerDiscEDF(self,acts,xi,Rc,frequencies):
                                        
        """ INNER DISC EDF PROBABILITY

        Arguments:
            acts        - actions in axisymmetric system ([Jr, Jz, Lz], (km/s kpc, km/s kpc, km/s kpc [matrix]))
            xi          - chemistry parameters ([[Fe/H], [a/Fe], age], (dex, dex, Gyr, [matrix])
            Rc          - circular radius with z component of angular momentum Lz (kpc, [vector])
            frequencies - potential frequencies (-, [matrix])
        Returns:
            Inner disc EDF probability [vector]
        """
                  
        # Galactic position
        acts = np.atleast_2d(acts)
        Jr   = acts[:,0]
        Jz   = acts[:,1]
        Lz   = acts[:,2]
         
        # Chemical parameters
        xi  = np.atleast_2d(xi)
        feh = xi[:,0]
        afe = xi[:,1]
        age = xi[:,2]
        
        # Calculate EDF probability
        sbprob = psd.SurfaceBrightnessDFWeibull(Lz,age,Rc,frequencies,
                                                self.globalDiscPar["Lz0"],
                                                self.innerDiscPar["taumax"],
                                                self.innerDiscPar["Rpmin"],
                                                self.innerDiscPar["Rpmax"],
                                                self.innerDiscPar["nfallmin"],
                                                self.innerDiscPar["nfallmax"])
        jrprob = psd.JrDFExponential(Jr,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.innerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.innerDiscPar["sigmar0tau0"],
                                     self.innerDiscPar["Rsigma"],
                                     self.innerDiscPar["betar"])
        jzprob = psd.JzDFExponential(Jz,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.innerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.innerDiscPar["sigmaz0tau0"],
                                     self.innerDiscPar["Rsigma"],
                                     self.innerDiscPar["betaz"])
        # Set df for stars beyond the solar radius and older than the starburst to be hotter
        index = (Lz > self.innerDiscPar["Lzsol"]) & (age > self.outerDiscPar["taumax"]):
        jrprob[index] = psd.JrDFExponential(Jr,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.innerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.innerDiscPar["sigmar0tau0"],
                                     self.innerDiscPar["Rsigma"],
                                     self.innerDiscPar["betar"])
        jzprob = psd.JzDFExponential(Jz,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.innerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.innerDiscPar["sigmaz0tau0"],
                                     self.innerDiscPar["Rsigma"],
                                     self.innerDiscPar["betaz"])
                                                                             
        out = cd.ChemicalAbundanceDFSB(feh,Lz,age,
                                       self.innerDiscPar["taumax"],
                                       self.innerDiscPar["fehbirth"],
                                       self.innerDiscPar["tauenrich"],
                                       self.innerDiscPar["fehLzsolgrad"],
                                       self.innerDiscPar["Lzsol"],
                                       self.innerDiscPar["fehsig"])
        fehprob = out[1]
        afeprob = cd.ChemicalAbundanceDFNormal(afe,
                                               self.innerDiscPar["afemu"],
                                               self.innerDiscPar["afesig"])
        ageprob = sfh.StarFormationRateSingleExponential(age,
                                                         self.innerDiscPar["taumax"],
                                                         self.innerDiscPar["sfrdecay"])
        
        prob = sbprob*jrprob*jzprob*fehprob*afeprob*ageprob
        
        return(prob)
               
    def OuterDiscEDF(self,acts,xi,Rc,frequencies):
                  
        """ OUTER DISC EDF PROBABILITY

        Arguments:                            
            acts        - actions in axisymmetric system ([Jr, Jz, Lz], (km/s kpc, km/s kpc, km/s kpc [matrix]))
            xi          - chemistry parameters ([[Fe/H], [a/Fe], age], (dex, dex, Gyr, [matrix])
            Rc          - circular radius with z component of angular momentum Lz (kpc, [vector])
            frequencies - potential frequencies (-, [matrix])
    
        Returns:
            Outer disc EDF probability [vector]
        """
                  
        # Galactic position
        acts = np.atleast_2d(acts)
        Jr   = acts[:,0]
        Jz   = acts[:,1]
        Lz   = acts[:,2]
         
        # Chemical parameters
        xi  = np.atleast_2d(xi)
        feh = xi[:,0]
        afe = xi[:,1]
        age = xi[:,2]
        
        # Calculate EDF probability
        sbprob = psd.SurfaceBrightnessDFWeibull(Lz,age,Rc,frequencies,
                                                self.globalDiscPar["Lz0"],
                                                self.outerDiscPar["taumax"],
                                                self.outerDiscPar["Rpmin"],
                                                self.outerDiscPar["Rpmax"],
                                                self.outerDiscPar["nfallmin"],
                                                self.outerDiscPar["nfallmax"])
        jrprob = psd.JrDFExponential(Jr,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.outerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.outerDiscPar["sigmar0tau0"],
                                     self.outerDiscPar["Rsigma"],
                                     self.outerDiscPar["betar"])
        jzprob = psd.JzDFExponential(Jz,age,Rc,frequencies,
                                     self.globalDiscPar["tausig"],
                                     self.outerDiscPar["taumax"],
                                     self.solpos[0],
                                     self.outerDiscPar["sigmaz0tau0"],
                                     self.outerDiscPar["Rsigma"],
                                     self.outerDiscPar["betaz"])
        fehprob = cd.ChemicalAbundanceDFNormal(feh,
                                               self.outerDiscPar["fehmu"],
                                               self.outerDiscPar["fehsig"])
        afeprob = cd.ChemicalAbundanceDFNormal(afe,
                                               self.outerDiscPar["afemu"],
                                               self.outerDiscPar["afesig"])
        ageprob = sfh.StarFormationRateSingleExponential(age,
                                                         self.outerDiscPar["taumax"],
                                                         self.outerDiscPar["sfrdecay"])
        
        prob = sbprob*jrprob*jzprob*fehprob*afeprob*ageprob
        
        return(prob)