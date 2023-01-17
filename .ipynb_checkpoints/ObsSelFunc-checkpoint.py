"""
LAMOST RED CLUMPS SELECTION FUNCTION

This module calculates the selection function for the LAMOST red clumps given 
Ra, Dec, [Fe/H], [a/Fe], and age.

Example:
    Initialize as lsf = LamostSelFunc(ramin,ramax,decmin,decmax,kmin,kmax,jminkmin,jminkmax)
    Returns SF prob = lsf(ra,dec,s,mh,age)
    
Author:
    Payel Das
    
To do:
    Nothing (I think).

"""
import numpy as np
import CoordTrans as ct
from scipy.interpolate import NearestNDInterpolator,RegularGridInterpolator
import dill as dill
import DistFunc as df
import time
import matplotlib.pyplot as plt

class LamostSelFunc:
    ## CLASS CONSTRUCTOR
    # ramin/ramax       - minimum/maximum right ascenscion
    # decmin/decmax     - minimum/maximum declination
    # kmin/kmax         - minimum/maximum apparent K-band magnitude
    # jminkmin/jminkmax - minimum/maximum J-K colour
    def  __init__(self,ramin,ramax,decmin,decmax,kmin,kmax,jminkmin,jminkmax):
        
        """ CLASS CONSTRUCTOR

        Arguments:
            ramin/ramax       - minimum/maximum right ascenscion (deg, [scalar])
            decmin/decmax     - minimum/maximum declination (deg, [scalar])
            kmin/kmax         - minimum/maximum apparent K-band magnitude (mag, [scalar])
            jminkmin/jminkmax - minimum/maximum J-K colour (mag, [scalar])
        
        Returns:
            Nothing
        """        
        
        # Parsec isochrones
        print("Undilling isochrone interpolants...")
        with open("stellarprop_parsecdefault_currentmass.dill", "rb") as input:
            self.pi = dill.load(input)
        print("...done.")
        print(" ")
        
        # Artificially make array for metallicity Z because not share with Parsec
        # isochrones
        isoZ = np.array([0.00010,0.00011,0.00013,0.00014,0.00016,0.00018,
                         0.00020,0.00022,0.00025,0.00028,0.00032,0.00036,
                         0.00040,0.00048,0.00050,0.00056,0.00064,0.00071,
                         0.00080,0.00089,0.00100,0.00112,0.00130,0.00141,
                         0.00160,0.00178,0.00200,0.00234,0.00250,0.00282,
                         0.00320,0.00355,0.00400,0.00447,0.00500,0.00562,
                         0.00640,0.00708,0.00800,0.00891,0.01000,0.01220,
                         0.01300,0.01410,0.01600,0.01780,0.02000,0.02240,
                         0.02500,0.02820,0.03200,0.03550,0.04000,0.04470,
                         0.05000,0.05620,0.06000])             
                  
        # Share variables with class
        self.ramin    = np.copy(ramin)
        self.ramax    = np.copy(ramax)
        self.decmin   = np.copy(decmin)
        self.decmax   = np.copy(decmax)
        self.kmin     = np.copy(kmin)
        self.kmax     = np.copy(kmax)
        self.jminkmin = np.copy(jminkmin)
        self.jminkmax = np.copy(jminkmax)
        self.isoage   = np.copy(self.pi.isoage) 
        self.isomh    = np.copy(self.pi.isomh)
        self.isoZ     = np.copy(isoZ)
        
    def  __call__(self,ra,dec,s,mh,age):
        
        return(self.sfinterp(ra,dec,s,mh,age))
    
    def checkInObsVol(self,ra,dec,kmag,jmink):
        
        """ CHECK WHETHER COORDINATES WITHIN OBSERVED VOLUME
        (THIS ACCOUNTS FOR THE FACT THAT THE NEAREST NEIGHBOUR INTERPOLANT WILL
        GIVE A NON-ZERO VALUE EVEN VERY FAR AWAY FROM INPUT COORDINATES)

        Arguments:
            ra    - right ascenscion (deg, [tensor])
            dec   - declination (deg, [tensor])
            kmag  - K-band magnitude (mag, [tensor])
            jmink - J-K colour (mag, [tensor])
    
        Returns:
            Boolean index [tensor]
        """     
        index = (sf.ramin<=ra)  & (ra<=sf.ramax) &\
                (sf.decmin<=dec) & (dec<=sf.decmax) &\
                (sf.kmin<=kmag) & (kmag<=sf.kmax) &\
                (sf.jminkmin<=jmink) & (jmink<=sf.jminkmax)
                     
        return(index)
        
    def checkRedClump(self,Z,mh,logg,teff,jmink):
        
        """ CHECK WHETHER RED CLUMP

        Arguments:
            Z     - metallicity (fraction, [tensor])
            mh    - log(metals/H vs. sun) (dex, [tensor])
            logg  - surface gravity (dex, [tensor])
            teff  - effective temperature (dex, [tensor])
            jmink - J-K colour (mag, [tensor])
    
        Returns:
            Boolean index [tensor]
        """     
        
        # Red clump selection (assume interchangeability between M/H and Z)
        teff_ref = -382.5*mh + 4607.
        index    = (logg>=1.8) & \
                   (logg<=(0.0018*(teff-teff_ref)+2.5)) &\
                   (Z <= 0.06) & \
                   (jmink >= 0.5) & \
                   (Z > 1.21*(jmink-0.05)**9 + 0.0011) & \
                   (Z < 2.58*(jmink-0.40)**3 + 0.0034)
                     
        return(index)
        
    def createObsSfInterp(self,obscoords,obssf):
        
        """ CREATE NEAREST NEIGHBOUR INTERPOLANT FOR `OBSERVED' SELECTION FUNCTION

        Arguments:
            obscoords - ra,dec,K-band magnitude,J-K colour (deg,deg,mag,mag,[matrix])
            obssf     - selection function probability (-, [vector])
    
        Returns:
           Nothing.
        """           
        # Create interpolant
        interp = NearestNDInterpolator(obscoords,obssf)
        
        # Copy data and interpolation object to class
        self.obssfinterp = interp
        
        return
 
    def createSkyposDistMhAgeInterp(self,ragrid,decgrid,smin,smax,ns,
                                    agemin,agemax,mhmin,mhmax,nskip):
                                        
        """ CREATE INTERPOLANT FOR 'POSITION-DISTANCE-METALLICITY-AGE' SELECTION FUNCTION

        Arguments:
            ragrid        - right ascencion (deg, [vector])
            decgrid       - declination (deg, [vector])
            smin/smax     - minimum/maximum distance (kpc, [scalar])
            ns            - number of distances (-, [integer scalar])
            agemin/agemax - minimum/maximum ages (Gyr, [vector])
            mhmin/mhmax   - minimum/maximum metallicities (dex, [vector])
            nskip         - what multiples of ages of isochrones to consider 
                            (useful for testing) (-, [integer scalar])
    
        Returns:
           Nothing.
        
        """
        
        # Copy variables locally
        isoage  = np.copy(sf.isoage)
        isomh   = np.copy(sf.isomh)
        isoZ    = np.copy(sf.isoZ)
        isodict = sf.pi.isodict
        
        # Sizes of sky position grids
        nra  = len(ragrid)
        ndec = len(decgrid)
        
        # Create distance grid
        sgrid = np.linspace(smin,smax,ns)
        
        # Construct metallicity and Z grids
        jmhmin = np.sum(isomh<mhmin)-1
        jmhmax = np.sum(isomh<mhmax)
        if (jmhmin < 0):
            jmhmin= 0
        if (jmhmax == len(isomh)):
            jmhmax = len(isomh)-1
        mhgrid  = isomh[jmhmin:jmhmax+1]
        Zgrid   = isoZ[jmhmin:jmhmax+1]
        nmh     = len(mhgrid)
        print("Considering "+str(nmh)+" metallicities...")
        
         # Construct age grid
        jagemin = np.sum(isoage<agemin)-1
        jagemax = np.sum(isoage<agemax)
        if (jagemin < 0):
            jagemin= 0
        if (jagemax == len(isoage)):
            jagemax = len(isoage)-1
        agegrid = isoage[jagemin:jagemax+1:nskip+1]
        nage    = len(agegrid)
        print("...and "+str(nage)+" ages...")
              
        # Create tensor of selection function probabilities
        sfprob_radec_s_mhage_tens = np.zeros([nra,ndec,ns,nmh,nage])
        
        # Start measuring time to calculate grid of selection function probabilities
        start_time = time.time()
               
        # Calculate selection function by iterating over isochrones, finding
        # the appropriate isochrone in the dictionary, and integrating the
        # `observed' SF over the IMF
        for jage in range(nage):
            for jmh in range(nmh):
                        
                # Find isochrone corresponding to age and metallicity
                isoname   = "age"+str(agegrid[jage])+"mh"+str(mhgrid[jmh])
                isochrone = isodict[isoname]  
                print("     Reading isochrone "+isoname)
                
                # Extract absolute magnitudes,logg,teff corresponding to 
                # different masses along the isochrones
                absJ  = isochrone[:,13]
                absKs = isochrone[:,15]
                logg  = isochrone[:,6]
                teff  = 10**(isochrone[:,5])
                    
                # Extract number of masses
                nmass = len(absJ)
                
                # Create tensors (nra*ndec*ns*nmass) of ra, dec, absolute magnitudes, apparent magnitudes, distances, apparent magnitudes, 
                absJtens  = np.array([np.array([np.array([absJ,]*ns),]*ndec),]*nra)
                absKstens = np.array([np.array([np.array([absKs,]*ns),]*ndec),]*nra)
                loggtens  = np.array([np.array([np.array([logg,]*ns),]*ndec),]*nra)
                tefftens  = np.array([np.array([np.array([teff,]*ns),]*ndec),]*nra)
                stens     = np.array([np.array([np.transpose(np.array([sgrid,]*nmass)),]*ndec),]*nra)
                ratens    = np.reshape(np.repeat(ragrid,ndec*ns*nmass), [nra,ndec,ns,nmass])
                dectens   = np.reshape(np.repeat(decgrid,ns*nmass), [ndec,ns,nmass])
                dectens   = np.array([dectens,]*nra)
                     
                # Calculate apparent magnitudes
                appJtens  = 5*np.log10(stens*1000./10.)+absJtens
                appKstens = 5*np.log10(stens*1000./10.)+absKstens
                JminKtens = appJtens-appKstens
                
                # Check tensor elements if needed
                if (False):
                    print("appJtens")
                    print(np.shape(appJtens))
                    print(appJtens[0,0,:,:])
                    print(appJtens[1,1,:,:])
                
                    print("appKstens")
                    print(np.shape(appKstens))
                    print(appKstens[0,0,:,:])
                    print(appKstens[1,1,:,:])
                
                    print("stens")
                    print(np.shape(stens))
                    print(stens[0,0,:,0])
                    print(stens[2,2,:,2])
                
                    print("ratens")
                    print(np.shape(ratens))
                    print(ratens[:,0,0,0])
                    print(ratens[:,3,3,3])
                
                    print("dectens")
                    print(np.shape(dectens))
                    print(dectens[0,:,0,0])
                    print(dectens[3,:,3,3])
                    
                # Check whether within observed volume  
                indexObsVol = sf.checkInObsVol(ratens,dectens,appJtens,JminKtens) 
                           
                # Check whether red clump
                Ztens         = np.zeros_like(ratens)+Zgrid[jmh]
                mhtens        = np.zeros_like(ratens)+mhgrid[jmh]
                indexRedClump = sf.checkRedClump(Ztens,mhtens,loggtens,tefftens,JminKtens)
                
                # Selection function for each ra, dec, distance, and mass given observed coordinates
                obscoords = np.stack((ratens,dectens,appJtens,JminKtens),axis=4)               
                sfmass    = sf.obssfinterp(obscoords)*indexObsVol*indexRedClump
                
                # Print terms to check
                if (False):
                    print(" ")
                    print(stens[0,0,0,:])
                    print(obscoords[0,0,0,:,:])
                    print(sfmass[0,0,0,:])
                    print(" ")
                    print(stens[10,10,10,:])
                    print(obscoords[10,10,10,:,:])
                    print(sfmass[10,10,10,:])
                                                
                # Integrate over IMF (last axis)
                yintegrand = sfmass*df.imf(isochrone[:,2])
                sfprob_radec_s_mhage_tens[:,:,:,jmh,jage] = np.trapz(yintegrand,isochrone[:,2],axis=3)

                # Test integration
                if (False):
                    print("Integration test 1:")
                    print(sfprob_radec_s_mhage_tens[0,0,0,jmh,jage])
                    integrand = np.zeros(nmass-1)
                    for jmass in range(nmass-1):
                        integrand[jmass] = (isochrone[jmass+1,2]-isochrone[jmass,2])*\
                            (sfmass[0,0,0,jmass]*df.imf(isochrone[jmass,2]) + sfmass[0,0,0,jmass+1]*df.imf(isochrone[jmass+1,2]))/2.
                    print(np.sum(integrand))
                    print("Integration test 2:")
                    print(sfprob_radec_s_mhage_tens[1,1,1,jmh,jage])
                    integrand = np.zeros(nmass-1)
                    for jmass in range(nmass-1):
                        integrand[jmass] = (isochrone[jmass+1,2]-isochrone[jmass,2])*\
                            (sfmass[1,1,1,jmass]*df.imf(isochrone[jmass,2]) + sfmass[1,1,1,jmass+1]*df.imf(isochrone[jmass+1,2]))/2.
                    print(np.sum(integrand))        
              
        # Print time required to calculate SF on grid of ra, dec, s, mh, age
        print("---- took %s seconds ----" % (time.time() - start_time))
                              
        # Make and share interpolant
        sfinterp = RegularGridInterpolator((ragrid,decgrid,sgrid,mhgrid,agegrid),
                                           sfprob_radec_s_mhage_tens,bounds_error=False,
                                           fill_value=0.)
        self.sfinterp = sfinterp
        
        return
        
#%% READ LAMOST DATA
# ID,ra,era,dec,edec,s,es,vr,evr,mra,emra,mdec,emdec,feh,efeh,afe,eafe,age,eage,
# fcmd,fieldid,color,mag,gl_cen,gb_cen
datafile  = "../data/lamost/LMRC-DR4-VF-SNR30.txt"
# Select ra,dec,s,vr,mra,mdec,feh,afe,age,fcmd,color,mag,gl_cen,gb_cen
data      = np.loadtxt(datafile,skiprows=1,usecols=((1,3,5,7,9,11,13,15,17,19,21,22,23,24)))
Obs       = data[:,0:9]
sf_l      = data[:,12]/180.*np.pi
sf_b      = data[:,13]/180.*np.pi
xg        = np.column_stack((sf_l,sf_b,sf_l*0.+1.))
xe        = ct.GalacticToEquatorial(xg)
obscoords = np.column_stack((xe[:,0]/np.pi*180.,xe[:,1]/np.pi*180.,data[:,11],data[:,10]))
obssf     = data[:,9]

#%% INSTANTIATE SELECTION FUNCTION CLASS
ragrid  = np.unique(np.round(obscoords[:,0],2))
nra     = len(ragrid)
dra     = ragrid[1]-ragrid[0]
ramin   = ragrid[0]-dra/2.
ramax   = ragrid[nra-1]+dra/2.                             

decgrid = np.unique(np.round(obscoords[:,1],2))
ndec    = len(decgrid)
ddec    = decgrid[1]-decgrid[0]
decmin  = decgrid[0]-ddec/2.
decmax  = decgrid[ndec-1]+ddec/2.

maggrid = np.unique(obscoords[:,2])
nmag    = len(maggrid)
dmag    = maggrid[1]-maggrid[0]
magmin  = maggrid[0]-dmag/2.
magmax  = maggrid[nmag-1]+dmag/2.

colgrid = np.unique(obscoords[:,3])
ncol    = len(colgrid)
dcol    = colgrid[1]-colgrid[0]
colmin  = colgrid[0]-dcol/2.
colmax  = colgrid[ncol-1]+dcol/2.

#%%
sf = LamostSelFunc(ramin,ramax,decmin,decmax,magmin,magmax,colmin,colmax)

#%% CREATE OBSERVED SELECTION FUNCTION INTERPOLANT AND PLOT
sf.createObsSfInterp(obscoords,obssf)
#%%
"""
nra_plot     = 5 # Number of right ascencions to plot
ndec_plot    = 5 # Number of declinations to plot
ncol_plot    = 10
nmag_plot    = 10
maggrid_plot = np.linspace(magmin,magmax,nmag_plot)
colgrid_plot = np.linspace(colmin,colmax,ncol_plot)

plt.rc('font',family='serif')
ra4d    = np.zeros([nra,ndec,nmag,ncol])
dec4d   = np.zeros([nra,ndec,nmag,ncol])
mag4d   = np.zeros([nra,ndec,nmag,ncol])
col4d   = np.zeros([nra,ndec,nmag,ncol])
sf4d    = np.zeros([nra,ndec,nmag,ncol]) 
for jra in range(nra):
    for jdec in range(ndec):
        for jmag in range(nmag):
            for jcol in range(ncol):
                ra4d[jra,jdec,jmag,jcol]  = ragrid[jra]
                dec4d[jra,jdec,jmag,jcol] = decgrid[jdec]
                mag4d[jra,jdec,jmag,jcol] = maggrid[jmag]
                col4d[jra,jdec,jmag,jcol] = colgrid[jcol]
                sf4d[jra,jdec,jmag,jcol]  = sf.obssfinterp((ragrid[jra],decgrid[jdec],
                                                            maggrid[jmag],colgrid[jcol]))
    
# Create 2D contour plots with matplotlib
fig,axarr   = plt.subplots(nra_plot,ndec_plot,figsize=(8,8),sharex=True,sharey=True)   
jra_plot  = 0
for jra in range(0,nra,nra/(nra_plot-1)):
    jdec_plot = 0     
    for jdec in range(0,ndec,ndec/(ndec_plot-1)):
        im = axarr[jra_plot,jdec_plot].contourf(mag4d[jra,jdec,:,:],
                                                col4d[jra,jdec,:,:],
                                                sf4d[jra,jdec,:,:],
                                                100,colormap='YlGnBu')
        #axarr[jra_plot,jdec_plot].set_xscale("log")
        #axarr[jra_plot,jdec_plot].set_yscale("log")
        if (jra_plot==nra_plot-1):
            axarr[jra_plot,jdec_plot].set_xlabel(r"$\delta$="+str(np.round(decgrid[jdec],1))+"$^o$",fontsize=10)
        if (jdec_plot==0):
            axarr[jra_plot,jdec_plot].set_ylabel(r"$\alpha$="+str(np.round(ragrid[jra],1))+"$^o$",fontsize=10)
        jdec_plot+=1
    jra_plot +=1
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.gcf().text(0.5,0.03,r"$K_s$ (mag)",fontsize=14)
plt.gcf().text(0.02,0.5,r"$J-K_s$ (mag)",fontsize=14,rotation=90)
fig.subplots_adjust(hspace=0,wspace=0)
cb = fig.add_axes([0.92, 0.125, 0.02, 0.75])
fig.colorbar(im,cax=cb)
plotfile = "../plots/lamost/LamostSelFunc_obs.eps"
fig.savefig(plotfile,format='eps')
"""
#%% CALCULATE SELECTION FUNCTION INTERPOLANTS
smin   = np.min(Obs[:,2])
smax   = np.max(Obs[:,2])  
ns     = 20
mhmin  = np.min(Obs[:,6])
mhmax  = np.max(Obs[:,6])
agemin = np.min(Obs[:,8])
agemax = 13.2#np.max(Obs[:,8])
nskip  = 1 # Need fine ages for red clump sample

#%% CALCULATE SELECTION FUNCTION AND DILL CLASS
sf.createSkyposDistMhAgeInterp(ragrid,decgrid,smin,smax,ns,
                               agemin,agemax,mhmin,mhmax,nskip)

#%%
print("Dill selection function...")
with open("../results/lamost/selfunc/selfunc_with_redclumpsel_highres.dill", "wb") as output:
    dill.dump(sf.sfinterp, output, dill.HIGHEST_PROTOCOL)  
print("...done.")         
print(" ")