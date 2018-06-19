import numpy as np
import os

pi = np.pi

###################################################
"""
Run this program before doing the reconstruction.
It simulates the reflectivity for all the combinations
of h, d, nin, wavelength and ina, within specific
intervals.
"""

###################################################
##################PARAMETERS#######################
###################################################

#List of refractive indices for the cytosplasm in green
#Decimal part only for simplicity (i.e. 340 -> 1.340)
ninG = [340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400]


#List of INA
#Decimal part only for simplicity (i.e. 45 -> 0.45)
INA = [50, 56, 68, 86, 105]


#Membrane thickness in nm
dm = 4

#Refractive indices in green
#Glass
nglassG = 1.525
#Lipids
nlipG = 1.486
#PBS
noutG = 1.335

#List of wavelengths in nm
lambdaB=436
lambdaG=546
lambdaR=625

#Define min, max, step of d and h in nm
dmin = 0; dmax = 1000; dstep = 10 #d range
hmin = 0; hmax = 500; hstep = 2 #h range  


###################################################
############REFLECTIVITY MODEL#####################
###################################################

def RefPol(llambda, ina, height, thickness, noutG, ninG):
    """
    Reflectivity model for a lamellipodium taking into account
    lambda, ina and polarization.
    Medium 1: Glass (semi-infinite)
    Medium 2: PBS
    Medium 3: Lipid membrane
    Medium 4: Cytoplasm
    Medium 5: Lipid membrane
    Medium 6: PBS (semi-infinite)

    This function can be easily generalized to a different amount of layers.

    #RefPol(546, 1.25, 5, 100, 1.333, 1.36)
    """

    #Parameters for integration
    tmax = np.arcsin(ina/nglassG) #max angle
    nt = 200 #number of angles
    dt = tmax/nt #angle increment
    
    
    #Taking into account the dispersion of light
    #Following Cauchy's law
    cauchy = 4200
    
    nin = ninG - cauchy*(1/lambdaG**2-1/llambda**2)   
    nout = noutG - cauchy*(1/lambdaG**2-1/llambda**2)   
    nglass = nglassG - cauchy*(1/lambdaG**2-1/llambda**2)   
    nlip = nlipG - cauchy*(1/lambdaG**2-1/llambda**2)

    #List of refractive indices
    n = np.array([nglass, nout, nlip, nin, nlip, nout])
    
    #List of thicknesses
    d = np.array([0, height, dm, thickness, dm, 0]) 

    
    m = len(n)-2

    #Initialization
    wt = np.array( [x*dt for x in range(nt)] )
    
    Rp, Rs, Rsp, Rpbg, Rsbg, Rspbg, integrand, integrandbg = \
    (np.ndarray(shape = (nt), dtype = 'cfloat') for i in range(8))
    
    b, theta, ns, nnp, nsp, m11, m12, m21, m22, mtot11, mtot12, mtot21, mtot22 = \
    (np.ndarray(shape = (m+2, nt), dtype = 'cfloat') for i in range(13))
   
    
    #Calculation
    theta[0,:] = wt
    

    for j in range(1,m+2):
        theta[j,:]=np.arcsin(n[j-1]/n[j]*np.sin(theta[j-1,:]))

    for p in range(m+2):
        b[p,:] = 2*pi*d[p]*n[p]/llambda*np.cos(theta[p,:])
        ns[p,:] = n[p]*np.cos(theta[p,:])
        nnp[p,:] = n[p]/np.cos(theta[p,:])
    
    for i in (0,1):
        if (i==0):
            nsp = ns
        else:
            nsp = nnp
            
        
        m11[:,:] = np.cos(b[:,:])
        m22[:,:] = np.cos(b[:,:])
        m12[:,:] = 1j*np.sin(b[:,:])/nsp[:,:]
        m21[:,:] = np.sin(b[:,:])*nsp[:,:]*1j
        mtot11[1,:] = m11[1,:]; mtot22[1,:] = m22[1,:]
        mtot12[1,:] = m12[1,:]; mtot21[1,:] = m21[1,:]
        
        
        for j in range(2,m+1):
            mtot11[j,:]= mtot11[j-1,:]*m11[j,:] + mtot12[j-1,:]*m21[j,:]
            mtot22[j,:]= mtot21[j-1,:]*m12[j,:] + mtot22[j-1,:]*m22[j,:] 
            mtot21[j,:]= mtot21[j-1,:]*m11[j,:] + mtot22[j-1,:]*m21[j,:] 
            mtot12[j,:]= mtot11[j-1,:]*m12[j,:] + mtot12[j-1,:]*m22[j,:] 
        
        Rsp[:] = mtot11[m,:]*nsp[0,:] - mtot22[m,:]*nsp[m+1,:]            \
                + mtot12[m,:]*nsp[0,:]*nsp[m+1,:] - mtot21[m,:]        

        
        Rsp[:]=Rsp[:]/( mtot11[m,:]*nsp[0,:] + mtot22[m,:]*nsp[m+1,:]     \
              + mtot12[m,:]*nsp[0,:]*nsp[m+1,:] + mtot21[m,:] )
        
        Rspbg[:]= (nsp[0,:] - nsp[1,:])/(nsp[0,:] + nsp[1,:])
        
        if (i==0): 
            Rs[:] = Rsp[:]; Rsbg[:] = Rspbg[:]
            
        else:
            Rp[:] = Rsp[:]; Rpbg[:] = Rspbg[:]

    
    integrandbg[:] = np.real((Rsbg[:]-Rpbg[:])*np.conj(Rsbg[:]-Rpbg[:])) * np.sin(theta[0,:]) * dt
    
    intbg = np.sum(integrandbg)/(1-np.cos(tmax))
    
    integrand[:] = np.real((Rs[:]-Rp[:]) * np.conj(Rs[:]-Rp[:])) * np.sin(theta[0,:]) * dt
    iint = np.sum(integrand)/(1-np.cos(tmax))
    
    
    return np.real(iint/intbg)
    


def LoopRefPol(llambda, ina, noutG, ninG, thickness):
    """
    Loop on RefPol simulations to vary h
    #LoopRefPol(546, 1.25, 1.333, 1.36, 5)
    """

    #Definition
    jmax = int((hmax-hmin)/hstep)
    
    #Initialization
    wh = np.array([0]*jmax)
    wr = 0.*wh
    
    #Calculation
    for j in range(jmax):
        wh[j] = hmin+j*hstep
        wr[j] = RefPol(llambda, ina, wh[j], thickness, noutG, ninG)
    tmax = np.arcsin(ina/nglassG)
    
    return wr

def LoopLoopRefPol(llambda, ina, noutG, ninG):
    """
    Loop on LoopRefPol simulations to vary d
    #LoopLoopRefPol(546, 1.25, 1.333, 1.36)
    """
    #Definition
    imax = int((dmax-dmin)/dstep)

    
    #Initialization
    Reflec = np.ndarray(shape = (imax, int((hmax-hmin)/hstep)))
    wd = np.array([0 for x in range(imax)])
    
    
    #Calculation
    
    #Loop on d
    for i in range(imax):
        wd[i] = dmin + i*dstep
        wr = LoopRefPol(llambda, ina, noutG, ninG, wd[i])   
        Reflec[i,:] = wr
        
        
    return Reflec
    

def LoopLoopLambdaRefPol(ina, nout, nin):
    """
    Loop on LoopLoopRefPol for variable lambda
    #LoopLoopLambdaRefPol(1.25, 1.333, 1.36)
    """
    
    SimuMapB = LoopLoopRefPol(lambdaB, ina, nout, nin)
    SimuMapG = LoopLoopRefPol(lambdaG, ina, nout, nin)
    SimuMapR = LoopLoopRefPol(lambdaR, ina, nout, nin)
   
    return SimuMapB, SimuMapG, SimuMapR

if __name__ == "__main__":
    
    #Loop on LoopLoopLambdaRefPol to vary nin
        
    imax=len(ninG)  
    jmax=len(INA)
    
    for i in range(imax):
        for j in range(jmax):
            
            SimuMapB, SimuMapG, SimuMapR = \
            LoopLoopLambdaRefPol(INA[j]*0.01, noutG, 1+0.001*ninG[i])
            folder = 'Simu/INA' + str(INA[j]) + '_nin' + str(ninG[i])+'/' 
            
            print(folder)
            
            #Check if the folder needs to be created
            try:
                os.makedirs(folder)
            except OSError:
                if not os.path.isdir(folder):
                    raise
            
            np.save(folder+'SimuMapB.npy',SimuMapB)
            np.save(folder+'SimuMapG.npy',SimuMapG)
            np.save(folder+'SimuMapR.npy',SimuMapR)



