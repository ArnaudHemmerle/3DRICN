import numpy as np
import os
import multiprocessing
from PIL import Image
from skimage import io

pi = np.pi

"""
Run this program to perform the reconstruction of a MCRICM stack of images.
The stack should contain the different images grouped by color and increasing INA.
The simulations should previously done with Main_simu.py.
"""

###################################################
##################PARAMETERS#######################
###################################################

#The name of the MCRICM stack
filename = 'MCRICM_image_small.tif'

#List of refractive indices for the cytosplasm in green
#Decimal part only for simplicity (i.e. 340 -> 1.340)
ninG = [340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400]

#List of INA
#Decimal part only for simplicity (i.e. 45 -> 0.45)
INA = [50, 56, 68, 86, 105]

#Folder containing the simulations
SimuFolder = '../Simulation/Simu'

#Number of processors used
#If multiprocessing does not work, try NbProc = 1
NbProc = 4

#Define min, max, step of d and h in nm
#Use the same parameters as your simulations for simplicity
dmin = 0; dmax = 1000; dstep = 10 #d range
hmin = 0; hmax = 500; hstep = 2 #h range  

#Number of different illuminations
NbIllum = 3*len(INA)

#Rescaled h and d for calculations
imax_d = int((dmax-dmin)/dstep)
jmax_h = int((hmax-hmin)/hstep)

    

def worker(rri_sub, q, out_q):
    
    """
    Function called for multiprocessing.
    Reconstruct the image stripe by stripe.
    """

    outdict = {}
    
    
    hh_dict_sub, dd_dict_sub, chi2_dict_sub = LoopNin(rri_sub)
    
    
    hh_sub, dd_sub, chi2_sub, nin_sub \
    = BestSol(hh_dict_sub, dd_dict_sub, chi2_dict_sub)
    
    
    outdict = [hh_sub, dd_sub, chi2_sub, nin_sub, hh_dict_sub, dd_dict_sub, chi2_dict_sub]
    
    out_q.put((q, outdict))

    

def Initialize(ninstring):
    """
    Create the maps of reflectivity vs conditions (i.e. illuminations),
    through selective extraction of the simulations previously done.
    """

    nmax = 3*len(INA)
    
    color_list = len(INA)*['B']+len(INA)*['G']+len(INA)*['R']
    
    I_vs_cd_Map = np.ndarray( shape = (imax_d, jmax_h, nmax)) 
    

    for n in range(nmax):
        
        folder = SimuFolder + '/INA' + str(INA[n%len(INA)]) + '_nin' + str(ninstring) 
        I_vs_cd_Map[:,:,n] = np.load(folder+'/SimuMap'+color_list[n]+'.npy')
        
    return I_vs_cd_Map

    

def LoopNin(rri_dict):
    """
    Reconstruction for each nin independently (i.e. fixes nin, find best h and d).
    """
    
    nmax = len(ninG)
    
    #Each elem at position n contains the reconstruction for the index ninG[n]
    #For h
    hh_dict = {}
    
    #For d
    dd_dict = {}
    
    #Corresponding chi2
    chi2_dict = {}
    
    for n in range(nmax):
        
        I_vs_cd_Map = Initialize(ninG[n])
        
        ddtemp, hhtemp, chi2temp\
        = BestSol_FixNin(I_vs_cd_Map, rri_dict, ninG[n])
    
        #Rescale the results
        dd_dict[n] = ddtemp*dstep
        hh_dict[n] = hhtemp*hstep
        chi2_dict[n] = chi2temp
        
    return hh_dict, dd_dict, chi2_dict
    


def BestSol(hh_dict, dd_dict, chi2_dict):
    """
    Find best nin within the dictionnaries of best solutions for each nin,
    by minimizing chi2 pixel by pixel.
    """
    
    nmax = len(hh_dict)
    
    imax = np.shape(hh_dict[0])[0]
    jmax = np.shape(hh_dict[0])[1]
    
    
    chi2_Map, hh_Map, dd_Map = (np.ndarray( shape = (imax, jmax, nmax)) for i in range(3))
    nin_loc = [ninG[i%len(ninG)] for i in range(nmax)]
    
    for n in range(nmax):
        chi2_Map[:,:,n] = chi2_dict[n]
        hh_Map[:,:,n] = hh_dict[n]
        dd_Map[:,:,n] = dd_dict[n]
        

    
    nin = np.zeros((imax, jmax))
    hh, dd, chi2 = (np.ndarray( shape = (imax, jmax)) for i in range(3))
 
    for i in range(0, imax):
        
        for j in range(0, jmax):
            
            V_minloc = chi2_Map[i,j,:].argmin()
            nin[i,j] = 1+0.001*nin_loc[V_minloc]
            hh[i,j] = hh_Map[i,j,V_minloc]
            dd[i,j] = dd_Map[i,j,V_minloc]
            chi2[i,j] = chi2_Map[i,j,V_minloc]
           
    return hh, dd, chi2, nin




def BestSol_FixNin(I_vs_cd_Map, rri_dict, nin_n):
    """
    Find the best solution, pixel by pixel, for a fix nin.
    """
    
    imax = np.shape(rri_dict[0])[0]
    jmax = np.shape(rri_dict[0])[1]
    
    nmax = 3*len(INA)
    
    dd_fixnin, hh_fixnin, chi2_fixnin = (np.zeros((imax, jmax)) for i in range(3))
    
    #Store the MCRICM images in one array
    rr = [np.array(rri_dict[n]) for n in range(nmax)]
    
    #Reconstruction pixel by pixel
    for i in range(imax):
        
        for j in range(jmax):
            
            I_vs_cd = [rr[n][i,j] for n in range(nmax)]

            ImapDiff_illum = np.divide(np.square(I_vs_cd_Map-I_vs_cd),I_vs_cd_Map)
            
            #ImapDiff is the map of Chi2 for all conditions. The min of ImapDiff gives the best solution.
            ImapDiff = np.sum(ImapDiff_illum, 2)
            ImapDiff = np.round(ImapDiff, 8)

            #Best solution for the current nin
            V_minloc = np.unravel_index(ImapDiff.argmin(), ImapDiff.shape)
            V_minRowLoc = V_minloc[0]
            V_minColLoc = V_minloc[1]  
            chi2_fixnin[i,j] = ImapDiff[V_minRowLoc,V_minColLoc]
            dd_fixnin[i,j] = V_minRowLoc
            hh_fixnin[i,j] = V_minColLoc
           
           
        #Allow to track the reconstruction
        if i==0:
             print("nin", nin_n,  "; step",  i, "/", imax)        
 
    return dd_fixnin, hh_fixnin, chi2_fixnin
           


def save_stack(dict_stack, name_stack):
    try:
        os.makedirs('Results/ForEachIndex/'+name_stack)
    except OSError:
        if not os.path.isdir('Results/ForEachIndex/'+name_stack):
            raise

    for m in dict_stack:
        img = Image.fromarray(np.transpose(dict_stack[m]))
        img.save('Results/ForEachIndex/'+name_stack+'/'+name_stack+'_nin'+str(ninG[m])+'.tif')


def save_img(img, name_img):
    try:
        os.makedirs('Results/')
    except OSError:
        if not os.path.isdir('Results/'):
            raise

    img = Image.fromarray(np.transpose(img))
    img.save('Results/'+name_img+'.tif')
    


if __name__ == '__main__':


    #import file
    file0 = io.imread(filename)
    stack = file0.astype(np.float32)
    dim_norm_image = np.shape(stack[0,:,:])
    
    #rri_dict contains the MCRICM images
    rri_dict = {}   

    for illum in range(NbIllum):
        rri_dict[illum] = np.transpose(stack[illum, :, :])
        
    #Dimensions of the images
    hdim = np.shape(rri_dict[0])[0]
    vdim = np.shape(rri_dict[0])[1]
    

    if NbProc == 1:

        
        #############################################################
        ##FOR SINGLE PROCESSOR USE
        #############################################################
        
        hh_dict, dd_dict, chi2_dict = LoopNin(rri_dict)
        
        hh, dd, chi2, nin = BestSol(hh_dict, dd_dict, chi2_dict)
        
        
    else:
         
        #############################################################
        ##FOR MULTI PROCESSOR USE
        #############################################################
        
        #We will divide the full images into subparts along the x axis
        #The number of parts will be the number of proc used
        
        dimx_full = dim_norm_image[1]
        #The list containing the dimensions of the subparts
        dimx_list = np.linspace(0, dimx_full+1, num=NbProc+1, endpoint=True, dtype = 'uint32')
        print("list of dimx", dimx_list)
        
     
        #Divide the full image into a sub-image
        #Here rri_sub[q][n][[x0:D1,y0:H1] with q the part number, n the Illumination
        rri_sub = {}
        for q in range(NbProc):
            rri_sub_q= {}
            for n in range(NbIllum):
                rri_sub_q[n] = rri_dict[n][dimx_list[q]:dimx_list[q+1],:]
            rri_sub[q] = rri_sub_q
        
        
        #Calls the function in charge of doing the calculations on multiple proc
        out_q = multiprocessing.Queue()
        procs = []
        
        for q in range(NbProc):
            p = multiprocessing.Process(
                    target = worker,
                    args = ( rri_sub[q], q,
                            out_q))
            procs.append(p)
            p.start()
            
        
        result = [out_q.get() for i in range(NbProc)]
        
            
        out_q.close()
        out_q.join_thread()
        
        for p in procs:
            p.join()
                
        result.sort()
        result = [r[1] for r in result]
       
        ########################################################
        #Concatenate all the different parts into the final maps
        ########################################################
        #Here resultdict[r][i][j][:] means
        #part i of the image (subdivised for parallel processing)
        #j is the position of the parameter in the Result array
        hh = np.concatenate([result[i][0] for i in range(NbProc)], axis = 0)
        dd = np.concatenate([result[i][1] for i in range(NbProc)], axis = 0)
        chi2 = np.concatenate([result[i][2] for i in range(NbProc)], axis = 0)
        nin = np.concatenate([result[i][3] for i in range(NbProc)], axis = 0)
        
        
        #Here Result[i][j][k][:] means
        #part i of the image (subdivised for parallel processing)
        #j is the position of the parameter in the Result array
        #k is the position of the nin in the nin array
        chi2_dict = {}
        dd_dict = {}
        hh_dict = {}
        for k in range(len(ninG)):
            hh_dict[k] = np.concatenate([result[i][4][k] for i in range(NbProc)], axis = 0)
            dd_dict[k] = np.concatenate([result[i][5][k] for i in range(NbProc)], axis = 0)
            chi2_dict[k] = np.concatenate([result[i][6][k] for i in range(NbProc)], axis = 0)
            
           
      
    save_stack(chi2_dict, 'CHI2')
    save_stack(dd_dict, 'D')
    save_stack(hh_dict, 'H')
    
    save_img(hh, 'h')
    save_img(dd, 'd')
    save_img(nin, 'nin')
    save_img(chi2, 'chi2')

    
