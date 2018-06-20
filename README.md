# MCRICM
Code for reconstruction of MCRICM images of a lamellipodium.

1) You first need to run simulations of the reflectivity for the various conditions (INA, wavelength, d, h, nin)
To do so, edit Main_simu.py to modify the parameters of the calculation and launch it with "python Main_simu.py"
It will create a folder with all the results stored in python .npy files, which will be used for the reconstruction.

2) Edit the file Main_reconstruction.py to change the parameters as you wish. A folder "Results" will be created containing the reconstruction (d, h, nin) and the associated chi2. 
It will also create a folder '/ForEachIndex' which contains the reconstruction for fixed refractive indices of the cytoplasm nin.

Two MCRICM stack of images are provided for example, and can be reconstructed using the initial parameters. The first small image can be used to test the program, and should only take a minute to be reconstructed. Another larger image of a lamellipodium is provided, and should be used with several processors in parallel.

Each program contains extensive comments to explain the reconstruction process step by step.
