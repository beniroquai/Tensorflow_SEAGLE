# -*- coding: utf-8 -*-
# author: Benedict Diederich 2018
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import scipy as scipy
import SEAGLE as seagle
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %load_ext autoreload
# %reload_ext autoreload



#%---------------------------------------------------------------------
#                  START CODE HERE                                    #
#%---------------------------------------------------------------------
       
# Define some parameters
is_debug = True # do you want to display everything?
learningrate = .1 
Niter = 40 # Optimization Steps

mysize = (100, 100, 100) # Z X Y
mymidpoint = int(mysize[1]/2)
mysample = np.zeros(mysize)

nObj = 1.4
nEmbb = 1.33
#nObj = 1.52 + 0.0 * 1j;
Boundary=9;

lambda0 = .5; # measured in µm
pixelsize = lambda0/4


# generate Sample, Assume nObj everywhere where mysample is 1, rest is background 
mysample = seagle.insertSphere((mysample.shape[0], mysample.shape[1], mysample.shape[2]), obj_dim=0.1, obj_type=0, diameter=1, dn=1) 

# define the source and insert it in the volume
kx = 0
ky = 0
myWidth = 20;
mySrc = seagle.insertSrc(mysize, myWidth, myOff=(Boundary+1, 0, 0), kx=kx,ky=ky);

#%-----------------------------------------------------------------------------
#                  SEAGLE PART COMES  HERE                                    #
#%-----------------------------------------------------------------------------

# Instantiate the SEAGLE
MySEAGLE = seagle.SEAGLE(mysample, mySrc, lambda0=lambda0, pixelsize=pixelsize, nObj = nObj, nEmbb=nEmbb, Boundary=Boundary)

if(is_debug):
    # displaying the Src and Obj just for debugging purposes
    plt.imshow(np.squeeze(np.real(MySEAGLE.f[:,:,np.int(np.floor(mysize[2]/2))]))), plt.colorbar(), plt.show()
    plt.imshow(np.squeeze(np.real(MySEAGLE.mySrc[:,:,np.int(np.floor(mysize[2]/2))]))), plt.colorbar(), plt.show()
    
    scipy.io.savemat('myobj.mat', mdict={'myobj': MySEAGLE.f})
    scipy.io.savemat('myrsc.mat', mdict={'mySrc': MySEAGLE.mySrc})



# Compute the model inside the convergent born series 
MySEAGLE.computeModel()

if(is_debug):
    # Visualize Greenzs function
    plt.imshow(np.abs(MySEAGLE.greens_fkt[:,:,mymidpoint])), plt.colorbar(), plt.show()
    plt.imshow(np.abs(MySEAGLE.greens_fkt_ft[:,:,mymidpoint])), plt.colorbar(), plt.show()


# Define Minimization step
MySEAGLE.minimize(learningrate)

# Initialize all operands
MySEAGLE.compileGraph()


#%% Do n iterations to let the series converge
print('Start Computing the result')
for i in range(Niter):

    start_time = time.time()
    _, myerror = MySEAGLE.sess.run([MySEAGLE.train_op, MySEAGLE.my_error])
    print('Step '+str(i) + ' took ' +str(0*(time.time()-start_time))+' s'+ ' and the error: '+str(myerror))
    if(np.mod(i,10)==0 and is_debug):
        #%% Display result 
        plt.subplot(1,2,1), plt.title('Magnitude of PSI X/Z Plot')
        plt.imshow(np.squeeze(np.abs(MySEAGLE.tf_u.eval()[:,mymidpoint,:])), cmap='gray'), plt.colorbar()
        plt.subplot(1,2,2), plt.title('Phase of PSI X/Z Plot')
        plt.imshow(np.squeeze(np.angle(MySEAGLE.tf_u.eval()[:,mymidpoint,:])), cmap='gray'), plt.colorbar()
        plt.show()



#%% Display result 
plt.subplot(1,2,1), plt.title('Magnitude of PSI')
plt.imshow(np.squeeze(np.abs(MySEAGLE.tf_u.eval()[:,:,mymidpoint ])), cmap='gray'), plt.colorbar()
plt.subplot(1,2,2), plt.title('Phase of PSI')
plt.imshow(np.squeeze(np.angle(MySEAGLE.tf_u.eval()[:,:,mymidpoint ])), cmap='gray'), plt.colorbar()
plt.show()

# Display result 
plt.subplot(1,2,1), plt.title('Magnitude of PSI')
plt.imshow(np.squeeze(np.abs(MySEAGLE.tf_u.eval()[mymidpoint,:,:])), cmap='gray'), plt.colorbar()
plt.subplot(1,2,2), plt.title('Phase of PSI')
plt.imshow(np.squeeze(np.angle(MySEAGLE.tf_u.eval()[mymidpoint,:,:])), cmap='gray'), plt.colorbar()
plt.show()
print('SERIOUS PROBLEM!!!! Why is the FT of Green so weird?')


scipy.io.savemat('exportresult.mat', mdict={'u_dash': MySEAGLE.tf_u.eval(), 
                                            'f_obj':MySEAGLE.f, 
                                            'mysrc': MySEAGLE.mySrc, 
                                            'u_in': MySEAGLE.u_in, 
                                            'greens_fkt': MySEAGLE.greens_fkt, 
                                            'greens_fkt_ft': MySEAGLE.greens_fkt_ft})

# save images
myamp_res = np.squeeze(np.abs(MySEAGLE.tf_u.eval()[mymidpoint,:,:]))
myamp_res  = np.uint8(myamp_res /np.max(myamp_res )*(2**8-1))
plt.imsave('result_abs_xy.png', myamp_res)

myphase_res = np.squeeze(np.angle(MySEAGLE.tf_u.eval()[mymidpoint,:,:]))
myphase_res = np.uint8(myphase_res/np.max(myphase_res)*(2**8-1))
plt.imsave('result_angle_xy.png', myphase_res)


myamp_res = np.squeeze(np.abs(MySEAGLE.tf_u.eval()[:,mymidpoint,:]))
myamp_res  = np.uint8(myamp_res /np.max(myamp_res )*(2**8-1))
plt.imsave('result_abs_xz.png', myamp_res)

myphase_res = np.squeeze(np.angle(MySEAGLE.tf_u.eval()[:,mymidpoint,:]))
myphase_res = np.uint8(myphase_res/np.max(myphase_res)*(2**8-1))
plt.imsave('result_angle_xz.png', myphase_res)