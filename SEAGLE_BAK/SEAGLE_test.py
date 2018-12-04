# -*- coding: utf-8 -*-

# performs a 2D simulation, assuming cylinder symmertry along Z

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


def save(filename, numpyarray):
    import scipy.io
    data = {}
    data['numpyarray'] = numpyarray
    scipy.io.savemat(filename, mdict=data)


# CUDA_DEVICE_VISIBLE=all my gpus

is_debug = True


mysize = (100, 100, 100) # X Y Z
mymidpoint = int(mysize[1]/2)
mysample = np.zeros(mysize)

SlabX = 260
SlabW = 10;
myN_obj = 1.51
myN_embb = 1.34
#myN_obj = 1.52 + 0.0 * 1j;
Boundary=0;

lambda0 = .630
pixelsize = lambda0/4

k0 = 2*pi/lambda0
k0 = .25/ abs(myN_obj); # 2pi/lambda #TODO: Sampling is totally wrong!

if(False):
    # generate Sample
    mysample = .3+seagle.insertSphere((mysample.shape[0], mysample.shape[1], mysample.shape[2]), obj_dim=0.1, obj_type=0, diameter=1, dn=myN_obj)
    mysample = mysample + 0j
    plt.imshow(np.squeeze(np.real(mysample[:,np.int(np.floor(mysize[2]/2)),:])), cmap='gray'), plt.colorbar(), plt.show()
    plt.imshow(np.squeeze(np.imag(mysample[:,np.int(np.floor(mysize[2]/2)),:])), cmap='gray'), plt.colorbar(), plt.show()
    
    #mysample, _ = seagle.insertPerfectAbsorber(mysample, 0, Boundary, -3, k0);
    #mysample, _ = seagle.insertPerfectAbsorber(mysample, mysample.shape[2] - Boundary, mysample.shape[2], 3, k0);
    plt.imshow(np.squeeze(np.real(mysample[:,np.int(np.floor(mysize[2]/2)),:])), cmap='gray'), plt.colorbar(), plt.show()
    plt.imshow(np.squeeze(np.imag(mysample[:,np.int(np.floor(mysize[2]/2)),:])), cmap='gray'), plt.colorbar(), plt.show()
        
    print("ATTENTION: Wrong Source!!")
    mySrc = np.zeros(mysample.shape)
    mygauss = seagle.rr_freq(mysample.shape[0], mysample.shape[1], 0)
    mygauss= np.exp(-mygauss**2/.1)
    mySrc[:,:,0] = mygauss

else:
    # generate Sample
    mysample = seagle.insertSphere((mysample.shape[0], mysample.shape[1], mysample.shape[2]), obj_dim=0.1, obj_type=0, diameter=1, dn=1) 

    # We need to insert the perfect-aborber boundeary condition allong to prevent reflection at th edges
    if(Boundary>0):
        mysample, _ = seagle.insertPerfectAbsorber(mysample, 0, Boundary, -1, k0);
        mysample, _ = seagle.insertPerfectAbsorber(mysample, mysample.shape[0] - Boundary, Boundary, 1, k0);

    # define the source and insert it in the volume
    kx = 0
    ky = 0
    myWidth = 30;
    mySrc = seagle.insertSrc(mysize, myWidth, myOff=(Boundary+1, 0, 0), kx=kx,ky=ky);

    # displaying the Src and Obj just for debugging purposes
    plt.imshow(np.squeeze(np.real(mysample[:,:,np.int(np.floor(mysize[2]/2))]))), plt.colorbar(), plt.show()
    plt.imshow(np.squeeze(np.real(mySrc[:,:,np.int(np.floor(mysize[2]/2))]))), plt.colorbar(), plt.show()

    scipy.io.savemat('myobj.mat', mdict={'myobj': mysample})
    scipy.io.savemat('myrsc.mat', mdict={'mySrc': mySrc})




# Instantiate the SEAGLE
MySEAGLE = seagle.SEAGLE(mysample, mySrc, k0=k0)

# Compute the model inside the convergent born series 
MySEAGLE.computeModel()

# Visualize Greenzs function
plt.imshow(np.abs(MySEAGLE.greens_fkt[:,:,mymidpoint])), plt.colorbar(), plt.show()
plt.imshow(np.abs(MySEAGLE.greens_fkt_ft[:,:,mymidpoint])), plt.colorbar(), plt.show()


# Define Minimization step
MySEAGLE.minimize(1)

# Initialize all operands
MySEAGLE.compileGraph()


#%% Do n iterations to let the series converge
print('Start Computing the result')
for i in range(100):

    start_time = time.time()
    _, myerror = MySEAGLE.sess.run([MySEAGLE.train_op, MySEAGLE.my_error])
    print('Step '+str(i) + ' took ' +str(0*(time.time()-start_time))+' s'+ ' and the error: '+str(myerror))
    if(np.mod(i,10)==0):
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


scipy.io.savemat('exportresult.mat', mdict={'u_dash': MySEAGLE.tf_u.eval(), 
                                            'f_obj':MySEAGLE.mysample, 
                                            'mysrc': MySEAGLE.mySrc, 
                                            'u_in': MySEAGLE.u_in, 
                                            'greens_fkt': MySEAGLE.greens_fkt, 
                                            'greens_fkt_ft': MySEAGLE.greens_fkt_ft})

