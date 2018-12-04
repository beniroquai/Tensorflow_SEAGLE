# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.client import timeline

import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import scipy as scipy

# Some helpful MATLAB functions
def abssqr(inputar):
    return np.real(inputar*np.conj(inputar))
    #return tf.abs(inp#utar)**2

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
def tf_abssqr(inputar):
    return tf.real(inputar*tf.conj(inputar))
    #return tf.abs(inputar)**2

def rr(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.linspace(-inputsize_x/2,inputsize_x/2, inputsize_x)
    y = np.linspace(-inputsize_y/2,inputsize_y/2, inputsize_y)

    if inputsize_z<=1:
        xx, yy = np.meshgrid(x+x_center, y+y_center)
        r = np.sqrt(xx**2+yy**2)
        r = np.transpose(r, [1, 0]) #???? why that?!
    else:
        z = np.linspace(-inputsize_z/2,inputsize_z/2, inputsize_z)
        xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.sqrt(xx**2+yy**2+zz**2)
        r = np.transpose(r, [1, 0, 2]) #???? why that?!
        
    return np.squeeze(r)

def rr_freq(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.linspace(-inputsize_x/2,inputsize_x/2, inputsize_x)/inputsize_x
    y = np.linspace(-inputsize_y/2,inputsize_y/2, inputsize_y)/inputsize_y
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x+x_center, y+y_center)
        r = np.sqrt(xx**2+yy**2)
        r = np.transpose(r, [1, 0]) #???? why that?!
    else:
        z = np.linspace(-inputsize_z/2,inputsize_z/2, inputsize_z)/inputsize_z
        xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.sqrt(xx**2+yy**2+zz**2)
        r = np.transpose(r, [1, 0, 2]) #???? why that?!

        
    return np.squeeze(r)


def xx(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        xx = np.transpose(xx, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        xx = np.transpose(xx, [1, 0, 2]) #???? why that?!
    return np.squeeze(xx)

def yy(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        yy = np.transpose(yy, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        yy = np.transpose(yy, [1, 0, 2]) #???? why that?!
    return np.squeeze(yy)

def zz(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    nx = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    ny = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    nz = np.arange(-inputsize_z/2,inputsize_z/2, 1)
    xxr, yyr, zzr = np.meshgrid(nx, ny, nz)
    zzr = np.transpose(zzr, [1, 0, 2]) #???? why that?!
    return (zzr)


# %% FT

# I would recommend to use this
def my_ft2d(tensor):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift2d(tf.fft2d(ifftshift2d(tensor)))

def my_ift2d(tensor):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of ifft unlike dip_image.
    """
    return fftshift2d(tf.ifft2d(ifftshift2d(tensor)))


# fftshifts
def fftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1  # from 0 to shape-1
    top, bottom = tf.split(tensor, 2, last_dim)  # split into two along last axis
    tensor = tf.concat([bottom, top], last_dim)  # concatenates along last axis
    left, right = tf.split(tensor, 2, last_dim - 1)  # split into two along second last axis
    tensor = tf.concat([right, left], last_dim - 1)  # concatenate along second last axis
    return tensor

def ifftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1
    left, right = tf.split(tensor, 2, last_dim - 1)
    tensor = tf.concat([right, left], last_dim - 1)
    top, bottom = tf.split(tensor, 2, last_dim)
    tensor = tf.concat([bottom, top], last_dim)
    return tensor

def fftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor

def ifftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor    


def my_ft3d(tensor):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift3d(tf.fft3d(ifftshift3d(tensor)))

def my_ift3d(tensor):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift3d(tf.ifft3d(ifftshift3d(tensor)))


def my_ft3d_np(numpy):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(numpy)))

def my_ift3d_np(numpy):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(numpy)))


def insertSphere(obj_shape = [100, 100, 100], obj_dim = 0.1, obj_type = 0, diameter = 1, dn = 0.1):
    ''' Function to generate a 3D RI distribution to give an artificial sample for testing the FWD system
    INPUTS:
        obj_shape - Nx, Ny, Nz e.g. [100, 100, 100]
        obj_dim - sampling in dx/dy/dz 0.1 (isotropic for now)
        obj_type - 0; One Sphere 
                 - 1; Two Spheres 
                 - 2; Two Spheres 
                 - 3; Two Spheres inside volume
                 - 4; Shepp Logan Phantom (precomputed in MATLAB 120x120x120 dim)
        diameter - 1 (parameter for diameter of sphere)
        dn - difference of RI e.g. 0.1
        
    OUTPUTS: 
        f - 3D RI distribution
            
    '''
    # one spherical object inside a volume
    f = (dn)*(rr(obj_shape[0], obj_shape[1], obj_shape[2])*obj_dim < diameter)
    return f
    
  
def insertPerfectAbsorber(mysample,SlabX,SlabW=1,direction=None,k0=None,N=4):
    '''
    % mysample=insertPerfectAbsorber(mysample,SlabX,SlabW) : inserts a slab of refractive index material into a dataset
    % mysample : dataset to insert into
    % SlabX : middle coordinate
    % SlabW : half the width of slab
    % direction : direction of absorber: 1= left to right, -1=right to left, 2:top to bottom, -2: bottom to top
    '''

    if k0==None:
        k0=0.25/np.max(np.real(mysample));

    k02 = k0**2;
    
    if mysample.ndim < 3:
        mysample = np.expand_dims(mysample, 2)
        
        
    myXX=xx(mysample.shape[0], mysample.shape[1], mysample.shape[2])+mysample.shape[0]/2
    
    if np.abs(direction) <= 1:
        myXX=xx(mysample.shape[0], mysample.shape[1], mysample.shape[2])+mysample.shape[0]/2
    elif np.abs(direction) <= 2:
        myXX=yy(mysample.shape[0], mysample.shape[1], mysample.shape[2])+mysample.shape[1]/2
    elif np.abs(direction) <= 3:
        myXX=zz(mysample.shape[0], mysample.shape[1], mysample.shape[2])+mysample.shape[1]/2


    alpha=0.035*100/SlabW #; % 100 slices
    if direction > 0:
        #% mysample(abs(myXX-SlabX)<=SlabW)=1.0+1i*alpha*exp(xx(mysize,'corner')/mysize(1));  % increasing absorbtion
        myX=myXX-SlabX
    else:
        # %mysample(abs(myXX-SlabX)<=SlabW)=1.0+1i*alpha*exp((mysize(1)-1-xx(mysize,'corner'))/mysize(1));  % increasing absorbtion
        myX=SlabX+SlabW-myXX-1
        
        
    myMask= (myX>=0) * (myX<SlabW)

    alphaX=alpha*myX[myMask]
    
    PN=0;
    for n in range(0, N+1):
        PN = PN + np.power(alphaX,n)/factorial(n)

    k2mk02 = np.power(alphaX,N-1)*abssqr(alpha)*(N-alphaX + 2*1j*k0*myX[myMask])/(PN*factorial(N))
    
 
    mysample[myMask] = np.sqrt((k2mk02+k02)/k02)
            
    #np.array(mysample)[0]
    
    return mysample,k0


def insertSrc(mysize,myWidth=20,myOff=None, kx=0, ky=0):
    '''
    % mysample=insertSrc(mysample,myWidth,myOff,kx) : inserts a Gaussian source bar for the SEAGLE simulation
    % mysample : refractive indext to insert into
    % myWidth : Gaussian width
    % myOff : 2-component vector for position
    % kx : determins angle of emission (default 0)
    '''
    
    
    if myOff == None:
        myOff=((101, np.floor((mysample.shape[1],2)/2)))

    mySrc = np.zeros(mysize)+1j*np.zeros(mysize)
    myOffX=myOff[0]
    myOffY=myOff[1]

    if np.size(mysize) > 2:
        myOffZ=myOff[2]
        print("WARNING: Not yet implemented")
        mySrc[myOffX,:,:] = np.exp(1j*kx * (myOffY+yy(1,mysize[1],mysize[2]))) * np.exp(-abssqr(((myOffY+yy(1,mysize[1],mysize[2]))))/(2*myWidth**2))
        mySrc[myOffX,:,:] = mySrc[myOffX,:,:] * np.exp(1j*ky * (myOffZ+zz(1,mysize[1],mysize[2]))) * np.exp(-abssqr(((myOffZ+zz(1,mysize[1],mysize[2]))))/(2*myWidth**2))
    else:

        mySrc[myOffX,:] = np.exp(1j*kx * (myOffY+yy(1,mysize[1]))) * np.exp(- abssqr(myOffY+yy(1,mysize[1]))/(2*myWidth**2))
        
    return mySrc


def extract3D(input_matrix, newsize = ((300,300,300)), center=((-1,-1,-1))):
    # This extracts/padds a 3D array with zeros
    
    # This extracts/padds a 3D array with zeros
    Nx, Ny, Nz = input_matrix.shape
    cx, cy, cz = np.floor(Nx/2), np.floor(Ny/2), np.floor(Nz/2)
    
    # define center of new image
    Nx_new, Ny_new, Nz_new = newsize
    cx_new, cy_new, cz_new = np.floor(Nx_new/2), np.floor(Ny_new/2), np.floor(Nz_new/2)
    
    # Create new volume 
    myimage_new = np.zeros((Nx_new, Ny_new, Nz_new))
    
    
    if(np.sum(np.int16(newsize>input_matrix.shape))>0):
        # we place the old volume inside the new one
        myimage_new[int(cx_new-cx):int(cx_new+cx),int(cy_new-cy):int(cy_new+cy),int(cz_new-cz):int(cz_new+cz)]=input_matrix
    else:
        # take care, that old volume fits in new volume
        if(Nx>Nx_new):
            input_matrix = input_matrix[int(cx-cx_new):int(cx+cx_new),:,:]
        if(Ny>Ny_new):
            input_matrix = input_matrix[:,int(cy-cy_new):int(cy+cy_new),:]
        if(Nz>Nz_new):
            input_matrix = input_matrix[:,:,int(cz-cz_new):int(cz+cz_new)]
        myimage_new = input_matrix

    return myimage_new


class SEAGLE:
    ''' psi = SEAGLESolver(mysample, mySrc, myeps, k0, startPsi, showupdate) : solves the inhomogeneous SEAGLE equation
    % mysample : refractive index distribution
    % mySrc : source distribution
    % myeps : can be empty [], default: max(abs(kr2Mk02)) + 0.001;
    % k0 : inverse wavelength k0 = (2*pi)/lambda. default: smallest Nyquist sampling: k0=0.25/max(real(mysample))
    % startPsi : can be the starting field (e.g. from a previous result)
    % showupdate : every how many itereations should the result be shown.
    %
    % based on 
    % http://arxiv.org/abs/1601.05997
    % communicated by Benjamin Judkewitz
    % derived from Rainer Heintzmanns Code
    '''
    
    # initialize the SEAGLE operator
    showupdate = 10
    
    def __init__(self, mysample, mySrc, mysample_embb=1.33, k0=None, startPsi=None, showupdate=10):
    
        # initialize Class instance
        self.showupdate = showupdate
        self.mysample = mysample
        self.mySrc = mySrc
        self.mysample_embb = mysample_embb
        self.startPsi = startPsi 
        
        # INternal step counter 
        self.step_counter = 0
        self.nsteps = 1
        self.logs_path = './tensorflow_logs/'
        
        # Get the sizes 
        self.mysize_old = mysample.shape
        self.mysize_new = 2*np.array(mysample.shape)
        

        # Define Wavenumber
        self.lambda0 = lambda0
        self.pixelsize = pixelsize
        if k0 == None:
            k0=0.25/np.max(np.real(mysample));
        else:
            self.k0=k0
        self.k02 = self.k0**2;

    def computeModel(self):
        # Open a new session object 
        # self.sess = tf.Session()  
        
        print('Initialize the SEAGLE System')
        config = tf.ConfigProto()
        jit_level = 0
        if True:
            # Turns on XLA JIT compilation.
            jit_level = tf.OptimizerOptions.ON_1
        else:
            # Turns off XLA JIT compilation.
            jit_level = tf.OptimizerOptions.OFF_1
        config.graph_options.optimizer_options.global_jit_level = jit_level
        self.run_metadata = tf.RunMetadata()
  
  
        # initialize Session object
        if(True):
            self.sess = tf.InteractiveSession(config=config)
        else:
            self.sess = tf.Session(config=config)
                
        
        #%---------------------------------------------------------------------
        #                  START CODE HERE                                    #
        #%---------------------------------------------------------------------
        
        # First we need to pad our volume to get rid of reflection/wrap arounds
        if(False):
            self.mysample = extract3D(self.mysample, newsize = self.mysize_new)
            self.mySrc = extract3D(self.mySrc, newsize = self.mysize_new)
            self.myMask = extract3D(np.ones(self.mySrc.shape), newsize = self.mysize_new)>.5

        # get the scattering potential f=k0^2*(epsillon(r) - epsillon_embb)
        if(False):
            f_obj = self.k02*(self.mysample-self.mysample_embb)
            print('Doubble check the sample preparation, maybe false scattering potential!')
        else:
            f_obj = mysample#self.k02*(self.mysample)
        
        # Compute Greensfunction
        print('Now computing Greens function and its fourier transformed')
        mykr=self.k0 * rr(self.mysample.shape[0], self.mysample.shape[1], self.mysample.shape[2])
        self.greens_fkt = np.exp((1j*2*np.pi)*mykr) / np.abs(2*np.pi*mykr)
        mycenter = [int(self.mysample.shape[0]/2), int(self.mysample.shape[1]/2), int(self.mysample.shape[2]/2)]
        self.greens_fkt[mycenter[0], mycenter[1], mycenter[2]]=1.0/np.sqrt(2)

        # Compute Green's Function in Frequency Space
        Fac=5;  # What is the correct factor and why?
        self.greens_fkt_ft = my_ft3d_np(self.greens_fkt * Fac)

        # Compute the pump field inside the volume
        self.u_in = my_ift3d_np(my_ft3d_np(self.mySrc) * self.greens_fkt_ft)
            
        #%---------------------------------------------------------------------
        #                  START TENSORFLOW STUFF HERE                        #
        #%---------------------------------------------------------------------
        
        ## Port everything to Tensorflow 
        self.tf_mySrc = tf.constant(np.squeeze(self.mySrc), tf.complex64)
        self.tf_greens_fkt_ft = tf.constant(np.squeeze(self.greens_fkt_ft))
        self.tf_greens_fkt_ft = tf.cast(self.tf_greens_fkt_ft, tf.complex64)
        self.tf_f_obj = tf.constant(np.squeeze(f_obj), tf.complex64)
        self.tf_u_in = tf.constant(self.u_in, tf.complex64)
        # self.myMask = tf.constant(self.myMask)
        
        # Initialize field with the input field
        self.tf_u_real = tf.Variable(np.real(self.u_in))
        self.tf_u_imag = tf.Variable(np.imag(self.u_in))
        self.tf_u = tf.complex(self.tf_u_real, self.tf_u_imag)
        self.tf_u = tf.cast(self.tf_u, tf.complex64)
        
        # Cast to float32/complex64 to save memory
        #self.tf_f_obj_complex = tf.complex(self.tf_f_obj, tf.zeros(self.tf_f_obj.get_shape(), tf.float32))
        #self.tf_f_obj_complex = tf.cast(self.tf_f_obj_complex, tf.complex64)        


        #%---------------------------------------------------------------------
        #                  DEFINE FOWARD MODEL HERE                           #
        #%---------------------------------------------------------------------
        
        # perform one step in the iteration of SEAGLE
        # it computes   u' = (I - G diag(f))*u - u_in, which gives u_in in Lippmann-Schwinger equation
        # altenatively: u' = u - G (*) (u*f) - u_in, which gives u_in in Lippmann-Schwinger equation
        # compute FT(f*u)
        if(True):
            tf_u_f = my_ft3d(self.tf_f_obj*self.tf_u)
        else:
            tf_u_f = my_ft3d((1j*(self.tf_f_obj - 1.0)*2*np.pi*self.k0)*self.tf_u) # 2 pi circumference corresponds to one lambda. One pixels is k0 * lambda
        
        # convolve with G and add the input field
        # 0 = u - conv(G, f*u) - u_in
        self.tf_u_d = self.tf_u - my_ift3d(self.tf_greens_fkt_ft * tf_u_f) - self.tf_u_in 
        self.tf_u_d = tf.cast(self.tf_u_d, tf.complex64)

        # In order to get tf_u_d we need to minimize self.tf_u_d 
        # 1/2 * ||Au - uin||^2_2
        self.my_error = 1/2*tf_abssqr(self.tf_u_d)
        
         # op to write logs to Tensorboard
        # self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())

                    
            
    def minimize(self, my_lr=0.01):
        # here we try to minimize our error function using Gradient Descent - will take long probably..
        print('Define Cost-Function and Optimizer')
        self.my_error = 1/2 * tf.reduce_mean(tf_abssqr(self.tf_u_d))
        self.my_error = tf.cast(self.my_error, tf.float32)
        
        #my_optimizer = tf.train.AdamOptimizer(my_lr)
        my_optimizer = tf.train.AdamOptimizer(my_lr)
        self.train_op = my_optimizer.minimize(self.my_error)
        
    
    def compileGraph(self):
        print("Init operands ")
        init_op = tf.global_variables_initializer()
        print("run init")
        self.sess.run(init_op)
        
