#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:53:53 2018

@author: bene
"""

import tensorflow as tf
import skimage.io as imgio
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

tf.reset_default_graph()

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


def fftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
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
    last_dim = len(tensor.get_shape()) - 1
    left, right = tf.split(tensor, 2, last_dim - 1)
    tensor = tf.concat([right, left], last_dim - 1)
    top, bottom = tf.split(tensor, 2, last_dim)
    tensor = tf.concat([bottom, top], last_dim)
    return tensor

def rr(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.linspace(-inputsize_x/2,inputsize_x/2, inputsize_x)
    y = np.linspace(-inputsize_y/2,inputsize_y/2, inputsize_y)
    z = np.linspace(-inputsize_z/2,inputsize_z/2, inputsize_z)
        
    xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
    r = np.sqrt(xx**2+yy**2+zz**2)
    r = np.transpose(r, [1, 0, 2]) #???? why that?!
    return r

class simple_deconv(object):
    
    def __init__(self):
        # here comes the forward model 
        p_imgfile = 'money.png'
        img_raw = plt.imread(p_imgfile)
        mysize = 200
        self.img_base = np.squeeze(img_raw[0:mysize , 0:mysize,1])
        self.myaperture = np.squeeze(rr(mysize , mysize,1)<mysize/12)
        mymeas = np.real(np.fft.ifft2( np.fft.fftshift(self.myaperture)*np.fft.fft2(self.img_base)))
        mymeas = mymeas/np.max(mymeas)
        self.mymeas = mymeas+(np.random.rand(mysize, mysize)>.9)*.2
        #plt.imshow(mymeas)

    def ComputeSystem(self):
        # tensorflow stuff
        # Initialize field with the input field
        tf_guess = tf.get_variable("tf_guess",
                                         shape=self.img_base.shape,
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(np.mean(self.img_base)))
    
        self.tf_img = tf.constant(self.img_base)
        self.tf_guess_cmplx = tf.complex(tf_guess, 0*tf_guess)
        self.tf_meas = tf.cast(tf.constant(self.mymeas), tf.float32)
        tf_pupil = tf.constant(1.*self.myaperture)
        self.tf_pupil_complx = tf.cast(tf.complex(tf_pupil, tf_pupil*0), tf.complex64)

    def FwdSystem(self):
        # Fwd model
        self.tf_fwd = tf.real(my_ift2d(my_ft2d(self.tf_guess_cmplx)*self.tf_pupil_complx))
        self.tf_cost = tf.reduce_mean(tf.square(self.tf_meas-self.tf_fwd) + .1*tf.expand_dims(self.tf_fwd, 2)) 
        return self.tf_cost, self.tf_fwd

    def Optimize(self):
        # Optimization part
        tf_optimize = tf.train.AdadeltaOptimizer(1).minimize(self.tf_cost)
        op_init = tf.initialize_all_variables()
        
        sess = tf.Session()
        sess.run(op_init)
        
        for i in range(50):
            plt.imshow(sess.run(self.tf_guess)), plt.colorbar(), plt.show()
            cost, _ = sess.run([self.tf_cost, tf_optimize])
            print('Opt-Val@'+str(i)+' is '+str(cost))
        
        
        plt.imshow(sess.run(self.tf_meas)), plt.colorbar(), plt.show()
        plt.imshow(sess.run(self.tf_guess)), plt.colorbar(), plt.show()

# example for simple deconv        
myDeconv = simple_deconv()
myDeconv.ComputeSystem()
cost, val = myDeconv.FwdSystem()
