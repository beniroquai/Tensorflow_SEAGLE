#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:33:12 2018

@author: bene
"""

import numpy as np



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
        

    return myimage_new


Nx, Ny, Nz=((200,200,200))
input_matrix=np.random.rand(Nx, Ny, Nz)

# expand
newsize = np.int16(2*np.array(input_matrix.shape))
mymatrix = extract3D(input_matrix, newsize = newsize)

# extract
newsize = np.int16(.5*np.array(input_matrix.shape))
mymatrix = extract3D(input_matrix, newsize = newsize)





