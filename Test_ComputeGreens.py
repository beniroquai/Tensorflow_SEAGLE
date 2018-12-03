#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:15:27 2018

@author: bene
"""

# Start the code
import Helmholtz as helm
import numpy as np
import matplotlib.pyplot as plt

k0 = .25
k02 = k0**2;
Nx = 128
Ny = 128
Nz = 128
# get the scattering potential f=k0^2*(epsillon(r) - epsillon_embb)

# Compute Greensfunction
print('Now computing Greens function and its fourier transformed')
mykr=k0 * helm.rr(Nx, Ny, Nz)
GreensFkt = np.exp((1j*2*np.pi)*mykr) / np.abs(2*np.pi*mykr)
mycenter = [int(Nx/2), int(Ny/2), int(Nz/2)]
GreensFkt[mycenter[0], mycenter[1], mycenter[2]]=1.0/np.sqrt(2)

# Compute Green's Function in Frequency Space
Fac=5;  # What is the correct factor and why?
FTGreens = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(GreensFkt * Fac)))

plt.imshow(np.abs(FTGreens[:,:,64]))
plt.imshow(np.abs(GreensFkt[:,:,64]))