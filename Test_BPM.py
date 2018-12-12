import SEAGLE as myseagle
import numpy as np
import matplotlib.pyplot as plt



Nx, Ny = 128, 128
uin = np.ones((Nx, Ny))+ 1j*np.zeros((Nx, Ny))
lambda0= 561e-9 # in nm
dx, dy, dz = 1e-6,1e-6,10e-6

# Initializethe Propagation Model
myBPM = myseagle.BPM(uin, mysize=(Nx,Ny), lambda0=lambda0, sampling=(dx, dy, dz), nEmbb=1.33)

#%% Add one PRopagation step to the que
dn = .5
obj = np.squeeze((myseagle.rr(Nx,Ny,0)<10)*dn)

for i in range(20):
    if(i==10):
        myf = np.exp(1j*obj*2*np.pi) # some arbitrary OPD 
    else:
        myf =0*np.exp(1j*obj*2*np.pi) # some arbitrary OPD 
    myBPM.PropagateStep(dz=dz, myf=myf)

#%% Execute the propgation 
myU = myBPM.ExecPropagation()

#%% Visualize
plt.title('My ABS'), plt.imshow(np.abs(myU), cmap='gray'), plt.colorbar(), plt.show()
plt.title('My Angle'), plt.imshow(np.angle(myU), cmap='gray'), plt.colorbar(), plt.show()

#%% Get Pupil
Po = myBPM.GetPupil(.25)
plt.imshow(Po)

#%% Filter Result
myU_filtered = myBPM.Filter(myU, Po)
#%% Visualize
plt.title('My ABS'), plt.imshow(np.abs(myU_filtered), cmap='gray'), plt.colorbar(), plt.show()
plt.title('My Angle'), plt.imshow(np.angle(myU_filtered), cmap='gray'), plt.colorbar(), plt.show()
