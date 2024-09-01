import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

def macroscopic(fin,nx,ny,ci):
    rho =np.sum(fin,axis=0) #rho(밀도) 구하는 곳
    u=np.zeros((2,nx,ny))
    for i in range(9):
        u[0,:,:] += ci[i,0]*fin[i,:,:]
        u[1,:,:] += ci[i,1]*fin[i,:,:]
    u /= rho 
    return rho, u

def equilibrium(rho, u, ci, wi, nx, ny):
    usqr=(3/2)*(u[0]**2+u[1]**2)
    feq=np.zeros(9,nx,ny)
    for i in range(9):
        cu = 3*(ci[i,0]*u[0,])


https://www.youtube.com/watch?v=sFZ2QfEXiTo&t=90s

