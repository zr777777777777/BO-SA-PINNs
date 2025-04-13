import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab as pylabplt
from scipy.interpolate import griddata
import time
import math
import scipy.special as sp
import os
os.environ["DDEBACKEND"] = "pytorch"
import deepxde as dde
from deepxde.backend import backend_name, tf, torch
from deepxde import utils
from bayes_opt import BayesianOptimization
from scipy.stats import qmc  # LHS
from scipy.optimize import differential_evolution

if backend_name == "tensorflow" or backend_name == "tensorflow.compat.v1":
   be = tf
elif backend_name == "pytorch":
   be = torch

# Default configuration for floating-point numbers
dde.config.set_default_float("float64")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#-------------------------------------------------------------------------------#
# Problem definition: A two-dimensional frequency-domain electromagnetic wave is incident on a dielectric circular scatterer.
#-------------------------------------------------------------------------------#
# Geometrical characteristics 
L = 2.0   # Length of rectangle (R) 
l = 2.0   # Width of rectangle (R)

# Bounds of x, y
x_lower = -L/2.0
x_upper =  L/2.0
y_lower = -l/2.0
y_upper =  l/2.0

# Radius of circle (S)
R = 0.25
# Center of the circle
Cx = 0.0
Cy = 0.0

# Geometry definition
outer = dde.geometry.Rectangle(xmin=(x_lower,y_lower), xmax=(x_upper,y_upper))
inter = dde.geometry.Disk([Cx, Cy], R)
geom = outer

eps1_r = 1.0    # Real part of electric permittivity outside the disk
eps1_i = 0.0    # Imaginary part of electric permittivity outside the disk
eps2_r = 1.5     # Real part of electric permittivity inside the disk 
eps2_i = 0.0     # Imaginary part of electric permittivity inside the disk

mu_r = 1.0

Z_0 = 1.0    # In vacuum (the incident plane wave is injected in vacuum)

v_0_1 = 0.3  # Velocity 1 (outside the disk) 

v_0_2 = v_0_1/np.sqrt(mu_r*eps2_r) # Velocity 2 (inside the disk) if you change eps2_r, you must also change v_0_2 (in case eps2_r = 4.0, we have v_0_2 = 0.15)

freq  = 0.3 

lam1  = v_0_1/freq # Wave length 1 (outside the disk)
lam2  = v_0_2/freq # Wave length 2 (inside the disk)

omk   = (2.0*math.pi*freq)/v_0_1 # Pulsation
omk2  = (2.0*math.pi*freq)/v_0_2 # Pulsation

kap   = 1.0/(omk*mu_r) # Constant used in the definition of the PDE 

ampE  = 1.0 # Amplitude of the electric field
ampH  = ampE/omk # Amplitude of the magnetic field

# define PDE residual
def pde(u, v):
    Ez_r, Ez_i = v[:,0:1], v[:,1:2]

    d2Ez_r_x2 = dde.grad.hessian(Ez_r, u, i=0, j=0) # d2Ez_r/dx2
    d2Ez_r_y2 = dde.grad.hessian(Ez_r, u, i=1, j=1) # d2Ez_r/dy2
    d2Ez_i_x2 = dde.grad.hessian(Ez_i, u, i=0, j=0) # d2Ez_i/dx2
    d2Ez_i_y2 = dde.grad.hessian(Ez_i, u, i=1, j=1) # d2Ez_i/dy2

    curl2E_z_r = - (d2Ez_r_x2 + d2Ez_r_y2)
    curl2E_z_i = - (d2Ez_i_x2 + d2Ez_i_y2)

    d2 = (u[:,0:1] - Cx)*(u[:,0:1] - Cx) + (u[:,1:2] - Cy)*(u[:,1:2] - Cy)

    d  = be.sqrt(d2) # Distance between a point X(x,y) and the center of the disk (Cx,Cy)

    cond = be.less(d[:], R)
   # A little reminder : kap = 1.0/(mu_r*omk)
   # Equation (31.1) with a leger modification in term of coefficients

    fEz_1 = omk*(eps1_r*Ez_r - eps1_i*Ez_i) - kap*curl2E_z_r  # outside the disk
    fEz_2 = omk*(eps2_r*Ez_r - eps2_i*Ez_i) - kap*curl2E_z_r  # inside the disk
    fEz = be.where(cond, fEz_2, fEz_1)

   # Equation (31.2) with a leger modification in term of coefficients
    gEz_1 =-omk*(eps1_r*Ez_i + eps1_i*Ez_r) + kap*curl2E_z_i  # outside the disk
    gEz_2 =-omk*(eps2_r*Ez_i + eps2_i*Ez_r) + kap*curl2E_z_i  # inside the disk
    gEz = be.where(cond, gEz_2, gEz_1)

   # 返回实部和虚部的残差
    return [fEz, gEz]

# We assume that the wave vector has only one component along the x-axis
kx1 = omk   # outside the disk
kx2 = omk2  # inside the disk

# Definition of Bessel and Hankel functions
def bessel_function(x, order):
    return sp.jv(order, x)

def bessel_derivative(x, order):
    return sp.jvp(order,x,n=1)

def hankel_first_kind(x, order):
    return sp.hankel1(order, x)

def hankel_first_kind_derivative(x, order):
    return sp.h1vp(order,x,n=1)

# Defintion of the field outside the disk
def u_e(r,theta):
    N = 100
    u_e = complex(0.0,0.0)

    for n in range(1, N + 1):
        i = complex(0,1)
        m = float(n)

        An1 = mu_r*kx2*bessel_derivative(-kx2*R,m)*bessel_function(-kx1*R,m)-kx1*bessel_function(-kx2*R,m)*bessel_derivative(-kx1*R,m)
        An2 = kx1*hankel_first_kind_derivative(-kx1*R,m)*bessel_function(-kx2*R,m)-mu_r*kx2*bessel_derivative(-kx2*R,m)*hankel_first_kind(-kx1*R,m)
        An  = An1/An2

        u_e = u_e + (i**m)*(bessel_function(-kx1*r,m)+ An*hankel_first_kind(-kx1*r,m))*np.cos(m*theta) 

    A01 = mu_r*kx2*bessel_derivative(-kx2*R,0)*bessel_function(-kx1*R,0)-kx1*bessel_function(-kx2*R,0)*bessel_derivative(-kx1*R,0)
    A02 = kx1*hankel_first_kind_derivative(-kx1*R,0)*bessel_function(-kx2*R,0)-mu_r*kx2*bessel_derivative(-kx2*R,0)*hankel_first_kind(-kx1*R,0)
    A0  = A01/A02

    u_e = bessel_function(-kx1*r,0) + A0*hankel_first_kind(-kx1*r,0) + 2*u_e

    return u_e 

# Definition of the field inside the disk
def u_i(r,theta):
    N = 100
    u_i = complex(0.0,0.0)

    for n in range(1, N + 1):
        i = complex(0,1)
        m = float(n)

        Bn1 = kx1*hankel_first_kind_derivative(-kx1*R,m)*bessel_function(-kx1*R,m)-kx1*hankel_first_kind(-kx1*R,m)*bessel_derivative(-kx1*R,m)
        Bn2 = kx1*hankel_first_kind_derivative(-kx1*R,m)*bessel_function(-kx2*R,m)-mu_r*kx2*bessel_derivative(-kx2*R,m)*hankel_first_kind(-kx1*R,m)
        Bn  = Bn1/Bn2

        u_i = u_i + (i**m)*Bn*bessel_function(-kx2*r,m)*np.cos(m*theta) 

    B01 = kx1*hankel_first_kind_derivative(-kx1*R,0)*bessel_function(-kx1*R,0)-kx1*hankel_first_kind(-kx1*R,0)*bessel_derivative(-kx1*R,0)
    B02 = kx1*hankel_first_kind_derivative(-kx1*R,0)*bessel_function(-kx2*R,0)-mu_r*kx2*bessel_derivative(-kx2*R,0)*hankel_first_kind(-kx1*R,0)
    B0  = B01/B02

    u_i = B0*bessel_function(-kx2*r,0) + 2*u_i

    return u_i

# Global field
def u(r, theta):

    u_i_values = u_i(r, theta)
    u_e_values = u_e(r, theta)

    return np.where(r <= R, u_i_values, u_e_values)

# Definition of a plane monochromatic wave using the Bessel and Hankel functions (in vacuum)

def plane_wave(r,theta):
    N = 100
    u_inc = complex(0.0,0.0)

    for n in range (1, N+1):
        i = complex(0.0,1.0)
        m = float(n)
        u_inc = u_inc + (i**m)*bessel_function(-kx1*r,m)*np.cos(m*theta)

    u_inc = bessel_function(-kx1*r,0) + 2*u_inc
    return u_inc


# This function enables us to determine if the point (x,y) belongs to the boundary, it returns either True or False
def boundary(x, on_boundary): 
    return on_boundary 

# definition of boundary conditions
def EHx_abc_r(x, y,_):
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    nx  = 0.0  
    ny  = 0.0
    nx  = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx  = be.where(x[:,0:1] == x_upper,  1.0,  nx)
    ny  = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny  = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz
    nEx =  ny * y[:,0:1] 
    nEy = -nx * y[:,0:1]

    # Compute H_r from E_i
    dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
    dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy

    # A little reminder : kap = 1/(mu_r*omk) 
    Hx =  -kap*dEz_i_y # Hrx function of Ezi Equation (36.1)
    Hy =   kap*dEz_i_x # Hry function of Ezi Equation (36.2)

    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    nHz = -nx*Hy + ny*Hx
    
    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the first equation of Equation (33)
    rEHx = nEx - Z_0*nHxn  # along the x-axis
    rEHy = nEy - Z_0*nHyn  # along the y-axis
    
    # Components of the incident plane wave, wave vector has an only component along the x-axis
    Ezinc   =   ampE*be.cos(kx1*x[:,0:1])
    Hyinc   =  -ampH*kx1*be.cos(kx1*x[:,0:1]) 
   
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
    nExinc  =  ny*Ezinc
    nEyinc  = -nx*Ezinc 
   
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    nHzinc  = -nx*Hyinc 
   
    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
   
    # Calculate the right side of the first equation of Equation (33)
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis

    return rEHx - rEHxinc

def EHy_abc_r(x, y,_):    
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    nx  = 0.0  
    ny  = 0.0
    nx  = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx  = be.where(x[:,0:1] == x_upper,  1.0,  nx)
    ny  = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny  = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz
    nEx =  ny * y[:,0:1] 
    nEy = -nx * y[:,0:1]

    # Compute H_r from E_i
    dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
    dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy

    # A little reminder : kap = 1/(mu_r*omk) 
    Hx =  -kap*dEz_i_y # Hrx function of Ezi Equation (36.1)
    Hy =   kap*dEz_i_x # Hry function of Ezi Equation (36.2)

    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 

    # Calculate the left side of the first equation of Equation (33)
    rEHx = nEx - Z_0*nHxn  # along the x-axis
    rEHy = nEy - Z_0*nHyn  # along the y-axis

    # Components (real part) of the incident plane wave, wave vector has an only component along the x-axis
    Ezinc   =   ampE*be.cos(kx1*x[:,0:1])
    Hyinc   =  -ampH*kx1*be.cos(kx1*x[:,0:1]) 

    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
    nExinc  =  ny*Ezinc
    nEyinc  = -nx*Ezinc

    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    nHzinc  = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc

    # Calculate the right side of the first equation of Equation (33)
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis

    return rEHy - rEHyinc


def EHx_abc_i(x, y,_):
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    nx = 0.0
    ny = 0.0
    nx = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx = be.where(x[:,0:1] == x_upper,  1.0,  nx)
    ny = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz
    nEx =  ny * y[:,1:2]
    nEy = -nx * y[:,1:2]

    # Compute H_i from E_r
    dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
    dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy

    # A little reminder : kap = 1/(mu_r*omk)
    Hx =   kap*dEz_r_y   # Hix function of Ezr Equation (36.3)
    Hy =  -kap*dEz_r_x  # Hix function of Ezr Equation (36.4)

    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 

    # Calculate the left side of the second equation of Equation (33)
    rEHx = nEx - Z_0*nHxn   # along the x-axis
    rEHy = nEy - Z_0*nHyn   # along the y-axis

    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
    Ezinc =   ampE*be.sin(-kx1*x[:,0:1])
    Hyinc =  -ampH*kx1*be.sin(-kx1*x[:,0:1]) 

    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
    nExinc =  ny*Ezinc
    nEyinc = -nx*Ezinc 

    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    nHzinc = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc

    # Calculate the right side of the second equati on of Equation (33)
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis

    return rEHx - rEHxinc

def EHy_abc_i(x, y,_):
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    nx = 0.0
    ny = 0.0
    nx = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx = be.where(x[:,0:1] == x_upper,  1.0,  nx)
    ny = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz
    nEx =  ny * y[:,1:2]
    nEy = -nx * y[:,1:2]

    # Compute H_i from E_r  
    dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
    dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy

    # A little reminder : kap = 1/(mu_r*omk) 
    Hx =   kap*dEz_r_y   # Hix function of Ezr Equation (36.3)
    Hy =  -kap*dEz_r_x  # Hix function of Ezr Equation (36.4)

    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    nHxn =  ny*nHz 
    nHyn = -nx*nHz

    # Calculate the left side of the second equation of Equation (33)
    rEHx = nEx - Z_0*nHxn   # along the x-axis
    rEHy = nEy - Z_0*nHyn   # along the y-axis 

    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
    Ezinc =   ampE*be.sin(-kx1*x[:,0:1])
    Hyinc =  -ampH*kx1*be.sin(-kx1*x[:,0:1])  

    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1) 
    nExinc =  ny*Ezinc
    nEyinc = -nx*Ezinc 

    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    nHzinc = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc

    # Calculate the right side of the second equation of Equation (33)
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis

    return rEHy - rEHyinc

# boundary conditions loss
abc_bc_EHx_r = dde.icbc.OperatorBC(geom, EHx_abc_r, boundary)
abc_bc_EHy_r = dde.icbc.OperatorBC(geom, EHy_abc_r, boundary)
abc_bc_EHx_i = dde.icbc.OperatorBC(geom, EHx_abc_i, boundary)
abc_bc_EHy_i = dde.icbc.OperatorBC(geom, EHy_abc_i, boundary)


start = time.time()

#-------------------------------------------------------------------------------#
# TG activation function
#-------------------------------------------------------------------------------#
def wavelet_tanh_gaussian(x):
    return torch.tanh(x) * torch.exp(-x**2 / 2)

#-------------------------------------------------------------------------------#
# BO for selecting hyperparameters
#-------------------------------------------------------------------------------#

# Global variables to store the best weights and error
global_best_error = float("inf")  # Initialize with a large value
global_best_weights = None  # Variable to store the best network weights

def train_and_evaluate(hparams):
    global global_best_error, global_best_weights
    # loss function weights, initial values are from others' research
    w_pde = hparams.get("w_pde", 0.015)
    w_abc = hparams.get("w_abc", 0.125)

    # initial learning rate for ADAM
    lr=hparams.get("lr",0.005)

    # number of sampling points
    num_domain_idx = int(round(hparams.get("num_domain_idx",100)))
    num_boundary_idx = int(round(hparams.get("num_boundary_idx",50)))
    num_domain=num_domain_idx * 50
    num_boundary = num_boundary_idx * 50

    # network architecture
    num_layers = int(round(hparams.get("num_layers", 3)))
    num_neurons = int(round(hparams.get("num_neurons", 50)))

    loss_weights = [w_pde, w_pde, w_abc, w_abc, w_abc, w_abc]
    
    # Define PDE Data
    data = dde.data.PDE(
            geom,
            pde,
            [abc_bc_EHx_r, abc_bc_EHy_r, abc_bc_EHx_i, abc_bc_EHy_i],
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_test=2500
        )

    # define fundamental nerual network
    layer_sizes = [2] + [num_neurons] * num_layers + [2] 
    net = dde.nn.FNN(layer_sizes, wavelet_tanh_gaussian, "Glorot uniform")

    # define nerual network combined loss and data
    model = dde.Model(data, net)
    
    # pre-training
    model.compile("adam", lr=lr, loss_weights=loss_weights) 
    losshistory, train_state = model.train(iterations=500)

    # Get current model parameters (weights)
    current_weights = list(model.net.parameters())  # Get all model parameters (weights)

    # Calculate L2 relative error, then generate tuples: (configuration, L2 relative error)
    nbx, nby = 100, 100
    xc = np.linspace(x_lower, x_upper, nbx)
    yc = np.linspace(y_lower, y_upper, nby)
    x_grid, y_grid = np.meshgrid(xc, yc)
    xy_grid = np.vstack((np.ravel(x_grid), np.ravel(y_grid))).T

    predictions = model.predict(xy_grid)
    pred_Erz = predictions[:, 0]
    
    pred_Eiz = predictions[:, 1]
    cartesian_u_values = u(
        np.sqrt(x_grid ** 2 + y_grid ** 2),
        np.arctan2(y_grid, x_grid)
    )
    
    cartesian_u_values = np.nan_to_num(cartesian_u_values)
    real_u_values = np.real(cartesian_u_values).ravel()
    imaginary_u_values = np.imag(cartesian_u_values).ravel()
    l2_error_erz = dde.metrics.l2_relative_error(real_u_values, pred_Erz)
    l2_error_eiz = dde.metrics.l2_relative_error(imaginary_u_values, pred_Eiz)
    
    mean_l2_error = 0.5 * (l2_error_erz + l2_error_eiz)

    # Update global best weights if the new error is smaller
    if mean_l2_error < global_best_error:
        global_best_error = mean_l2_error
        global_best_weights = current_weights  # Store the best weights

    return mean_l2_error

# Bayesian Optimization function
def objective_for_bayes(w_pde, w_abc,lr,num_layers,num_neurons,num_domain_idx, num_boundary_idx):
    # step:0.0001
    w_pde = round(w_pde, 4)
    w_abc = round(w_abc, 4)

    # step:0.001
    lr = round(lr, 3)

    # step:1
    num_layers = int(round(num_layers))
    num_neurons = int(round(num_neurons))

    # step:50
    num_domain_idx = int(round(num_domain_idx))
    num_boundary_idx = int(round(num_boundary_idx))
    
    # types of hyperparameters
    hparams = {
        "w_pde": w_pde,
        "w_abc": w_abc,
        "lr":lr,
        "num_domain_idx": num_domain_idx,
        "num_boundary_idx": num_boundary_idx,
        "num_layers": num_layers,
        "num_neurons": num_neurons,
    }

    error = train_and_evaluate(hparams)

    return -error  # Maximize the negative of the error

# Bayesian Optimization setup
def run_bayesian_optimization():
    # hyperparameter search space
    pbounds = {
        'w_pde': (0.01, 0.10),
        'w_abc': (0.01, 0.25),
        'lr': (0.001, 0.01),
        'num_domain_idx': (10, 100),
        'num_boundary_idx': (2, 60),
        'num_layers': (2, 8),
        'num_neurons': (20, 80),
    }

    # random seed
    optimizer = BayesianOptimization(
        f=objective_for_bayes,
        pbounds=pbounds,
        verbose=2,
        random_state=1234
    )

    # Using Latin Hypercube Sampling for initial points
    initial_samples = 10
    sampler = qmc.LatinHypercube(d=len(pbounds))
    lhs_samples = sampler.random(n=initial_samples)
    bounds = np.array(list(pbounds.values()))
    l_bounds = bounds[:, 0]
    u_bounds = bounds[:, 1]
    scaled_samples = qmc.scale(lhs_samples, l_bounds, u_bounds)

    for sample in scaled_samples:
        params = dict(zip(pbounds.keys(), sample))
        optimizer.probe(params, lazy=True)
    
    # 20 iterations for updating surrogate model and return the optimal hyperparameters group every 10 iterations
    total_iterations = 20
    batch_size = 10
    for i in range(1, int(total_iterations / batch_size) + 1):
        optimizer.maximize(init_points=0, n_iter=batch_size)
        current_best = optimizer.max
        best_params = current_best["params"]
        best_error = -current_best["target"]
        print(f"Best Params: {best_params}")
        print(f"Best Error: {best_error}")

    best_result = optimizer.max
    best_params = best_result["params"]
    best_error = -best_result["target"]

    print("\n===== Bayesian Optimization Completed =====")
    print("Best Parameters:", best_params)
    print("Best Error:", best_error)

    # Return the best parameters, best error, and best saved weights
    return best_params, best_error, global_best_weights

# Run Bayesian Optimization
best_params, best_error, best_weights = run_bayesian_optimization()

# get optimal hyperparameters selected by BO
w_pde = best_params['w_pde']
w_abc = best_params['w_abc']
lr=best_params['lr']
num_domain = int(round(best_params['num_domain_idx'] )) * 50
num_boundary = int(round(best_params['num_boundary_idx'] )) * 50
num_layers = int(round(best_params['num_layers']))
num_neurons = int(round(best_params['num_neurons']))

# Config subsequent training network, sampling points and loss function weights
loss_weights = [w_pde, w_pde, w_abc, w_abc, w_abc, w_abc]

# Define PDE Data
data = dde.data.PDE(
        geom,
        pde,
        [abc_bc_EHx_r, abc_bc_EHy_r, abc_bc_EHx_i, abc_bc_EHy_i],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_test=2500
    )

layer_sizes = [2] + [num_neurons] * num_layers + [2] 
net = dde.nn.FNN(layer_sizes, wavelet_tanh_gaussian, "Glorot uniform")
model = dde.Model(data, net)

#-------------------------------------------------------------------------------#
# Define callback class for calculating L2 relative error in training process
#-------------------------------------------------------------------------------#
test_points = data.test_x
r_test = np.sqrt(test_points[:, 0]**2 + test_points[:, 1]**2)
theta_test = np.arctan2(test_points[:, 1], test_points[:, 0])
u_test = u(r_test, theta_test)
u_test = np.nan_to_num(u_test)
real_u_test = np.real(u_test)
imag_u_test = np.imag(u_test)

class L2ErrorCallback(dde.callbacks.Callback):
    def __init__(self, test_x, real_u, imag_u, every=1000):
        super().__init__()
        self.test_x = test_x
        self.real_u = real_u
        self.imag_u = imag_u
        self.every = every

    def on_epoch_end(self):
        current_step = self.model.train_state.step
        if current_step % self.every == 0:

            pred = self.model.predict(self.test_x)
            pred_Erz = pred[:, 0]
            pred_Eiz = pred[:, 1]

            l2_erz = dde.metrics.l2_relative_error(self.real_u, pred_Erz)
            l2_eiz = dde.metrics.l2_relative_error(self.imag_u, pred_Eiz)

            print(f"Step {current_step}: L2 Erz error: {l2_erz:.6e}, L2 Eiz error: {l2_eiz:.6e}")

l2_cb = L2ErrorCallback(test_points, real_u_test, imag_u_test, every=1000)
dropout_uncertainty = dde.callbacks.DropoutUncertainty(period=1000)

#-------------------------------------------------------------------------------#
# EMA for adjusting loss function weights
#-------------------------------------------------------------------------------#
class AdaptiveWeights(torch.nn.Module):
    def __init__(self, init_w_pde, init_w_abc):
        super(AdaptiveWeights, self).__init__()
        # Initialize weights and ensure they require gradients
        self.w_pde = torch.nn.Parameter(torch.tensor(init_w_pde, dtype=torch.float32, requires_grad=True))
        self.w_abc = torch.nn.Parameter(torch.tensor(init_w_abc, dtype=torch.float32, requires_grad=True))

    def forward(self):
        return self.w_pde, self.w_abc

adaptive_weights = AdaptiveWeights(w_pde, w_abc)

class EMAUpdater:
    def __init__(self, beta=0.999, gamma=0.999, init_pde=1.0, init_abc=1.0, init_w_pde=0.1, init_w_abc=0.1):
        self.beta = beta  
        self.gamma = gamma 
        self.m_pde = init_pde
        self.m_abc = init_abc
        self.w_pde = init_w_pde
        self.w_abc = init_w_abc

    def update(self, loss_pde, loss_abc):
        # calculate EMA of loss
        self.m_pde = self.beta * self.m_pde + (1 - self.beta) * loss_pde
        self.m_abc = self.beta * self.m_abc + (1 - self.beta) * loss_abc

        # calculate the new weight ratio based on EMA
        total = self.m_pde + self.m_abc
        new_w_pde = self.m_pde / total
        new_w_abc = self.m_abc / total

        # blend the new weights with the old ones to smooth the update
        self.w_pde = self.gamma * self.w_pde + (1 - self.gamma) * new_w_pde
        self.w_abc = self.gamma * self.w_abc + (1 - self.gamma) * new_w_abc

        # limit the weight value range
        self.w_pde = torch.clamp(self.w_pde, 0.01, 0.15)
        self.w_abc = torch.clamp(self.w_abc, 0.01, 0.25)

        return self.w_pde, self.w_abc

#-------------------------------------------------------------------------------#
# RAR-D for adjusting sampling points distribution
#-------------------------------------------------------------------------------#
# RAR-D is based on optimal network parameters after the first stage
parameters = list(model.net.parameters())

# Get the parameter names for the FCNN model
layer_names = []
for i in range(len(parameters)):
    if i % 2 == 0:  # Weight parameters (even index)
        layer_names.append(f"linears.{i//2}.weight")
    else:  # Bias parameters (odd index)
        layer_names.append(f"linears.{i//2}.bias")

# Now, ensure that best_weights is properly matched to the expected layer names
best_weights_dict = {layer_names[i]: best_weights[i] for i in range(len(best_weights))}

# Now load the dictionary into the model's net
model.net.load_state_dict(best_weights_dict)

def filter_unique_points(existing_points, new_points, tol=1e-3):
    #Check each point in new_points. If the distance to any point in existing_points is less than tol, the point is considered duplicate and will not be retained.
    unique_points = []

    for p in new_points:
        # Calculate the Euclidean distance between the new point p and all existing points

        dists = np.linalg.norm(existing_points - p, axis=1)

        if np.min(dists) > tol:
            unique_points.append(p)

    return np.array(unique_points)

num_rar_iterations = 10

for i in range(num_rar_iterations):
    # Generate candidate points
    candidate_points = geom.random_points(1000)

    # Compute the PDE residual for each candidate point (two equations: real and imaginary)
    residuals_pde = model.predict(candidate_points, operator=pde) 

    # Calculate the residual for each point
    residuals_combined = np.sqrt(np.array(residuals_pde[0])**2 + np.array(residuals_pde[1])**2).ravel()

    # Constructing probability distributions
    sum_res = np.sum(residuals_combined)
    if sum_res == 0:
        p_distribution = np.ones_like(residuals_combined) / len(residuals_combined)
    else:
        p_distribution = residuals_combined / sum_res
    
    # Select new points based on probability (here 50 points are selected)
    num_new_points = 50

    new_indices = np.random.choice(
        a=len(candidate_points),
        size=num_new_points,
        replace=False,
        p=p_distribution
    )
    new_points = candidate_points[new_indices]

    # Filter out points that are duplicated with existing training points before adding them
    existing_points = data.train_x_all
    new_points_unique = filter_unique_points(existing_points, new_points, tol=1e-3)
    if new_points_unique.shape[0] > 0:
        data.add_anchors(new_points_unique)
    else:
        print(f"{i+1} epoch: no new points")

print("RAR-D adaptive sampling finished.")

#-------------------------------------------------------------------------------#
# training process
#-------------------------------------------------------------------------------#
def train_and_evaluate_with_dynamic_weights():
    w_pde1 = adaptive_weights.w_pde
    w_abc1 = adaptive_weights.w_abc
    
    loss_weights = [w_pde1.item(), w_pde1.item(), w_abc1.item(), w_abc1.item(), w_abc1.item(), w_abc1.item()]
    
    layer_sizes = [2] + [num_neurons] * num_layers + [2]
    net = dde.nn.FNN(layer_sizes, wavelet_tanh_gaussian, "Glorot uniform")
    model = dde.Model(data, net)
    
    # Convert the generator to a list to get the length and access parameters
    parameters = list(model.net.parameters())

    # Get the parameter names for the FNN model
    layer_names = []
    for i in range(len(parameters)):
        if i % 2 == 0:  # Weight parameters (even index)
            layer_names.append(f"linears.{i//2}.weight")
        else:  # Bias parameters (odd index)
            layer_names.append(f"linears.{i//2}.bias")
    # Now, ensure that best_weights is properly matched to the expected layer names
    best_weights_dict = {layer_names[i]: best_weights[i] for i in range(len(best_weights))}

    # Now load the dictionary into the model's net
    model.net.load_state_dict(best_weights_dict)

    # first 1000 for stable training
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    losshistory, train_state = model.train(iterations=1000, display_every=1000, callbacks=[l2_cb])

    # calculate current pde loss and boundary loss
    loss_pde = sum(train_state.loss_train[:2]) 
    loss_abc = sum(train_state.loss_train[2:6])

    # config EMA
    ema_updater = EMAUpdater(
            beta=0.999, gamma=0.999,
            init_pde=loss_pde,    
            init_abc=loss_abc,    
            init_w_pde=w_pde1,         
            init_w_abc=w_abc1      
        )
    
    for step in range(1000, 5000, 1000):
        loss_pde = sum(train_state.loss_train[:2])  
        loss_abc = sum(train_state.loss_train[2:6]) 

        # adjust loss function weights every 1000 iterations
        if step % 1000 == 0:
            model.compile("adam", lr=lr, loss_weights=loss_weights, decay=("step", 1000, 0.9))
            
            losshistory, train_state = model.train(iterations=1000, display_every=1000, callbacks=[l2_cb])
            updated_w_pde, updated_w_abc = ema_updater.update(loss_pde, loss_abc)
            print(f"Updated w_pde: {updated_w_pde}")
            print(f"Updated w_abc: {updated_w_abc}")
           
            loss_weights = [updated_w_pde.item(), updated_w_pde.item(), updated_w_abc.item(), updated_w_abc.item(), updated_w_abc.item(), updated_w_abc.item()]
    
    # third stage : L-BFGS for stable traing
    dde.optimizers.set_LBFGS_options(maxiter=10000)
    model.compile("L-BFGS", loss_weights=loss_weights)
    losshistory, train_state = model.train(callbacks=[l2_cb])
    return model, losshistory, train_state, updated_w_pde, updated_w_abc

model, losshistory, train_state, updated_w_pde, updated_w_abc = train_and_evaluate_with_dynamic_weights()

#-------------------------------------------------------------------------------#
# Caculate L2 relative error and plot
#-------------------------------------------------------------------------------#
# Plotting training loss history
dde.utils.external.plot_loss_history(losshistory, fname='loss_history.png')

# Calculate L2 error
nbx = 1000
nby = 1000
xc = np.linspace(x_lower, x_upper, nbx)
yc = np.linspace(y_lower, y_upper, nby)
x_grid, y_grid = np.meshgrid(xc, yc)
xy_grid = np.vstack((np.ravel(x_grid), np.ravel(y_grid))).T

predictions = model.predict(xy_grid)
pred_Erz = predictions[:, 0].reshape(nbx, nby)
pred_Eiz = predictions[:, 1].reshape(nbx, nby)

# Compute the true solution
cartesian_u_values = u(np.sqrt(x_grid ** 2 + y_grid ** 2), np.arctan2(y_grid, x_grid))
cartesian_u_values = np.nan_to_num(cartesian_u_values)

# Extract the real and imaginary parts of the true solution
real_u_values = np.real(cartesian_u_values)
imaginary_u_values = np.imag(cartesian_u_values)

# Calculate L2 relative error
l2_difference_Erz = dde.metrics.l2_relative_error(real_u_values, pred_Erz)
l2_difference_Eiz = dde.metrics.l2_relative_error(imaginary_u_values, pred_Eiz)

print("L2 relative error in Erz:", l2_difference_Erz)
print("L2 relative error in Eiz:", l2_difference_Eiz)

# Plot predictions and errors
fig, ax = plt.subplots(2, 3, figsize=(12, 12))

axp0 = ax[0, 0].pcolor(x_grid, y_grid, pred_Eiz, cmap='seismic', shading='auto')
cbar0 = fig.colorbar(axp0, ax=ax[0, 0], shrink=0.5)
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('y')
ax[0, 0].set_title('Eiz_pred')
ax[0, 0].set_aspect('equal')

axp1 = ax[0, 1].pcolor(x_grid, y_grid, imaginary_u_values, cmap='seismic', shading='auto')
cbar1 = fig.colorbar(axp1, ax=ax[0, 1], shrink=0.5)
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('y')
ax[0, 1].set_title('Eiz_exact')
ax[0, 1].set_aspect('equal')

axp2 = ax[0, 2].pcolor(x_grid, y_grid, np.abs(imaginary_u_values - pred_Eiz), cmap='seismic', shading='auto')
cbar2 = fig.colorbar(axp2, ax=ax[0, 2], shrink=0.5)
ax[0, 2].set_xlabel('x')
ax[0, 2].set_ylabel('y')
ax[0, 2].set_title('Eiz_error')
ax[0, 2].set_aspect('equal')

axp3 = ax[1, 0].pcolor(x_grid, y_grid, pred_Erz, cmap='seismic', shading='auto')
cbar3 = fig.colorbar(axp3, ax=ax[1, 0], shrink=0.5)
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('y')
ax[1, 0].set_title('Erz_pred')
ax[1, 0].set_aspect('equal')

axp4 = ax[1, 1].pcolor(x_grid, y_grid, real_u_values, cmap='seismic', shading='auto')
cbar4 = fig.colorbar(axp4, ax=ax[1, 1], shrink=0.5)
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('y')
ax[1, 1].set_title('Erz_exact')
ax[1, 1].set_aspect('equal')

axp5 = ax[1, 2].pcolor(x_grid, y_grid, np.abs(real_u_values - pred_Erz), cmap='seismic', shading='auto')
cbar5 = fig.colorbar(axp5, ax=ax[1, 2], shrink=0.5)
ax[1, 2].set_xlabel('x')
ax[1, 2].set_ylabel('y')
ax[1, 2].set_title('Erz_error')
ax[1, 2].set_aspect('equal')

plt.tight_layout()
plt.show()