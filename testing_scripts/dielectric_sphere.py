# # Test script simulating a 3D dielectric sphere between two parallel plates. This script should run in parallel.
# 
# If MAKE_PLOTS is set, then plots a 2D slice of the potential and compares a 1D lineout to the analytic solution.
#
#
# 9/07/2018
#  
# Nathan Cook

from __future__ import division
import sys
import os

import matplotlib
matplotlib.use('Agg')

from warp import * 
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from warp.data_dumping.openpmd_diag import ElectrostaticFieldDiagnostic

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5


# Useful utility function - See the rswarp repository for further details:
# https://github.com/radiasoft/rswarp/blob/master/rswarp/utilities/file_utils.py
def cleanupPrevious(particleDirectory, fieldDirectory):

    """
    Remove old diagnostic files.

    Parameters:
            particleDirectory (str): Path to particle diagnostics

    """
    if os.path.exists(particleDirectory):
        files = os.listdir(particleDirectory)
        for file in files:
            if file.endswith('.h5'):
                os.remove(os.path.join(particleDirectory,file))
    if isinstance(fieldDirectory,dict):
        for key in fieldDirectory:
            if os.path.exists(fieldDirectory[key]):
                files = os.listdir(fieldDirectory[key])
                for file in files:
                    if file.endswith('.h5'):
                        os.remove(os.path.join(fieldDirectory[key],file))
    elif isinstance(fieldDirectory, list):
        for directory in fieldDirectory:
            if os.path.exists(directory):
                files = os.listdir(directory)
                for file in files:
                    if file.endswith('.h5'):
                        os.remove(os.path.join(directory, file))
    elif isinstance(fieldDirectory, str):
            if os.path.exists(fieldDirectory):
                files = os.listdir(fieldDirectory)
                for file in files:
                    if file.endswith('.h5'):
                        os.remove(os.path.join(fieldDirectory, file))


# Constants imports
from scipy.constants import e, m_e, c, k
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e

USE_3D = True #If true, use 3D solver
MAKE_PLOTS = True #If true, plots a 2D slice of the potential and compares a 1D lineout to the analytic solution

diagDir = 'diags/xzsolver/hdf5/'
field_base_path = 'diags/fields/'
diagFDir = {'magnetic':'diags/fields/magnetic','electric':'diags/fields/electric'}

# Cleanup previous files
if comm_world.rank == 0:
    cleanupPrevious(diagDir,diagFDir)

if comm_world.size != 1:
    synchronizeQueuedOutput_mpi4py(out=False, error=False)

#print "rank:", comm_world.rank

top.inject = 0 
top.npinject = 0

#Dimensions

PLATE_SPACING = 1.e-6 #plate spacing
CHANNEL_WIDTH = 1.e-6 #width of simulation box

X_MAX = CHANNEL_WIDTH*0.5
X_MIN = -1.*X_MAX
Y_MAX = CHANNEL_WIDTH*0.5
Y_MIN = -1.*Y_MAX
Z_MIN = 0.
Z_MAX = PLATE_SPACING


#Grid parameters - increase number of cells if running in parallel
if comm_world.size > 1:
    N_ALL = 96
else:
    N_ALL = 64 

NUM_X = N_ALL
NUM_Y = N_ALL
NUM_Z = N_ALL


## Solver Geometry

# Set boundary conditions
w3d.bound0  = neumann
w3d.boundnz = dirichlet
w3d.boundxy = periodic 

# Set mesh boundaries
w3d.xmmin = X_MIN
w3d.xmmax = X_MAX
w3d.zmmin = 0.
w3d.zmmax = Z_MAX

# Set mesh cell counts
w3d.nx = NUM_X
w3d.nz = NUM_Z

w3d.dx = (w3d.xmmax-w3d.xmmin)/w3d.nx
w3d.dz = (w3d.zmmax-w3d.zmmin)/w3d.nz

if USE_3D:
    w3d.solvergeom = w3d.XYZgeom
    w3d.ymmin = Y_MIN
    w3d.ymmax = Y_MAX
    w3d.ny = NUM_Y
    w3d.dy = (w3d.ymmax-w3d.ymmin)/w3d.ny
else:
    w3d.solvergeom = w3d.XZgeom

zmesh = np.linspace(0,Z_MAX,NUM_Z+1) #holds the z-axis grid points in an array
xmesh = np.linspace(X_MIN,X_MAX,NUM_X+1)

ANODE_VOLTAGE = 10.
CATHODE_VOLTAGE = 0.
vacuum_level = ANODE_VOLTAGE - CATHODE_VOLTAGE
beam_beta = 5e-4
#Determine an appropriate time step based upon estimated final velocity
vzfinal = sqrt(2.*abs(vacuum_level)*np.abs(e)/m_e)+beam_beta*c
dt = w3d.dz/vzfinal
top.dt = 0.5*dt

if vzfinal*top.dt > w3d.dz:
    print "Time step dt = {:.3e}s does not constrain motion to a single cell".format(top.dt)

top.depos_order = 1
f3d.mgtol = 1e-6 # Multigrid solver convergence tolerance, in volts. 1 uV is default in Warp.

if USE_3D:
    solverE = MultiGrid3DDielectric()
else:
    solverE = MultiGrid2DDielectric()
    
registersolver(solverE)

#Define conductor/dielectrics

source = ZPlane(voltage=CATHODE_VOLTAGE, zcent=w3d.zmmin+0*w3d.dz,zsign=-1.)
solverE.installconductor(source, dfill=largepos)

plate = ZPlane(voltage=ANODE_VOLTAGE, zcent=Z_MAX-0.*w3d.dz)
solverE.installconductor(plate,dfill=largepos)

#Define dielectric sphere
r_sphere = Z_MAX/8.
Z0 = 0.5e-6
epsn = 7.5 #dielectric constant for sphere

sphere = Sphere(radius=r_sphere, xcent=0.0, ycent=0.0, zcent=Z0, permittivity=epsn)
solverE.installconductor(sphere,dfill=largepos)


#Define diagnostics
particleperiod = 1
particle_diagnostic_0 = ParticleDiagnostic(period = particleperiod, top = top, w3d = w3d,
                                          species = {species.name: species for species in listofallspecies},
                                          comm_world=comm_world, lparallel_output=False, write_dir = diagDir[:-5])
fieldperiod = 1
efield_diagnostic_0 = ElectrostaticFieldDiagnostic(solver=solverE, top=top, w3d=w3d, comm_world = comm_world,
                                      period=fieldperiod, write_dir = diagFDir['electric'])

installafterstep(particle_diagnostic_0.write)
installafterstep(efield_diagnostic_0.write)

# Generate PIC code and Run Simulation
solverE.mgmaxiters = 1

#prevent GIST from starting upon setup
top.lprntpara = false
top.lpsplots = false
top.verbosity = 0 

solverE.mgmaxiters = 16000 #rough approximation of steps needed for generate() to converge
package("w3d")
generate()
solverE.mgmaxiters = 100
step(1)
    

################
#NOW PLOT STUFF
################

if (comm_world.rank == 0 and MAKE_PLOTS):
    
    #load file
    f0 = "{}{}".format(diagFDir['electric'],'/data00000001.h5')
    dat = h5.File(f0) 
    phi = dat['data/1/fields/phi']

    #Now plot phi
    fig = plt.figure(figsize=(12,6))

    X_CELLS = NUM_X
    Z_CELLS = NUM_Z

    xl = 0
    xu = NUM_X
    zl = 0 
    zu = NUM_Z 

    plt.xlabel("z ($\mu$m)")
    plt.ylabel("x ($\mu$m)")
    plt.title(r"$\phi$ across domain - with sphere - {}x{}x{}".format(NUM_X,NUM_Y,NUM_Z))

    pxmin = ((X_MAX - X_MIN) / X_CELLS * xl + X_MIN) * 1e6
    pxmax = ((X_MAX - X_MIN) / X_CELLS * xu + X_MIN) * 1e6
    pzmin = (Z_MIN + zl / Z_CELLS * Z_MAX) * 1e6
    pzmax = (Z_MAX * zu / Z_CELLS) * 1e6

    plt.xlim(pzmin, pzmax)
    plt.ylim(pxmin, pxmax)

    x_mid = int(NUM_X/2)
    y_mid = int(NUM_Y/2)

    eps_plt = plt.imshow(phi[xl:xu,y_mid,zl:zu],cmap='viridis',extent=[pzmin, pzmax, pxmin, pxmax],aspect='auto')

    cbar = fig.colorbar(eps_plt)
    cbar.ax.set_xlabel(r"$\phi [V]$")
    cbar.ax.xaxis.set_label_position('top')

    fig.savefig('phi_{}x{}x{}.png'.format(NUM_X,NUM_Y,NUM_Z),bbox_inches='tight')
    
    
    # Calculate Theoretical Potential
    d = 1e-6
    a = d/8.
    phi_a = 10. # 10 V

    e_r = 7.5

    [Z,X] = np.meshgrid(zmesh+1e-12,xmesh+1e-12)

    Z0 = 0.5e-6

    R = np.sqrt((Z-Z0)**2 + X**2)

    inside = (R <= a)

    phicalc = phi_a *(Z-Z0) /d *(1.-(e_r-1.)/(e_r+2.)*(a/R)**3.)
    phicalc[inside] = 3./(e_r+2.)*phi_a *(Z[inside]-Z0)/d


    #comparison
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    ax.plot(zmesh*1e6,phi[x_mid,y_mid,:], label = 'Simulated')
    plt.plot(zmesh*1e6,phicalc[x_mid,:]+5., label = 'Analytical')
    ax.set_title('Potential through dielectric sphere')
    ax.set_xlabel('z [$\mu$m]')
    ax.set_ylabel('$\phi$ [V]')
    plt.legend()
    fig.savefig('phi_analytic-{}x{}x{}.png'.format(NUM_X,NUM_Y,NUM_Z),bbox_inches='tight')