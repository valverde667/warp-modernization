from __future__ import division
from warp import * 
import numpy as np
import h5py
import random

"""
Testing parallel 2D dielectric solver.
File has modes to generate:
	 1. A linearly varying (in z) dielectric
	 2. A random dielectric value in each cell. 
	 	Constrained between 1-2 * eps_0.
"""

if comm_world.size != 1:
    synchronizeQueuedOutput_mpi4py(out=False, error=False)

print "rank:", comm_world.rank

top.inject = 0 
top.npinject = 0

#Dimensions

PLATE_SPACING = 10e-6
CHANNEL_WIDTH = 1e-6

X_MAX = CHANNEL_WIDTH*0.5
X_MIN = -1.*X_MAX
Y_MAX = CHANNEL_WIDTH*0.5
Y_MIN = -1.*Y_MAX
Z_MIN = 0.
Z_MAX = PLATE_SPACING


# Grid parameters
NUM_X = 203
NUM_Z = 254

top.dt = 1e-15

w3d.solvergeom = w3d.XZgeom


# Set boundary conditions
w3d.bound0  = dirichlet
w3d.boundnz = dirichlet
w3d.boundxy = periodic 


# Set grid boundaries
w3d.xmmin = X_MIN
w3d.xmmax = X_MAX
w3d.zmmin = 0. 
w3d.zmmax = Z_MAX

w3d.nx = NUM_X
w3d.nz = NUM_Z

# Field Solver

testarray = np.ones([w3d.nx + 2, w3d.nz + 2]) * 8.854e-12
# testarray[:, 6+(w3d.nz + 2) // 2:] = 8.854e-12 * 3
# RUNMODES:
# linear: linear variation of epsilon in z
# random: random values of epsilon

eps_min = 1e-11
eps_max = 10e-11

eps_per_step = (eps_max - eps_min) / (w3d.nz + 1)

runmode = 'linear'

if runmode == 'linear':
	for i in range(w3d.nz + 2):
		testarray[:,i] = eps_per_step * i + eps_min
		# testarray[:,(w3d.nz + 2) //2 -2: (w3d.nz + 2) //2 +2] = (w3d.nz + 2) //2 *  eps_per_step  + eps_min
elif runmode == 'random':
	np.random.seed(56433)
	testarray = (np.random.random([w3d.nx + 2, w3d.nz + 2]) + 1 ) * 8.854e-12
else:
	pass

top.depos_order = 1

solverE = MultiGrid2DDielectric(epsilon=testarray)
registersolver(solverE)


# Conductors

source = ZPlane(zcent=w3d.zmmin,zsign=-1.,voltage=0.)
installconductor(source, dfill=largepos)


plate = ZPlane(voltage=10., zcent=Z_MAX)
installconductor(plate,dfill=largepos)


# Generate PIC code and Run Simulation

# Restric to one v-cycle per step (doesn't really matter in this case with no particles)
solverE.mgmaxiters = 1

# Set SOR coefficient for faster convergence
omega = 2./(1. + np.sin(np.pi/min(NUM_X+1,NUM_Z+1)))
solverE.mgparam = omega

#prevent GIST from starting upon setup
top.lprntpara = false
top.lpsplots = false
top.verbosity = 0 

package("w3d")
generate()

step(500)

zfield = getselfe('z')
print comm_world.rank,solverE.zmminlocal,solverE.zmmaxlocal,solverE.nzlocal,(solverE.zmmaxlocal - solverE.zmminlocal) /solverE.nzlocal
print comm_world.rank,solverE.getphi().shape
if comm_world.size > 1:
	if comm_world.rank == 0:
		np.save('phi_0_0.npy',solverE.getphi())
	if comm_world.rank == 1:
		np.save('phi_0_1.npy',solverE.getphi())
	if comm_world.rank == 0:
		np.save('diel_para.npy',zfield)
elif comm_world.size == 1:
	np.save('diel_ser.npy',zfield)
	np.save('phi_serial.npy',solverE.getphi())