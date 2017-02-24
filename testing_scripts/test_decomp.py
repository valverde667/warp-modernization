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
With width = 1um and length = 10 um cutoff to parallel/serial divergence starts at 31 cells in z
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
NUM_X = 35
NUM_Z = 35

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

# RUNMODES:
# linear: linear variation of epsilon in z
# random: random values of epsilon

runmode = 'linear'

if runmode == 'linear':
	for i in range(w3d.nz + 2):
		testarray[:,i] = (i+1) * 1.e-11
elif runmode == 'random':
	np.random.seed(56433)
	testarray = (np.random.random([w3d.nx + 2, w3d.nz + 2]) + 1 ) * 8.854e-12
else:
	"NOT A RUNMODE"
	sys.exit()

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

#prevent GIST from starting upon setup
top.lprntpara = false
top.lpsplots = false
top.verbosity = 0 

package("w3d")
generate()

step(1)

zfield = getselfe('z')

if comm_world.size > 1:
	if comm_world.rank == 0:
		np.save('diel_para.npy',zfield)
elif comm_world.size == 1:
	np.save('diel_ser.npy',zfield)
