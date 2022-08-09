from warp import * 
import numpy as np
import h5py

"""
(TEST WILL NOT WORK WITHOUT MODIFICATION NOW THAT MULTIGRIDDIELECTRIC2D 
HAS BEEN ADJUSTED TO AUTOMATICALLY DECOMPOSE EPSILON)


Test for new parallel 2D dielectric solver.
This is an initial test that creates two dielectric regions with the middle cell (where the decomposition boundary sits)
at an average value of the two.
The initial domain decomposition for epsilon is done manually and this test will only work on 2 processors.
"""

if comm_world.size != 1:
    synchronizeQueuedOutput_mpi4py(out=False, error=False)

print("rank:", comm_world.rank)

top.inject = 0 
top.npinject = 0

#Dimensions

PLATE_SPACING = 10e-6 #plate spacing
CHANNEL_WIDTH = 1e-6 #width of simulation box

X_MAX = CHANNEL_WIDTH*0.5
X_MIN = -1.*X_MAX
Y_MAX = CHANNEL_WIDTH*0.5
Y_MIN = -1.*Y_MAX
Z_MIN = 0.
Z_MAX = PLATE_SPACING


#Grid parameters
NUM_X = 7
NUM_Z = 7

top.dt = 1e-12


# # Solver Geometry

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

# Hack to run in in parallel with 2 processors
z_dependence = True
if comm_world.size == 2:
	testarray = np.ones([w3d.nx + 2, (w3d.nz + 1) //2 + 2]) * 8.854e-12
	if z_dependence:
		if comm_world.rank ==0:
			for i in range((w3d.nz+1) // 2 + 2):
				testarray[:,i] = 1. * 1.e-11 
			testarray[:,-3:-1] = 2. * 1.e-11 
		if comm_world.rank ==1:
			for i in range((w3d.nz+1) // 2 + 2):
				testarray[:,i] = 2. * 1.e-11 
	if not z_dependence:		
		testarray = np.ones([w3d.nx + 2, (w3d.nz+1) // 2 + 2]) * 8.854e-12
elif comm_world.size == 1:
	if z_dependence:
		for i in range(3):
			testarray[:,i] = 1. * 1.e-11
		for i in range(3,w3d.nz + 2):
			testarray[:,i] = 2. * 1.e-11
	if not z_dependence:		
		testarray = np.ones([w3d.nx + 2, w3d.nz + 2]) * 8.854e-12

print('shape:',testarray.shape)
top.depos_order = 1

solverE = MultiGrid2DDielectric(epsilon=testarray)
registersolver(solverE)


# Conductors

source = ZPlane(zcent=w3d.zmmin,zsign=-1.,voltage=0.)
installconductor(source, dfill=largepos)


plate = ZPlane(voltage=10., zcent=Z_MAX)
installconductor(plate,dfill=largepos)


# Generate PIC code and Run Simulation
solverE.mgmaxiters = 1

#prevent GIST from starting upon setup
top.lprntpara = false
top.lpsplots = false
top.verbosity = 0 

package("w3d")
generate()

step(150)

print(solverE.getselfe().shape)
print(getselfe('z').shape)
zfield = getselfe('z')
if comm_world.size > 1:
	if comm_world.rank == 0:
		np.save('diel_para.npy',zfield)
elif comm_world.size == 1:
	np.save('diel_ser.npy',zfield)
