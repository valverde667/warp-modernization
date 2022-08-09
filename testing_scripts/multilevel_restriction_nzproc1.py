from restrict2dcellcentered import Restrict2dCellCentered
import numpy as np

# CELL NUMBERS DO NOT INCLUDE GUARD CELL COUNTS
# BUT
# EPSILON/EPSILONCOARSE HAVE GUARD CELLS

np.set_printoptions(linewidth=160)
# Setup top level:
# On domain decomposition:
nx = 4
nz = 35 # floor(N/2) + N % 2
nxlocal = 4
nzlocal = 18

# Zero level
level_0_data = {}
level_0_data['nxcoarse'] = 2
level_0_data['nzcoarse'] = 18

level_0_data['nxlocalcoarse'] = 2
level_0_data['nzlocalcoarse'] = 10
level_0_data['localbounds'] = [2, 2, 1, 1, 0, -1]
level_0_data['localboundscoarse'] = [2, 2, 1, 1, 0, -1]
level_0_data['lxoffset'] = 0
level_0_data['lzoffset'] = 0


# First restriction level
# level_1_data = {}
# level_1_data['nxcoarse'] = (level_0_data['nxcoarse'] - 2) // 2 + 2
# level_1_data['nzcoarse'] = level_0_data['nzcoarse']
# level_1_data['nxlocalcoarse'] = level_1_data['nxcoarse']
# level_1_data['nzlocalcoarse'] = level_0_data['nzlocalcoarse']
# level_1_data['localbounds'] = [2, 2, 1, 1, 0, -1]
# level_1_data['localboundscoarse'] = [2, 2, 1, 1, 0, -1]
# level_1_data['lxoffset'] = 0
# level_1_data['lzoffset'] = 0


# epsilon and epsiloncoarse

epsilon_0 = np.ones([nxlocal + 2, nzlocal + 2])
for i in range(epsilon_0.shape[1]):
    epsilon_0[:, i] = (i + 1) * 1e-11

epsiloncoarse_0 = np.ones([level_0_data['nxlocalcoarse'] + 2, level_0_data['nzlocalcoarse'] + 2])

restriction = Restrict2dCellCentered(nx, nz, nxlocal, nzlocal,
                                     epsilon_0, level_0_data['nxcoarse'], level_0_data['nzcoarse'],
                                     level_0_data['nxlocalcoarse'], level_0_data['nzlocalcoarse'],
                                     epsiloncoarse_0, level_0_data['localbounds'], level_0_data['localboundscoarse'],
                                     level_0_data['lxoffset'], level_0_data['lzoffset'])

restriction.calculate_z_values()

restriction.calculate_x_values()
restriction.wxa[3, :] = 1.8999999999999996E-010
restriction.create_coarse_grid(restriction.wxa, restriction.wza)

print(restriction.wxa)
print(restriction.wza)
print(restriction.ucoarse)