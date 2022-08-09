from restrict2dcellcentered import Restrict2dCellCentered
import numpy as np

# CELL NUMBERS DO NOT INCLUDE GUARD CELL COUNTS
# BUT
# EPSILON/EPSILONCOARSE HAVE GUARD CELLS

np.set_printoptions(linewidth=160)
# Setup top level:
top_level_cells = 35
# On domain decomposition:
nx = top_level_cells
nz = top_level_cells # floor(N/2) + N % 2
nxlocal = nx
nzlocal = top_level_cells // 2 + top_level_cells % 2

# Zero level
level_0_data = {}
level_0_data['nxcoarse'] = top_level_cells // 2 + top_level_cells % 2
level_0_data['nzcoarse'] = nz

level_0_data['nxlocalcoarse'] = top_level_cells // 2 + top_level_cells % 2
level_0_data['nzlocalcoarse'] = top_level_cells // 2 + top_level_cells % 2
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
print(restriction.wza.shape)
restriction.calculate_x_values()
print(restriction.wxa.shape)
restriction.create_coarse_grid(restriction.calculate_x_values(), restriction.calculate_z_values())
print(restriction.ucoarse.shape)
print(restriction.ucoarse)