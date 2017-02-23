import numpy as np


class Restrict2dCellCentered(object):
    """
    Class that emulates Fortran subroutine restrict2dcellcentered from f3d_mgrid.F
    """

    def __init__(self, nx, nz, nxlocal, nzlocal,
                 u, nxcoarse, nzcoarse,
                 nxlocalcoarse, nzlocalcoarse, ucoarse,
                 localbounds, localboundscoarse,
                 lxoffset, lzoffset):
        self.nx = nx
        self.nz = nz
        self.nxlocal = nxlocal
        self.nzlocal = nzlocal
        self.u = u
        self.nxcoarse = nxcoarse
        self.nzcoarse = nzcoarse
        self.nxlocalcoarse = nxlocalcoarse
        self.nzlocalcoarse = nzlocalcoarse
        self.ucoarse = ucoarse
        self.localbounds = localbounds
        self.localboundscoarse = localboundscoarse
        self.lxoffset = lxoffset
        self.lzoffset = lzoffset

        self.dx = 1. * nx / nxcoarse
        self.dz = 1. * nz / nzcoarse
        self.dxi = 1. * nxcoarse / nx
        self.dzi = 1. * nzcoarse / nz


    def calculate_z_values(self):
        wz = np.zeros([4, ])
        wza = np.zeros([4, self.nzlocalcoarse])
        self.izmina = []
        self.izmaxa = []

        for izcoarse in range(0, self.nzlocalcoarse):
            izmin = ((izcoarse - 1 + 1) * self.nz - self.lzoffset + 4 * self.nzcoarse) / self.nzcoarse - 3
            izmax = ((izcoarse + 1 + 1) * self.nz - self.lzoffset - 1) / self.nzcoarse

            if izmin < 0:
                izmin = 0
            if izmax > self.nzlocal + 1:
                izmax = self.nzlocal + 1
            if izmax < izmin:
                pass

            # print izmin, izmax
            for iz in range(izmin, izmax+1):
                wz[iz - izmin] = 1. - abs(izcoarse + 1 - (iz + 1. * self.lzoffset / self.nzcoarse) * self.dzi)

            self.izmina.append(izmin)
            self.izmaxa.append(izmax)
            wza[:, izcoarse] = wz

        return wza

    def calculate_x_values(self):
        wx = np.zeros([4, ])
        self.wxa = np.zeros([4, self.nxlocalcoarse])
        self.ixmina = []
        self.ixmaxa = []

        for ixcoarse in range(0, self.nxlocalcoarse):
            ixmin = ((ixcoarse - 1 + 1) * self.nx - self.lxoffset + 4 * self.nxcoarse) / self.nxcoarse - 3
            ixmax = ((ixcoarse + 1 + 1) * self.nx - self.lxoffset - 1) / self.nxcoarse

            if ixmin < 0:
                ixmin = 0
            if ixmax > self.nxlocal + 1:
                ixmax = self.nxlocal + 1
            if ixmax < ixmin:
                pass

            # print ixmin, ixmax
            for ix in range(ixmin, ixmax + 1):
                wx[ix - ixmin] = 1. - abs(ixcoarse + 1 - (ix + 1. * self.lxoffset / self.nxcoarse) * self.dxi)

            self.ixmina.append(ixmin)
            self.ixmaxa.append(ixmax)
            self.wxa[:, ixcoarse] = wx

        return self.wxa

    def create_coarse_grid(self, wxa, wza):
        print len(self.izmina)
        print len(self.izmaxa)
        for izcoarse in range(1, self.nzlocalcoarse + 1):
            print izcoarse
            izmin = self.izmina[izcoarse - 1]
            izmax = self.izmaxa[izcoarse - 1]
            wz = wza[:, izcoarse - 1]

            for ixcoarse in range(1, self.nxlocalcoarse + 1):
                ixmin = self.ixmina[ixcoarse - 1]
                ixmax = self.ixmaxa[ixcoarse - 1]
                wx = wxa[:, ixcoarse - 1]

                r = 0.
                w = 0.

                for iz in range(izmin, izmax + 1):
                    for ix in range(ixmin, ixmax + 1):
                        r = r + wx[ix - ixmin] * wz[iz - izmin] * self.u[ix, iz]
                        w = w + wx[ix - ixmin] * wz[iz - izmin]

                if w > 0.:
                    self.ucoarse[ixcoarse, izcoarse] = r / w
                else:
                    self.ucoarse[ixcoarse, izcoarse] = 0.

        return self.ucoarse


if __name__ == "__main__":
    np.set_printoptions(linewidth=160)
    # Use a grid defined in Warp with
    # nx = 7
    # nz = 7

    # Serial values
    nx = nz = 7
    nxlocal = nzlocal = 7
    # First restriction level
    nxcoarse = 4
    nzcoarse = 7
    nxlocalcoarse = 4
    nzlocalcoarse = 7
    localbounds = [2, 2, 1, 1, 0, 0]
    localboundscoarse = [2, 2, 1, 1, 0, 0]
    lxoffset = 0
    lzoffset = 0

    # epsilon and epsiloncoarse
    epsilon = np.ones([nx + 2, nz + 2]) * 8.857e-12
    epsiloncoarse = np.ones([nxcoarse + 2, nzcoarse + 2])

    restriction = Restrict2dCellCentered(nx, nz, nxlocal, nzlocal,
                                         epsilon, nxcoarse, nzcoarse,
                                         nxlocalcoarse, nzlocalcoarse,
                                         epsiloncoarse, localbounds, localboundscoarse,
                                         lxoffset, lzoffset)

    print restriction.calculate_z_values()

    print restriction.calculate_x_values()

    print restriction.create_coarse_grid(restriction.calculate_x_values(), restriction.calculate_z_values())



