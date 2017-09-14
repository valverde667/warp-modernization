"""Classes following the PICMI standard
"""
import PICMI_Base
from warp import *
import warp

codename = 'Warp'

# --- Always assume that relativity should be turned on.
warp.top.lrelativ = warp.true

class Grid(PICMI_Base.PICMI_Grid):

    def init(self, **kw):
        w3d.nx = self.nx
        w3d.ny = self.ny
        w3d.nz = self.nz
        w3d.xmmin = self.xmin
        w3d.xmmax = self.xmax
        w3d.ymmin = self.ymin
        w3d.ymmax = self.ymax
        w3d.zmmin = self.zmin
        w3d.zmmax = self.zmax

        defaultsdict = {'dirichlet':dirichlet,
                        'neumann':neumann,
                        'periodic':periodic,
                        'open':openbc}
        self.bounds = [defaultsdict[self.bcxmin], defaultsdict[self.bcxmax],
                       defaultsdict[self.bcymin], defaultsdict[self.bcymax],
                       defaultsdict[self.bczmin], defaultsdict[self.bczmax]]
        w3d.boundxy = self.bounds[1]
        w3d.bound0 = self.bounds[4]
        w3d.boundnz = self.bounds[5]
        top.pboundxy = self.bounds[1]
        top.pbound0 = self.bounds[4]
        top.pboundnz = self.bounds[5]
        if top.pboundxy == openbc: top.pboundxy = absorb
        if top.pbound0 == openbc: top.pbound0 = absorb
        if top.pboundnz == openbc: top.pboundnz = absorb

        if self.moving_window_velocity is not None:
            top.vbeam = top.vbeamfrm = self.moving_window_velocity[2]
            top.lgridqnt = true

    def getdims(self, **kw):
        return array([w3d.dx, w3d.dy, w3d.dz])

    def getmins(self, **kw):
        return array([w3d.xmmin, w3d.ymmin, w3d.zmmin + top.zgrid])

    def getmaxs(self, **kw):
        return array([w3d.xmmax, w3d.ymmax, w3d.zmmax + top.zgrid])


class EM_solver(PICMI_Base.PICMI_EM_solver):

    def init(self, **kw):

        if self.laser is not None:
            laser_func = self.laser.laser
        else:
            laser_func = None

        self.solver = EM3D(laser_func=laser_func)
        registersolver(self.solver)


class Gaussian_laser(PICMI_Base.PICMI_Gaussian_laser):
    def init(self, **kw):
        dim = '3d'
        if self.em_solver is None:
            from .field_solvers.laser.laser_profiles import GaussianProfile
            self.laser = GaussianProfile(self.k0, self.waist, self.duration, self.t_peak, self.a0, dim,
                                         focal_length=-self.focal_length, temporal_order=2, boost=None, source_v=0)
        else:
            from .init_tools import add_laser
            add_laser(self.em_solver.solver, dim, self.a0, self.waist, self.duration*warp.clight, self.z0, self.focal_position, lambda0=self.wavelength,
                      theta_pol=self.pol_angle, source_z=self.antenna_z0, zeta=0, beta=0, phi2=0, gamma_boost=None, laser_file=None, laser_file_energy=None )


class Species(warp.Species):

    def __init__(self, **kw):

        # --- If weight is specified, then use that single value for all particles.
        # --- Otherwise setup variable weights. In that case, the species weight, sw,
        # --- will be set to 1, assuming that each particles carries its own weight.
        if 'weight' not in kw:
            kw['lvariableweights'] = True

        warp.Species.__init__(self, **kw)

        weight = kw.get('weight', 1.)
        for pg in self.iterpgroups():
            pg.sw = weight

    def add_particles(self, n=None,
                      x=None, y=None, z=None,
                      ux=None, uy=None, uz=None, w=None,
                      unique_particles=None, **kw):
        return self.addparticles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w, unique_particles=unique_particles)


class Simulation(PICMI_Base.PICMI_Simulation):
    def init(self, **kw):
        if self.verbose is not None:
            top.verbosity = self.verbose + 1

        if not hasattr(setup, 'pname'):
            # --- Setup the graphics if needed.
            setup()

        package('w3d')
        generate()

        installafterstep(self.makedumps)

    def makedumps(self):
        if self.plot_int is not None and self.plot_int > 0:
            if top.it%self.plot_int == 0:
                dump()

    def step(self, nsteps=1):
        step(nsteps)

    def finalize(self):
        uninstallafterstep(self.makedumps)


