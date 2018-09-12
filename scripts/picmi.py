"""Classes following the PICMI standard
"""
import re
import picmistandard
import numpy as np
from .warp import *
from .init_tools.plasma_initialization import PlasmaInjector
import warp

codename = 'warp'
picmistandard.register_codename(codename)

c = warp.clight
ep0 = warp.eps0
mu0 = warp.mu0
q_e = warp.echarge
m_e = warp.Electron.mass
m_p = warp.Proton.mass

# --- Always assume that relativity should be turned on.
warp.top.lrelativ = warp.true

# --- Species name dictionary.
# --- This maps the PICMI type (a string) to the Warp species types
# --- This will go in a separate file, since it will be big.
species_type_dict = {'electron':warp.Electron,
                     'positron':warp.Positron,
                     'proton':warp.Proton}
# --- Add all of the elements to the dictionary
for name,d in warp.periodic_table.items():
    symbol = d['Symbol']
    m = re.search('[0-9]', symbol)
    if m is not None:
        symbol = '#' + m.group() + symbol
    species_type_dict[symbol] = getattr(warp, name)


class Species(picmistandard.PICMI_Species):
    def init(self, kw):

        # --- When using the PICMI standard, always make the particles
        # --- have variable weights. This will greatly simplify the
        # --- handling of particle initialization. The species weight
        # --- will be set to 1, and the variable weight will hold
        # --- the actual weight.

        wtype = species_type_dict.get(self.particle_type, None)
        self.wspecies = warp.Species(type = wtype,
                                     name = self.name,
                                     charge = self.charge,
                                     charge_state = self.charge_state,
                                     mass = self.mass,
                                     lvariableweights = True,
                                     weight = 1.)

    def initialize_inputs(self, layout):

        if self.particle_shape is not None:
            if isinstance(self.particle_shape, str):
                interpolation_order = {'NGP':0, 'linear':1, 'quadratic':2, 'cubic':3}[self.particle_shape]
            else:
                interpolation_order = self.particle_shape
            self.wspecies.depos_order = interpolation_order

        if self.initial_distribution is not None:
            self.layout = layout
            installparticleloader(self.particle_loader)

    def particle_loader(self):
        self.initial_distribution.loaddistribution(self.wspecies, self.layout)


picmistandard.PICMI_MultiSpecies.Species_class = Species
class MultiSpecies(picmistandard.PICMI_MultiSpecies):
    pass


class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
    def loaddistribution(self, species, layout):
        assert isinstance(layout, PseudoRandomLayout), Exception('Warp only supports PseudoRandomLayout with GaussianBunchDistribution')
        assert layout.n_macroparticles is not None, Exception('Warp only support n_macroparticles with PseudoRandomLayout with GaussianBunchDistribution')
        np = layout.n_macroparticles
        deltax = self.rms_bunch_size[0]
        deltay = self.rms_bunch_size[1]
        deltaz = self.rms_bunch_size[2]
        vthx = self.rms_velocity[0]
        vthy = self.rms_velocity[1]
        vthz = self.rms_velocity[2]
        xmean = self.centroid_position[0]
        ymean = self.centroid_position[1]
        zmean = self.centroid_position[2]
        vxmean = self.centroid_velocity[0]
        vymean = self.centroid_velocity[1]
        vzmean = self.centroid_velocity[2]
        vxdiv = self.velocity_divergence[0]
        vydiv = self.velocity_divergence[1]
        vzdiv = self.velocity_divergence[2]
        w = self.n_physical_particles/np
        species.add_gaussian_dist(np, deltax, deltay, deltaz, vthx, vthy, vthz,
                                  xmean, ymean, zmean, vxmean, vymean, vzmean, vxdiv, vydiv, vzdiv,
                                  zdist='random', rdist='linear', fourfold=False, lmomentum=True, w=w)


class UniformDistribution(picmistandard.PICMI_UniformDistribution):
    def loaddistribution(self, species, layout):
        xmin = self.lower_bound[0]
        if xmin is None: xmin = w3d.xmmin
        ymin = self.lower_bound[1]
        if ymin is None: ymin = w3d.ymmin
        zmin = self.lower_bound[2]
        if zmin is None: zmin = w3d.zmmin
        xmax = self.upper_bound[0]
        if xmax is None: xmax = w3d.xmmax
        ymax = self.upper_bound[1]
        if ymax is None: ymax = w3d.ymmax
        zmax = self.upper_bound[2]
        if zmax is None: zmax = w3d.zmmax
        ux_th = self.rms_velocity[0]
        uy_th = self.rms_velocity[1]
        uz_th = self.rms_velocity[2]
        ux_m = self.directed_velocity[0]
        uy_m = self.directed_velocity[1]
        uz_m = self.directed_velocity[2]
        if isinstance(layout, GriddedLayout):
            # --- Note that layout.grid is ignored
            p_nx = layout.n_macroparticle_per_cell[0]
            p_ny = layout.n_macroparticle_per_cell[1]
            p_nz = layout.n_macroparticle_per_cell[2]

            npreal_per_cell = self.density*w3d.dx*w3d.dy*w3d.dz
            w = npreal_per_cell/(p_nx*p_ny*p_nz)
            def dens_func(x, y, z, w=w):
                return w

            if top.vbeamfrm > 0:
                injection_direction = +1
            else:
                injection_direction = -1
            plasmainjector = PlasmaInjector(elec=species, ions=None, w3d=w3d, top=top, dim='3d',
                                            p_nx=p_nx, p_ny=p_ny, p_nz=p_nz,
                                            p_xmin=xmin, p_ymin=xmin, p_zmin=zmin,
                                            p_xmax=xmax, p_ymax=ymax, p_zmax=zmax,
                                            dens_func=dens_func,
                                            ux_m=ux_m, uy_m=uy_m, uz_m=uz_m,
                                            ux_th=ux_th, uy_th=uy_th, uz_th=uz_th,
                                            injection_direction=injection_direction)
            if self.fill_in:
                installuserinjection(plasmainjector.continuous_injection)

        elif isinstance(layout, PseudoRandomLayout):
            assert not self.fill_in, Exception('Warp does not support fill_in with PseudoRandomLayout')
            if layout.n_macroparticles_per_cell is not None:
                np = layout.n_macroparticles_per_cell*w3d.nx*w3d.ny*w3d.nz
            elif layout.n_macroparticles is not None:
                np = layout.n_macroparticles
            npreal = self.density*(xmax - xmin)*(ymax - ymin)*(zmax - zmin)
            w = np.full(np, npreal/np)
            species.add_uniform_box(np=np, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                                    vthx=ux_th, vthy=uy_th, vthz=uz_th,
                                    vxmean=ux_m, vymean=uy_m, vzmean=uz_m,
                                    lmomentum=1, spacing='random',
                                    lallindomain=warp.true,
                                    w=w)


class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
    def loaddistribution(self, species, layout):
        xmin = self.lower_bound[0]
        if xmin is None: xmin = w3d.xmmin
        ymin = self.lower_bound[1]
        if ymin is None: ymin = w3d.ymmin
        zmin = self.lower_bound[2]
        if zmin is None: zmin = w3d.zmmin
        xmax = self.upper_bound[0]
        if xmax is None: xmax = w3d.xmmax
        ymax = self.upper_bound[1]
        if ymax is None: ymax = w3d.ymmax
        zmax = self.upper_bound[2]
        if zmax is None: zmax = w3d.zmmax
        ux_th = self.rms_velocity[0]
        uy_th = self.rms_velocity[1]
        uz_th = self.rms_velocity[2]
        ux_m = self.directed_velocity[0]
        uy_m = self.directed_velocity[1]
        uz_m = self.directed_velocity[2]
        if isinstance(layout, GriddedLayout):
            # --- Note that layout.grid is ignored
            p_nx = layout.n_macroparticle_per_cell[0]
            p_ny = layout.n_macroparticle_per_cell[1]
            p_nz = layout.n_macroparticle_per_cell[2]

            if isinstance(self.density_expression, str):
                cell_volume_per_particle = w3d.dx*w3d.dy*w3d.dz/(p_nx*p_ny*p_nz)
                def dens_func(x, y, z, density=self.density_expression, cell_volume_per_particle=cell_volume_per_particle):
                    # --- Include globals so that numpy is available
                    d = eval(density, locals(), globals())
                    return d*cell_volume_per_particle
            else:
                npreal_per_cell = self.density_expression*w3d.dx*w3d.dy*w3d.dz
                w = npreal_per_cell/(p_nx*p_ny*p_nz)
                def dens_func(x, y, z, w=w):
                    return w

            if top.vbeamfrm > 0:
                injection_direction = +1
            else:
                injection_direction = -1
            plasmainjector = PlasmaInjector(elec=species, ions=None, w3d=w3d, top=top, dim='3d',
                                            p_nx=p_nx, p_ny=p_ny, p_nz=p_nz,
                                            p_xmin=xmin, p_ymin=xmax, p_zmin=zmin,
                                            p_xmax=xmax, p_ymax=ymax, p_zmax=zmax,
                                            dens_func=dens_func,
                                            ux_m=ux_m, uy_m=uy_m, uz_m=uz_m,
                                            ux_th=ux_th, uy_th=uy_th, uz_th=uz_th,
                                            injection_direction=injection_direction)
            if self.fill_in:
                installuserinjection(plasmainjector.continuous_injection)

        elif isinstance(layout, PseudoRandomLayout):
            assert not self.fill_in, Exception('Warp does not support fill_in with PseudoRandomLayout')
            assert not isinstance(self.density_expression, str), Exception('Warp does not support PseudoRandomLayout with nonuniform distribution')
            if layout.n_macroparticles_per_cell is not None:
                np = layout.n_macroparticles_per_cell*w3d.nx*w3d.ny*w3d.nz
            elif layout.n_macroparticles is not None:
                np = layout.n_macroparticles
            npreal = self.density_expression*(xmax - xmin)*(ymax - ymin)*(zmax - zmin)
            w = np.full(np, npreal/np)
            species.add_uniform_box(np=np, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                                    vthx=ux_th, vthy=uy_th, vthz=uz_th,
                                    vxmean=ux_m, vymean=uy_m, vzmean=uz_m,
                                    lmomentum=1, spacing='random',
                                    lallindomain=warp.true,
                                    w=w)


class ParticleListDistribution(picmistandard.PICMI_ParticleListDistribution):
    def loaddistribution(self, species, layout):
        species.addparticles(x=self.x, y=self.y, z=self.z, ux=self.ux, uy=self.uy, uz=self.uz,
                             vx=None, vy=None, vz=None)


class ParticleDistributionPlanarInjector(picmistandard.PICMI_ParticleDistributionPlanarInjector):
    pass


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
    pass


class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
    pass


class BinomialSmoother(picmistandard.PICMI_BinomialSmoother):
    pass


class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
    def init(self, kw):
        raise Exception('WarpX does not support CylindricalGrid yet')


class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
    def init(self, kw):
        raise Exception('WarpX does not support Cartesian2DGrid yet')


class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
    def init(self, kw):
        w3d.nx = self.nx
        w3d.ny = self.ny
        w3d.nz = self.nz
        w3d.xmmin = self.xmin
        w3d.xmmax = self.xmax
        w3d.ymmin = self.ymin
        w3d.ymmax = self.ymax
        w3d.zmmin = self.zmin
        w3d.zmmax = self.zmax

        bc_dict = {'dirichlet':warp.dirichlet,
                        'neumann':warp.neumann,
                        'periodic':warp.periodic,
                        'open':warp.openbc}
        self.bounds = [bc_dict[self.lower_boundary_conditions[0]], bc_dict[self.upper_boundary_conditions[0]],
                       bc_dict[self.lower_boundary_conditions[1]], bc_dict[self.upper_boundary_conditions[1]],
                       bc_dict[self.lower_boundary_conditions[2]], bc_dict[self.upper_boundary_conditions[2]]]
        w3d.boundxy = self.bounds[1]
        w3d.bound0 = self.bounds[4]
        w3d.boundnz = self.bounds[5]
        top.pboundxy = self.bounds[1]
        top.pbound0 = self.bounds[4]
        top.pboundnz = self.bounds[5]
        if top.pboundxy == warp.openbc: top.pboundxy = warp.absorb
        if top.pbound0 == warp.openbc: top.pbound0 = warp.absorb
        if top.pboundnz == warp.openbc: top.pboundnz = warp.absorb

        if self.moving_window_velocity is not None:
            top.vbeam = top.vbeamfrm = self.moving_window_velocity[2]
            top.lgridqnt = true


class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
    def init(self, kw):
        if self.method is not None:
            stencil = {'Yee':0, 'CKC':1}[self.method]
        else:
            stencil = 0
        self.solver = EM3D(stencil=stencil)
        registersolver(self.solver)


class ElectrostaticSolver(picmistandard.PICMI_ElectrostaticSolver):
    def init(self, kw):
        self.solver = MultiGrid3d()
        registersolver(self.solver)


class GaussianLaser(picmistandard.PICMI_GaussianLaser):
    def initialize_inputs(self, solver, antenna):
        from .init_tools import add_laser
        dim = '3d'
        if self.zeta is None: self.zeta = 0.
        if self.beta is None: self.beta = 0.
        if self.phi2 is None: self.phi2 = 0.
        antenna_z0 = antenna.position[2]
        add_laser(solver.solver, dim, self.a0, self.waist, self.duration*warp.clight,
                  self.centroid_position[2], self.focal_position[2],
                  lambda0=self.wavelength, theta_pol=self.polarization_angle, source_z=antenna_z0,
                  zeta=self.zeta, beta=self.beta, phi2=self.phi2, 
                  gamma_boost=None, laser_file=None, laser_file_energy=None)


class LaserAntenna(picmistandard.PICMI_LaserAntenna):
    pass


class Simulation(picmistandard.PICMI_Simulation):
    def init(self, kw):
        if self.verbose is not None:
            top.verbosity = self.verbose + 1

        if not hasattr(setup, 'pname'):
            # --- Setup the graphics if needed.
            setup()

        self.inputs_initialized = False

    def initilize_inputs(self):
        if self.inputs_initialized:
            return

        self.inputs_initialized = True

        for i in range(len(self.species)):
            if self.species[i].particle_shape is None:
                self.species[i].particle_shape = self.particle_shape
            self.species[i].initialize_inputs(self.layouts[i])

        package('w3d')
        generate()

        for i in range(len(self.lasers)):
            self.lasers[i].initialize_inputs(self.solver, self.laser_injection_methods[i])

    def step(self, nsteps=1):
        self.initilize_inputs()
        step(nsteps)

    def write_input_file(self, file_name='inputs'):
        self.initilize_inputs()
        pass

