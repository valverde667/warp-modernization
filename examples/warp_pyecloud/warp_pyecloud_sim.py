import numpy as np
import numpy.random as random
from warp import picmi
from scipy.stats import gaussian_kde
from warp.particles.Secondaries import *
import matplotlib.pyplot as plt
import scipy.io as sio
from io import BytesIO as StringIO

# Construct PyECLOUD secondary emission object
import PyECLOUD.sec_emission_model_ECLOUD as seec
sey_mod = seec.SEY_model_ECLOUD(Emax=300., del_max=1.8, R0=0.7, E_th=30,
        sigmafit=1.09, mufit=1.66,
        secondary_angle_distribution='cosine_3D')



mysolver = 'ES' # solver type ('ES'=electrostatic; 'EM'=electromagnetic)
unit = 1e-3

##################################
# Define the grid
##################################

zs_dipo = -500*unit
ze_dipo = 500*unit
r = 23.e-3
h = 18.e-3

nx = 50
ny = 50
nz = 50

dh_x = 2*r/nx
dh_y = 2*h/ny

xmin = -r-4*dh_x
xmax = r+4*dh_x
ymin = -h-4*dh_y
ymax = h+4*dh_y
zmin = zs_dipo-50*unit
zmax = ze_dipo+50*unit

##################################
# Setup beam
##################################
sigmat= 1.000000e-09/4.
sigmaz = sigmat*299792458.
sigmax = 2e-4
sigmay = 2e-4

b_spac = 25e-9
t_offs = b_spac-6*sigmat
n_bunches = 1

beam_gamma = 479.
beam_uz = beam_gamma*picmi.constants.c

bunch_physical_particles  = 2.5e11
bunch_w = 1e8
bunch_macro_particles = bunch_physical_particles/bunch_w


bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,zs_dipo-10*unit]
bunch_centroid_velocity   = [0.,0.,beam_uz]
gauss_dist = picmi.GaussianBunchDistribution(
                                n_physical_particles = bunch_physical_particles,
                                rms_bunch_size       = bunch_rms_size,
                                rms_velocity         = bunch_rms_velocity,
                                centroid_position    = bunch_centroid_position,
                                centroid_velocity    = bunch_centroid_velocity )

beam = picmi.Species(particle_type = 'proton',
                     particle_shape = 'linear',
                     name = 'beam')

##################################
# Setup electrons
##################################
electron_background_dist = picmi.UniformDistribution(
                                            lower_bound = [-r, -h, zs_dipo],
                                            upper_bound = [r, h, ze_dipo],
                                            density = 1.e5/(4*r*h)
                                            )

elecb = picmi.Species(particle_type = 'electron',
                     particle_shape = 'linear',
                     name = 'Electron background',
                     initial_distribution = electron_background_dist)

secelec = picmi.Species(particle_type = 'electron',
                        particle_shape = 'linear',
                        name = 'Secondary electrons')

##################################
# Numeric components
##################################
if mysolver=='ES':
    lower_boundary_conditions = ['dirichlet', 'dirichlet', 'dirichlet']
    upper_boundary_conditions = ['dirichlet', 'dirichlet', 'dirichlet']
if mysolver=='EM':
    lower_boundary_conditions = ['open', 'open', 'open']
    upper_boundary_conditions = ['open', 'open', 'open']


grid = picmi.Cartesian3DGrid(number_of_cells = [nx, ny, nz],
                        lower_bound = [xmin, ymin, zmin],
                        upper_bound = [xmax, ymax, zmax],
                        lower_boundary_conditions = lower_boundary_conditions,
                        upper_boundary_conditions = upper_boundary_conditions)

if mysolver=='ES':
    solver = picmi.ElectrostaticSolver(grid = grid)

if mysolver=='EM':
    smoother = picmi.BinomialSmoother(n_pass = [[1], [1], [1]],
                                      compensation = [[False], [False], [False]],
                                      stride = [[1], [1], [1]],
                                      alpha = [[0.5], [0.5], [0.5]])

    solver = picmi.ElectromagneticSolver(grid = grid,
                                         method = 'CKC',
                                         cfl = 1.,
                                         source_smoother = smoother,
                                         warp_l_correct_num_Cherenkov = False,
                                         warp_type_rz_depose = 0,
                                         warp_l_setcowancoefs = True,
                                         warp_l_getrho = False)


##################################
# Setup Simulation
##################################
upper_box = picmi.warp.YPlane(y0=h,ysign=1,condid=1)
lower_box = picmi.warp.YPlane(y0=-h,ysign=-1,condid=1)
left_box = picmi.warp.XPlane(x0=r,xsign=1,condid=1)
right_box = picmi.warp.XPlane(x0=-r,xsign=-1,condid=1) 
sim = picmi.Simulation(solver = solver, verbose = 1,
                           warp_initialize_solver_after_generate = 1)
        
sim.conductors = upper_box + lower_box + left_box + right_box

beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

sim.add_species(beam, layout=beam_layout,
                initialize_self_field = solver=='EM')

elecb_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

sim.add_species(elecb, layout=elecb_layout,
                initialize_self_field = solver=='EM')

sim.add_species(secelec, layout=None, initialize_self_field=False)
                
##################################
# Add Dipole
##################################
By = 0.53
picmi.warp.addnewdipo(zs = zs_dipo, ze = ze_dipo, by = By)


##################################
# Beam Injection (Gaussian beam)
##################################
def time_prof(t):
    val = 0
    sigmat = sigmaz/picmi.clight
    for i in range(1,n_bunches+1):
        val += bunch_macro_particles*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*b_spac+t_offs)*(t-i*b_spac+t_offs)/(2*sigmat*sigmat))*picmi.warp.top.dt
    return val

def nonlinearsource():
    NP = int(time_prof(top.time))
    x = random.normal(bunch_centroid_position[0],bunch_rms_size[0],NP)
    y = random.normal(bunch_centroid_position[1],bunch_rms_size[1],NP)
    z = bunch_centroid_position[2]
    vx = random.normal(bunch_centroid_velocity[0],bunch_rms_velocity[0],NP)
    vy = random.normal(bunch_centroid_velocity[1],bunch_rms_velocity[1],NP)
    vz = picmi.warp.clight*np.sqrt(1-1./(beam_gamma**2))
    beam.wspecies.addparticles(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,gi = 1./beam_gamma, w=bunch_w)

picmi.warp.installuserinjection(nonlinearsource)



##################################
# Setup particle scraper and
# secondaries
##################################
sim.step(1)
solver.solver.installconductor(sim.conductors, dfill = picmi.warp.largepos)
sim.step(1)

pp = warp.ParticleScraper(sim.conductors,
        lsavecondid=1, lsaveintercept=1, lcollectlpdata=1)

sec=Secondaries(conductors=sim.conductors,
                l_usenew=1, pyecloud_secemi_object=sey_mod,
                pyecloud_nel_mp_ref=1., pyecloud_fact_clean=1e-6,
                pyecloud_fact_split=1.5)

sec.add(incident_species = elecb.wspecies,
        emitted_species  = secelec.wspecies,
        conductor        = sim.conductors)
sec.add(incident_species = secelec.wspecies,
        emitted_species = secelec.wspecies,
        conductor       = sim.conductors)

# define shortcuts
pw = picmi.warp
pw.winon()
em = solver.solver
step=pw.step

# Define time step
if mysolver=='ES':
    pw.top.dt = 25e-12


ntsteps_p_bunch = b_spac/top.dt
tot_nsteps = int(round(b_spac*n_bunches/top.dt))
b_pass = 0
perc = 10
original = sys.stdout
text_trap = StringIO()
t0 = time.time()

##################################
# Perform Simulation
##################################
for n_step in range(tot_nsteps):
    if n_step/ntsteps_p_bunch > b_pass:
        b_pass+=1
        perc = 10
        print ('===========================')
        print ('Bunch passage: %d' %b_pass)
        print ('Number of electrons in the dipole: %d' %(np.sum(secelec.wspecies.getw())+np.sum(elecb.wspecies.getw())))
    if n_step%ntsteps_p_bunch/ntsteps_p_bunch*100>=perc:
        print ('%d%% of bunch passage' %perc)
        perc = perc+10
    
#    original = sys.stdout
#    sys.stdout = text_trap
    step(1)
#    sys.stdout = original
    
t1 = time.time()
totalt = t1-t0

print ('Run terminated in %ds' %totalt)

