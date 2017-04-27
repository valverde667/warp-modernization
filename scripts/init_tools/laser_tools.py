import numpy as np
from scipy.constants import c, m_e, e
from .boost_tools import BoostConverter
# Import laser antenna and laser profiles
from ..field_solvers.laser.laser_profiles import *
from warp import openbc

def add_laser( em, dim, a0, w0, ctau, z0, zf=None, lambda0=0.8e-6,
               theta_pol=0., source_z=0., zeta=0, beta=0, phi2=0,
               gamma_boost=None, laser_file=None, laser_file_energy=None ):
    """
    Add a linearly-polarized, Gaussian laser pulse in the em object,
    by setting the correct laser_func, laser_emax, laser_source_z
    and laser_polangle

    NB: When using this interface, the antenna is necessarily
    motionless in the lab-frame.

    Parameters
    ----------
    em : an EM3D object
       The structure that contains the fields of the simulation

    dim: str
       Either "2d", "3d" or "circ"

    a0 : float (unitless)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The a0 of a Gaussian pulse at focus

    w0 : float (in meters)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The waist of the Gaussian pulse at focus

    ctau : float (in meters)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The "longitudinal waist" (or pulse length) of the Gaussian pulse

    z0 : float (in meters)
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The position of the laser centroid relative to z=0.

    zf : float (in meters), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       The position of the laser focus relative to z=0.
       If not provided, then the laser focus is at z0

    lambda0 : float (in meters), optional
       The central wavelength of the laser
       Default : 0.8 microns (Ti:Sapph laser)

    theta_pol : float (in radians), optional
       The angle of polarization with respect to the x axis
       Default : 0 rad

    source_z : float (in meters), optional
       The position of the antenna that launches the laser

    zeta: float (in m.s), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       Spatial chirp, at focus,
       as defined in Akturk et al., Opt Express, vol 12, no 19 (2014)

    beta: float (in s), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       Angular dispersion, at focus,
       as defined in Akturk et al., Opt Express, vol 12, no 19 (2014)

    phi2: float (in s^2), optional
       *Used only if no laser_file is provided, i.e. for a Gaussian pulse*
       Temporal chirp, at focus,
       as defined in Akturk et al., Opt Express, vol 12, no 19 (2014)

    gamma_boost : float, optional
        When initializing the laser in a boosted frame, set the value of
        `gamma_boost` to the corresponding Lorentz factor. All the other
        quantities (ctau, zf, source_z, etc.) are to be given in the lab frame.

    laser_file: str or None
       If None, the laser will be initialized as Gaussian
       Otherwise, the laser_file should point to a standardized HDF5 file
       which contains the following datasets:
       - 't' and 'r': 1D datasets of coordinates, in SI
       - 'Ereal' and 'Eimag': 2D datasets in SI, so that the laser energy is 1J

    laser_file_energy: float or None
       *Used only if a laser_file is provided*
       Total energy of the pulse, in Joules
    """
    # Wavevector and speed of the antenna
    k0 = 2*np.pi/lambda0
    source_v = 0.
    inv_c = 1./c
    tau = ctau * inv_c
    t_peak = - z0 * inv_c
    if zf is None:
        focal_length = source_z - z0
    else:
        focal_length = source_z - zf

    # Create a laser_profile object
    # Note that the laser_profile needs to be a callable instance of a class,
    # i.e. an instance of a class with the __call__ method. This avoids the
    # problem of the EM solver not being picklable if laser_func were an
    # instance method, which is not picklable.

    # - Case of a Gaussian pulse
    if laser_file is None:

        # When running a simulation in boosted frame, convert these parameters
        boost = None
        if (gamma_boost is not None):
            boost = BoostConverter( gamma_boost )
            source_z, = boost.copropag_length([ source_z ],
                                              beta_object=source_v/c)
            source_v, = boost.velocity([ source_v ])

        # Create a laser profile object to store these parameters
        if (beta == 0) and (zeta == 0) and (phi2 == 0):
            # Without spatio-temporal correlations
            laser_profile = GaussianProfile( k0, w0, tau, t_peak, a0, dim,
                focal_length=focal_length, boost=boost, source_v=source_v )
        else:
            # With spatio-temporal correlations
            laser_profile = GaussianSTCProfile( k0, w0, tau, t_peak, a0, zeta,
                                   beta, phi2, dim, focal_length=focal_length,
                                   boost=boost, source_v=source_v )

    # - Case of an experimental profile
    else:

        # Reject boosted frame
        if (gamma_boost is not None) and (gamma_boost != 1.):
            raise ValueError('Boosted frame not implemented for '
                             'arbitrary laser profile.')

        # Create a laser profile object
        laser_profile = ExperimentalProfile( k0, laser_file,
                                             laser_file_energy )

    # Link its profile function the em object
    em.laser_func = laser_profile

    # Link the rest of the parameters to the em objects
    em.laser_emax = laser_profile.E0
    em.laser_source_z = source_z
    em.laser_source_v = source_v
    em.laser_polangle = theta_pol

    # Additional parameters for the order of deposition of the laser
    em.laser_depos_order_x=1
    em.laser_depos_order_y=1
    em.laser_depos_order_z=1


#===============================================================================
def retropropagation(em, w3d, negative_propagation=False):
    """
    This routine is used to retropropagate a laser.

	When the function is called, the B field sign is changed to inverse the
	direction of propagation.
	Then, after considering the plane of the antenna, which separates the box
	into 2 half-spaces, all the fields in one half-space are set to zero.
	Only one pulse generated by the antenna is thus kept, depending on the
    value of the flag negative_propagation.

	Parameter:
    -----------

	negative_propagation: boolean
		Indicate the half-space set to 0. If False, it suppresses the pulse
		propagating along laser_vector. If None, none of the spaces are set to
		0.
    """
    f = em.fields

    # Change the sign of B
    f.Bx = - f.Bx
    f.By = - f.By
    f.Bz = - f.Bz

    # Put zero values in the half space where the propagation was positive by
    # default.
    nbpoints = f.Ex.shape
    xmin = w3d.xmminlocal - em.nxguard*em.dx
    xmax = w3d.xmmaxlocal + em.nxguard*em.dx
    ymin = w3d.ymminlocal - em.nyguard*em.dy
    ymax = w3d.ymmaxlocal + em.nyguard*em.dy
    zmin = w3d.zmminlocal - em.nzguard*em.dz
    zmax = w3d.zmmaxlocal + em.nzguard*em.dz

    x = np.linspace(xmin, xmax, nbpoints[0])
    y = np.linspace(ymin, ymax, nbpoints[1])
    z = np.linspace(zmin, zmax, nbpoints[2])
    x,y,z = np.meshgrid(x,y,z,indexing='ij')

    vect = em.laser_antenna.vector
    spot = em.laser_antenna.spot

    mesh_points_antenna_frame = (x-spot[0]) * vect[0] + (y-spot[1]) * vect[1] \
                                + (z-spot[2]) * vect[2]

    # Set the fields to 0 if negative_propagation is defined
    if negative_propagation is not None :
        # Condition to find the corresponding halfspace depending on the value
        # of negative_propagation.
        if negative_propagation:
            zero_condition = (mesh_points_antenna_frame < 0 )
        else:
            zero_condition = (mesh_points_antenna_frame > 0 )

        # All the field arrays are put to 0 in this halfspace.
        # The Rho array is not reset assuming there were no particles before
        # and then no charges.
        f.Ex[zero_condition] = 0
        f.Ey[zero_condition] = 0
        f.Ez[zero_condition] = 0
        f.Bx[zero_condition] = 0
        f.By[zero_condition] = 0
        f.Bz[zero_condition] = 0
        f.Jx[zero_condition] = 0
        f.Jy[zero_condition] = 0
        f.Jz[zero_condition] = 0

        # Set the fields in the PML to 0 if existing
        b = em.bounds
        list_boundaries = []

        # side
        if b[0] == openbc:
            list_boundaries.append(em.block.sidexl.syf)
        if b[1] == openbc:
            list_boundaries.append(em.block.sidexr.syf)
        if b[2] == openbc:
            list_boundaries.append(em.block.sideyl.syf)
        if b[3] == openbc:
            list_boundaries.append(em.block.sideyr.syf)
        if b[4] == openbc:
            list_boundaries.append(em.block.sidezl.syf)
        if b[5] == openbc:
            list_boundaries.append(em.block.sidezr.syf)

        # edge
        if b[0] == openbc and b[2] == openbc:
            list_boundaries.append(em.block.edgexlyl.syf)
        if b[0] == openbc and b[3] == openbc:
            list_boundaries.append(em.block.edgexlyr.syf)
        if b[0] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.edgexlzl.syf)
        if b[0] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.edgexlzr.syf)
        if b[1] == openbc and b[2] == openbc:
            list_boundaries.append(em.block.edgexryl.syf)
        if b[1] == openbc and b[3] == openbc:
            list_boundaries.append(em.block.edgexryr.syf)
        if b[1] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.edgexrzl.syf)
        if b[1] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.edgexrzr.syf)
        if b[2] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.edgeylzl.syf)
        if b[2] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.edgeylzr.syf)
        if b[3] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.edgeyrzl.syf)
        if b[3] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.edgeyrzr.syf)

        # corner
        if b[0] == openbc and b[2] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.cornerxlylzl.syf)
        if b[0] == openbc and b[2] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.cornerxlylzr.syf)
        if b[0] == openbc and b[3] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.cornerxlyrzl.syf)
        if b[0] == openbc and b[3] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.cornerxlyrzr.syf)
        if b[1] == openbc and b[2] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.cornerxrylzl.syf)
        if b[1] == openbc and b[2] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.cornerxrylzr.syf)
        if b[1] == openbc and b[3] == openbc and b[4] == openbc:
            list_boundaries.append(em.block.cornerxryrzl.syf)
        if b[1] == openbc and b[3] == openbc and b[5] == openbc:
            list_boundaries.append(em.block.cornerxryrzr.syf)

        for syf in list_boundaries :
            syf.exx[...] = 0.;           syf.bxx[...] = 0.
            syf.exy[...] = 0.;           syf.bxy[...] = 0.
            syf.exz[...] = 0.;           syf.bxz[...] = 0.
            syf.eyx[...] = 0.;           syf.byx[...] = 0.
            syf.eyy[...] = 0.;           syf.byy[...] = 0.
            syf.eyz[...] = 0.;           syf.byz[...] = 0.
            syf.ezx[...] = 0.;           syf.bzx[...] = 0.
            syf.ezy[...] = 0.;           syf.bzy[...] = 0.
            syf.ezz[...] = 0.;           syf.bzz[...] = 0.

    print "================================================"
    print " Retropropagation completed."
    print "================================================"
