"""
This file defines the class BoostedParticleDiagnostic

Major features:
- The class reuses the existing methods of ParticleDiagnostic
  as much as possible, through class inheritance
- The class implements memory buffering of the slices, so as
  not to write to disk at every timestep
- The boosted frame diagnostics cannot use parallel HDF5 output
"""
import os
import numpy as np
import time
from scipy.constants import c
from particle_diag import ParticleDiagnostic
from parallel import gatherarray

class BoostedParticleDiagnostic(ParticleDiagnostic):
    """
    Class that writes the particles *in the lab frame*,
    from a simulation in the boosted frame

    Usage
    -----
    After initialization, the diagnostic is called by using
    the 'write' method.
    """
    def __init__(self, zmin_lab, zmax_lab, v_lab, dt_snapshots_lab,
                 Ntot_snapshots_lab, gamma_boost, period,
                 em, top, w3d, comm_world=None,
                 particle_data=["position", "momentum", "weighting"],
                 select=None, write_dir=None, species={"electrons": None}):
        """
        Initialize diagnostics that retrieve the data in the lab frame,
        as a series of snapshot (one file per snapshot),
        within a virtual moving window defined by zmin_lab, zmax_lab, v_lab.

        Note: In the current implementation, these diagnostics do not
        use parallel HDF5 output. Rank 0 creates and writes all the files.
        
        Parameters
        ----------
        zmin_lab, zmax_lab: floats (meters)
            Positions of the minimum and maximum of the virtual moving window,
            *in the lab frame*, at t=0

        v_lab: float (m.s^-1)
            Speed of the moving window *in the lab frame*

        dt_snapshots_lab: float (seconds)
            Time interval *in the lab frame* between two successive snapshots

        Ntot_snapshots_lab: int
            Total number of snapshots that this diagnostic will produce

        period: int
            Number of iterations for which the data is accumulated in memory,
            before finally writing it to the disk.

        See the documentation of ParticleDiagnostic for the other parameters
        """
        # Do not leave write_dir as None, as this may conflict with
        # the default directory ('./diags') in which diagnostics in the
        # boosted frame are written
        if write_dir is None:
            write_dir = 'lab_diags'

        # Initialize Particle diagnostic normal attributes
        ParticleDiagnostic.__init__(self, period, top, w3d, comm_world,
            species=species, particle_data=particle_data, select=select,
            write_dir=write_dir, lparallel_output=False)
        # Note: The boosted frame diagnostics cannot use parallel HDF5 output

        # Register the boost quantities
        self.gamma_boost = gamma_boost
        self.inv_gamma_boost = 1./gamma_boost
        self.beta_boost = np.sqrt(1. - self.inv_gamma_boost**2)
        self.inv_beta_boost = 1./self.beta_boost

        # Create the list of LabSnapshot objects
        self.snapshots = []

        # Record the time it takes
        if self.rank == 0:
            measured_start = time.clock()
            print('\nInitializing the lab-frame diagnostics: %d files...' %(
                Ntot_snapshots_lab) )

        # Loop through the lab snapshots and create the corresponding files
        self.particle_catcher = ParticleCatcher(
            self.gamma_boost, self.beta_boost, top )
        self.particle_catcher.allocate_previous_instant()

        for i in range( Ntot_snapshots_lab ):
            t_lab = i*dt_snapshots_lab
            snapshot = LabSnapshot( t_lab, zmin_lab + v_lab*t_lab,
                            top.dt, zmax_lab + v_lab*t_lab,
                            self.write_dir, i, self.species_dict, self.rank )
            self.snapshots.append( snapshot )
            # Initialize a corresponding empty file
            if self.rank == 0:
                self.create_file_empty_particles(
                    snapshot.filename, i, snapshot.t_lab, self.top.dt)

        # Print a message that records the time for initialization
        if self.rank == 0:
            measured_end = time.clock()
            print('Time taken for initialization of the files: %.5f s' %(
                measured_end - measured_start) )

    def write( self ):
        """
        Redefines the method write of the parent class ParticleDiagnostic

        Should be registered with installafterstep in Warp
        """
        # At each timestep, store a slice of the particles in memory buffers
        self.store_snapshot_slices()

        # Every self.period, write the buffered slices to disk
        if self.top.it % self.period == 0:
            self.flush_to_disk()

    def store_snapshot_slices( self ):
        """
        Store slices of the particles in the memory buffers of the
        corresponding lab snapshots

        The particles (in a slice) are also selected in accordance with
        the selection rules provided as argument to BoostedParticleDiagnostic
        """
        # Loop through the labsnapshots
        for snapshot in self.snapshots:

            # Update the positions of the output slice of this snapshot
            # in the lab and boosted frame (current_z_lab and current_z_boost)
            snapshot.update_current_output_positions( self.top.time,
                            self.inv_gamma_boost, self.inv_beta_boost)

            # For this snapshot:
            # - check if the output position *in the lab frame*
            #   is within the lab-frame boundaries of the current snapshot
            if ( (snapshot.current_z_lab > snapshot.zmin_lab) and \
                 (snapshot.current_z_lab < snapshot.zmax_lab) ):

                # Loop through the particle species and register the
                # particle arrays in the snapshot objects (buffering)
                for species_name, species in self.species_dict.iteritems():

                    slice_array = self.particle_catcher.extract_slice(
                        species, self.select, snapshot.prev_z_boost,
                        snapshot.current_z_boost, snapshot.t_lab )
                    snapshot.register_slice( slice_array, species_name )

    def flush_to_disk(self):
        """
        Writes the buffered slices of particles to the disk. Erase the
        buffered slices of the LabSnapshot objects

        Notice: In parallel version, data are gathered to proc 0
        before being saved to disk
        """
        # Loop through the labsnapshots and flush the data

        for snapshot in self.snapshots:

            # Compact the successive slices that have been buffered
            # over time into a single array
            for species_name in self.species_dict:

                particle_array = snapshot.compact_slices(species_name)

                if self.comm_world is not None:
                    # In MPI mode: gather and an array containing the number
                    # of particles on each process
                    n_rank = self.comm_world.allgather(
                        np.shape(particle_array)[1])

                    # Note that gatherarray routine in parallel.py only works
                    # with 1D array. Here we flatten the 2D particle arrays
                    # before gathering.
                    g_curr = gatherarray(particle_array.flatten(), root=0,
                        comm=self.comm_world)

                    if self.rank == 0:
                        # Get the number of quantities
                        nquant = np.shape(
                            self.particle_catcher.particle_to_index.keys())[0]

                        # Prepare an empty array for reshaping purposes. The
                        # final shape of the array is (8, total_num_particles)
                        p_array = np.empty((nquant, 0))

                        # Index needed in reshaping process
                        n_ind = 0

                        # Loop over all the processors, if the processor
                        # contains particles, we reshape the gathered_array
                        # and reconstruct by concatenation
                        for i in xrange(self.top.nprocs):

                            if n_rank[i] != 0:
                                p_array = np.concatenate((p_array, np.reshape(
                                    g_curr[n_ind:n_ind+nquant*n_rank[i]],
                                    (nquant,n_rank[i]))),axis=1)

                                # Update the index
                                n_ind += nquant*n_rank[i]

                else:
                    p_array = particle_array

                # Write this array to disk (if this snapshot has new slices)
                if self.rank == 0 and p_array.size:
                    self.write_slices(p_array, species_name, snapshot,
                        self.particle_catcher.particle_to_index)

                # Erase the buffers
                snapshot.buffered_slices[species_name] = []

    def write_boosted_dataset(self, species_grp, path, data, quantity):
        """
        Writes each quantity of the buffered dataset to the disk, the
        final step of the writing
        """
        dset = species_grp[path]
        index = dset.shape[0]

        # Resize the h5py dataset
        dset.resize(index+len(data), axis=0)

        # Write the data to the dataset at correct indices
        dset[index:] = data

    def write_slices( self, particle_array, species_name, snapshot, p2i ):
        """
        For one given snapshot, write the slices of the
        different species to an openPMD file

        Parameters
        ----------
        particle_array: array of reals
            Array of shape (8, num_part)

        species_name: String
            A String that acts as the key for the buffered_slices dictionary

        snapshot: a LabSnaphot object

        p2i: dict
            Dictionary of correspondance between the particle quantities
            and the integer index in the particle_array
        """
        # Open the file without parallel I/O in this implementation
        f = self.open_file( snapshot.filename, parallel_open=False )
        particle_path = "/data/%d/particles/%s" %(snapshot.iteration,
                                                    species_name)
        species_grp = f[particle_path]

        # Loop over the different quantities that should be written
        for particle_var in self.particle_data:

            if particle_var == "position":
                for coord in ["x","y","z"]:
                    quantity= coord
                    path = "%s/%s" %(particle_var, quantity)
                    data = particle_array[ p2i[ quantity ] ]
                    self.write_boosted_dataset(
                            species_grp, path, data, quantity)

            elif particle_var == "momentum":
                for coord in ["x","y","z"]:
                    quantity= "u%s" %coord
                    path = "%s/%s" %(particle_var,coord)
                    data = particle_array[ p2i[ quantity ] ]
                    self.write_boosted_dataset(
                            species_grp, path, data, quantity)

            elif particle_var == "weighting":
               quantity= "w"
               path = 'weighting'
               data = particle_array[ p2i[ quantity ] ]
               self.write_boosted_dataset(species_grp, path, data, quantity)

        # Close the file
        f.close()

class LabSnapshot:
    """
    Class that stores data relative to one given snapshot
    in the lab frame (i.e. one given *time* in the lab frame)
    """
    def __init__(self, t_lab, zmin_lab, dt, zmax_lab, write_dir, i,
        species_dict, rank):
        """
        Initialize a LabSnapshot

        Parameters
        ----------
        t_lab: float (seconds)
            Time of this snapshot *in the lab frame*

        zmin_lab, zmax_lab: floats
            Longitudinal limits of this snapshot

        write_dir: string
            Absolute path to the directory where the data for
            this snapshot is to be written

        i: int
            Number of the file where this snapshot is to be written

        species_dict: dict
            Contains all the species name of the species object
            (inherited from Warp)
        """
        # Deduce the name of the filename where this snapshot writes
        if rank == 0:
            self.filename = os.path.join( write_dir, 'hdf5/data%08d.h5' %i)
        self.iteration = i
        self.dt = dt

        # Time and boundaries in the lab frame (constants quantities)
        self.zmin_lab = zmin_lab
        self.zmax_lab = zmax_lab
        self.t_lab = t_lab

        # Positions where the fields are to be registered
        # (Change at every iteration)
        self.current_z_lab = 0
        self.current_z_boost = 0

        # Prepare buffered slices
        self.buffered_slices = {}
        for species_name in species_dict:
            self.buffered_slices[species_name] = []

    def update_current_output_positions( self, t_boost, inv_gamma, inv_beta ):
        """
        Update the current and previous positions of output for this snapshot,
        so that it corresponds to the time t_boost in the boosted frame

        Parameters
        ----------
        t_boost: float (seconds)
            Time of the current iteration, in the boosted frame

        inv_gamma, inv_beta: floats
            Inverse of the Lorentz factor of the boost, and inverse
            of the corresponding beta
        """
        # Some shorcuts for further calculation's purposes
        t_lab = self.t_lab
        t_boost_prev = t_boost - self.dt

        # This implements the Lorentz transformation formulas,
        # for a snapshot having a fixed t_lab
        self.current_z_boost = (t_lab*inv_gamma - t_boost)*c*inv_beta
        self.prev_z_boost = (t_lab*inv_gamma - t_boost_prev)*c*inv_beta
        self.current_z_lab = (t_lab - t_boost*inv_gamma)*c*inv_beta
        self.prev_z_lab = (t_lab - t_boost_prev*inv_gamma)*c*inv_beta

    def register_slice(self, slice_array, species):
        """
        Store the slice of particles represented by slice_array

        Parameters
        ----------
        slice_array: array of reals
            An array of packed fields that corresponds to one slice,
            as given by the ParticleCatcher object

        species: String, key of the species_dict
            Act as the key for the buffered_slices dictionary
        """
        # Store the values
        self.buffered_slices[species].append(slice_array)

    def compact_slices(self, species):
        """
        Compact the successive slices that have been buffered
        over time into a single array.

        Parameters
        ----------
        species: String, key of the species_dict
            Act as the key for the buffered_slices dictionary

        Returns
        -------
        paticle_array: an array of reals of shape (9, numPart)
        regardless of the dimension

        Returns None if the slices are empty
        """
        if self.buffered_slices[species] != []:
            particle_array = np.concatenate(
                self.buffered_slices[species], axis=1)
        else:
            particle_array = np.empty((9,0))

        return particle_array

class ParticleCatcher:
    """
    Class that extracts, Lorentz-transforms and gathers particles
    """
    def __init__(self, gamma_boost, beta_boost, top):
        """
        Initialize the ParticleCatcher object

        Parameters
        ----------
        gamma_boost, beta_boost: float
            The Lorentz factor of the boost and the corresponding beta

        top: WARP object
        """
        # Some attributes neccessary for particle selections
        self.gamma_boost = gamma_boost
        self.beta_boost = beta_boost
        self.top = top

        # Create a dictionary that contains the correspondance
        # between the particles quantity and array index
        self.particle_to_index = {'x':0, 'y':1, 'z':2, 'ux':3,
                'uy':4, 'uz':5, 'w':6, 'gamma':7, 't':8}

    def get_particle_slice( self, species, prev_z_boost, current_z_boost ):
        """
        Select the particles for the current slice, and extract their
        positions and momenta at the current and previous timestep

        Parameters
        ----------
        species: a Species object of Warp
            Contains the particle data from which one slice will be extracted

        current_z_boost, prev_z_boost: floats
            The position, in the boosted frame, of the plane that
            corresponds to the lab snapshot, at the current iteration and
            at the previous iteration.
            Only particles that have crossed this plane, between the previous
            iteration and the current iteration, are extracted

        Returns
        -------
        num_part: int
            Number of selected particles
        """
        # Quantities at current time step
        current_x = self.get_quantity( species, "x" )
        current_y = self.get_quantity( species, "y" )
        current_z = self.get_quantity( species, "z" )
        current_ux = self.get_quantity( species, "ux" )
        current_uy = self.get_quantity( species, "uy" )
        current_uz = self.get_quantity( species, "uz" )
        current_weights = self.get_quantity( species, "w" )

        # Quantities at previous time step
        previous_x = self.get_quantity( species, "x", l_prev=True )
        previous_y = self.get_quantity( species, "y", l_prev=True )
        previous_z = self.get_quantity( species, "z", l_prev=True )
        previous_ux = self.get_quantity( species, "ux", l_prev=True )
        previous_uy = self.get_quantity( species, "uy", l_prev=True )
        previous_uz = self.get_quantity( species, "uz", l_prev=True )

        # A particle array for mapping purposes
        particle_indices = np.arange( len(current_z) )

        # For this snapshot:
        # - check if the output position *in the boosted frame*
        #   crosses the zboost in a forward motion
        # - check if the output position *in the boosted frame*
        #   crosses the zboost_prev in a backward motion
        selected_indices = np.compress((
            ((current_z >= current_z_boost ) & (previous_z <= prev_z_boost )) |
            ((current_z <= current_z_boost ) & (previous_z >= prev_z_boost ))
            ), particle_indices)

        num_part = np.shape(selected_indices)[0]

        ## Particle quantities that satisfy the aforementioned condition
        self.mass = species.mass

        self.x_captured = np.take(current_x, selected_indices)
        self.y_captured = np.take(current_y, selected_indices)
        self.z_captured = np.take(current_z, selected_indices)
        self.ux_captured = np.take(current_ux, selected_indices)
        self.uy_captured = np.take(current_uy, selected_indices)
        self.uz_captured = np.take(current_uz, selected_indices)
        self.w_captured = np.take(current_weights, selected_indices)
        self.gamma_captured = np.sqrt(1. + (self.ux_captured**2+\
            self.uy_captured**2 + self.uz_captured**2)/c**2)

        self.x_prev_captured = np.take(previous_x, selected_indices)
        self.y_prev_captured = np.take(previous_y, selected_indices)
        self.z_prev_captured = np.take(previous_z, selected_indices)
        self.ux_prev_captured = np.take(previous_ux, selected_indices)
        self.uy_prev_captured = np.take(previous_uy, selected_indices)
        self.uz_prev_captured = np.take(previous_uz, selected_indices)
        self.gamma_prev_captured = np.sqrt(1. + (self.ux_prev_captured**2+\
            self.uy_prev_captured**2 + self.uz_prev_captured**2)/c**2)

        return( num_part )

    def transform_particles_to_lab_frame(self):
        """
        Transform the particle quantities from the boosted frame to the
        lab frame. These are classical Lorentz transformation equations
        """
        uzfrm = -self.beta_boost*self.gamma_boost*c

        # Time in lab frame
        self.t = self.gamma_boost*self.top.time - uzfrm*self.z_captured/c**2
        self.t_prev = self.gamma_boost*(self.top.time - self.top.dt) \
          - uzfrm*self.z_prev_captured/c**2

        # Position in lab frame
        self.z_captured = self.gamma_boost*(self.z_captured + \
            self.beta_boost*c*self.top.time)
        self.z_prev_captured = self.gamma_boost*(self.z_prev_captured \
            + self.beta_boost*c*(self.top.time-self.top.dt))

        # Momentum in lab frame
        self.uz_captured = self.gamma_boost*self.uz_captured \
        - self.gamma_captured*uzfrm
        self.uz_prev_captured = self.gamma_boost*self.uz_prev_captured \
        - self.gamma_prev_captured*uzfrm

    def interpolate_to_time(self, t_output):
        """
        Interpolate the particle quantities in time to the instant t

        Parameter
        ---------
        t_output: float
            Time **in the lab frame** at which the quantities should be
            interpolated. (Typically, the time of a given LabSnapshot)
        """
        # Calculation interpolation weights, for the time interpolation
        weight_prev = (self.t - t_output)/(self.t - self.t_prev)
        weight_next = (t_output - self.t_prev)/(self.t - self.t_prev)

        # Perform the interpolation
        self.x_captured = \
          self.x_prev_captured * weight_prev + self.x_captured * weight_next
        self.y_captured = \
          self.y_prev_captured * weight_prev + self.y_captured * weight_next
        self.z_captured = \
          self.z_prev_captured * weight_prev + self.z_captured * weight_next
        self.ux_captured = \
          self.ux_prev_captured * weight_prev + self.ux_captured * weight_next
        self.uy_captured = \
          self.uy_prev_captured * weight_prev + self.uy_captured * weight_next
        self.uz_captured = \
          self.uz_prev_captured * weight_prev + self.uz_captured * weight_next
        self.gamma_captured = self.gamma_prev_captured * weight_prev + \
          self.gamma_captured * weight_next

    def gather_array(self, quantity):
        """
        Gather the quantity arrays and normalize the momenta
        Parameters
        ----------
        quantity: String
            Quantity of the particles that is wished to be gathered

        Returns
        -------
        ar: array of reals
            An array of gathered particle's quantity
        """
        ar = np.zeros(np.shape(self.x_captured)[0])

        if quantity == "x":
            ar = np.array(self.x_captured)
        elif quantity == "y":
            ar = np.array(self.y_captured)
        elif quantity == "z":
            ar = np.array(self.z_captured)
        elif quantity == "ux":
            ar = np.array(self.ux_captured)
        elif quantity == "uy":
            ar = np.array(self.uy_captured)
        elif quantity == "uz":
            ar = np.array(self.uz_captured)
        elif quantity == "w":
            ar = np.array(self.w_captured)
        elif quantity == "gamma":
            ar = np.array(self.gamma_captured)
        return ar

    def extract_slice(self, species, select, prev_z_boost,
                      current_z_boost, t_output ):
        """
        Extract a slice of the particles that corresponds to a given
        lab snapshot, and transform them to the lab frame

        If select is present, extract only particles that satisfy the criteria

        Parameters
        ----------
        species: a Species object of Warp
            Contains the particle data from which one slice will be extracted

        select: dict
            A set of rules defined by the users in selecting the particles
            Ex: {"uz": [50, 100]} for particles which have normalized
            values between 50 and 100

        current_z_boost, prev_z_boost: floats
            The position, in the boosted frame, of the plane that
            corresponds to the lab snapshot, at the current iteration and
            at the previous iteration.
            Only particles that have crossed this plane, between the previous
            iteration and the current iteration, are extracted

        t_output: float (s)
            Time **in the lab frame** at which the quantities should be
            obtained. (Typically, the time of a given LabSnapshot.) This
            is used when interpolating between two simulation timesteps.

        Returns
        -------
        slice_array: An array of reals of shape (8, num_part)
            An array that packs together the different particle quantities
            (x, y, z, ux, uy, uz, weight)
        """
        # Declare an attribute for convenience
        p2i = self.particle_to_index

        # Get the particles
        num_part = self.get_particle_slice( species, prev_z_boost,
                        current_z_boost )

        # Transform the particles from boosted frame back to lab frame
        self.transform_particles_to_lab_frame()

        # Interpolate the particle quantities in time, to t_output
        self.interpolate_to_time( t_output )
        slice_array = np.empty((np.shape(p2i.keys())[0], num_part,))

        for quantity in self.particle_to_index.keys():
            # Here typical values for 'quantity' are e.g. 'z', 'ux', 'gamma'
            # you should just gather array locally
            slice_array[ p2i[quantity], ... ] = self.gather_array(quantity)

        # Choose the particles based on the select criteria defined by the
        # users. Notice: this implementation still comes with a cost,
        # one way to optimize it would be to do the selection before Lorentz
        # transformation back to the lab frame
        if (select is not None) and slice_array.size:
            select_array = self.apply_selection(select, slice_array)
            row, column =  np.where(select_array==True)
            temp_slice_array = slice_array[row,column]

            # Temp_slice_array is a 1D numpy array, we reshape it so that it
            # has the same size as slice_array
            slice_array = np.reshape(
                temp_slice_array,(np.shape(p2i.keys())[0],-1))

        # Multiplying momenta by the species mass to make them unitless
        for quantity in ["ux", "uy", "uz"]:
            slice_array[p2i[quantity]] *= species.mass

        return slice_array

    def get_quantity(self, species, quantity, l_prev=False):
        """
        Get a given particle quantity

        Parameters
        ----------
        species: a Species object of Warp
            Contains the particle data from which the quantity is extracted

        quantity: string
            Describes which quantity is queried
            Either "x", "y", "z", "ux", "uy", "uz" or "w"

        l_prev: boolean
            If True, then return the quantities of the previous timestep;
            else return quantities of the current timestep
        """
        # Extract the chosen quantities

        # At current timestep
        if not(l_prev):
            if quantity == "x":
                quantity_array = species.getx( gather=False )
            elif quantity == "y":
                quantity_array = species.gety( gather=False )
            elif quantity == "z":
                quantity_array = species.getz( gather=False )
            elif quantity == "ux":
                quantity_array = species.getux( gather=False )
            elif quantity == "uy":
                quantity_array = species.getuy( gather=False )
            elif quantity == "uz":
                quantity_array = species.getuz( gather=False )
            elif quantity == "w":
                quantity_array = species.getweights( gather=False )

        # Or at previous timestep
        else:
            if quantity == "x":
                quantity_array = species.getpid( id=self.top.xoldpid-1,
                    gather=0, bcast=0)
            elif quantity == "y":
                quantity_array = species.getpid( id=self.top.yoldpid-1,
                    gather=0, bcast=0)
            elif quantity == "z":
                quantity_array = species.getpid( id=self.top.zoldpid-1,
                    gather=0, bcast=0)
            elif quantity == "ux":
                quantity_array = species.getpid( id=self.top.uxoldpid-1,
                    gather=0, bcast=0)
            elif quantity == "uy":
                quantity_array = species.getpid( id=self.top.uyoldpid-1,
                    gather=0, bcast=0)
            elif quantity == "uz":
                quantity_array = species.getpid( id=self.top.uzoldpid-1,
                    gather=0, bcast=0)

        return( quantity_array )

    def allocate_previous_instant(self):
        """
        Allocate the top.'quantity'oldpid arrays. This is used to store
        the previous values of the quantities.
        """
        if not self.top.xoldpid:
            self.top.xoldpid = self.top.nextpid()
        if not self.top.yoldpid:
            self.top.yoldpid = self.top.nextpid()
        if not self.top.zoldpid:
            self.top.zoldpid = self.top.nextpid()
        if not self.top.uxoldpid:
            self.top.uxoldpid = self.top.nextpid()
        if not self.top.uyoldpid:
            self.top.uyoldpid = self.top.nextpid()
        if not self.top.uzoldpid:
            self.top.uzoldpid = self.top.nextpid()

    def apply_selection(self, select, slice_array):
        """
        Apply the rules of self.select to determine which
        particles should be written

        Parameters
        ----------
        select: a dictionary that defines all selection rules based
        on the quantities

        Returns
        -------
        A 1d array of the same shape as that particle array
        containing True for the particles that satify all
        the rules of self.select
        """
        p2i = self.particle_to_index

        # Initialize an array filled with True
        select_array = np.ones( np.shape(slice_array), dtype='bool' )

        # Apply the rules successively
        # Go through the quantities on which a rule applies
        for quantity in select.keys():
            # Lower bound
            if select[quantity][0] is not None:
                select_array = np.logical_and(
                    slice_array[p2i[quantity]] >\
                     select[quantity][0], select_array )
            # Upper bound
            if select[quantity][1] is not None:
                select_array = np.logical_and(
                    slice_array[p2i[quantity]] <\
                    select[quantity][1], select_array )

        return select_array
