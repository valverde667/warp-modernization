# Using Warp on on Cori

There are three options for using Warp on Cori.

1. Use the publicly available, pre-built installation. To use it, the python3 module must be loaded,
   and set the environment variable PYTHONUSERBASE to /global/common/software/warp/cori/public.

2. Build your own installation, following the instructions below.

3. Use the Shifter version, following the instructions further down.

# Installing Warp with mpi4py on Cori

This section describes how to install Warp on the Cori cluster.
This installation is done by compiling the sources.

## Setting up the environnement

Create a directory where Warp and its associated packages will be installed. For example:
```
mkdir $SCRATCH/warp_install
```
NB: This directory is ideally in the `$SCRATCH` directory, for fast reading
access. Also, the `$SCRATCH` directory is specific to each cluster and is thus
well-adapted for system-specific installation, whereas the
`$HOME` directory is shared for all the clusters.

Set the environment variable PYTHONUSERBASE to that location:

For tcsh (add this to .tchrc.ext:

```
setenv PYTHONUSERBASE $SCRATCH/warp_install
```

For bash (add this to .bashrc.ext:
```
export PYTHONUSERBASE=$SCRATCH/warp_install
```

Add the following module commands to either .tcshrc.ext or .bashrc.ext as appropriate.
Note that the installation of Warp works better using the gnu environment.
```
module swap PrgEnv-intel PrgEnv-gnu
module load python3
module load h5py-parallel
```

Log out of Cori and then log back in again.

## Installing Forthon

As explained on the Warp website, Forthon is installed with this command:

```
pip install Forthon --user
```

## Installing Warp itself

Download the source by entering `git clone https://bitbucket.org/berkeleylab/warp.git`.
This can be done anywhere within $HOME or $SCRATCH.

### Installing with gnu

- Go to the `warp/pywarp90` directory and create a file called `Makefile.local3.pympi`,
  adding the following lines to this file:
```
FCOMP = -F gfortran --fargs "-fPIC" --cargs "-fPIC"
FCOMPEXEC = --fcompexec ftn
FORTHON = $(PYTHONUSERBASE)/bin/Forthon
INSTALLOPTIONS = --user
```

- Then, in the directory `warp/pywarp90`, enter `make pinstall`. The compilation will take a few minutes.

### Installing with the Intel compiler

- Setup the modules to use the intel compiler: `module load PrgEnv-intel`  

- Go to the `warp/pywarp90` directory and create a file called `Makefile.local3.pympi`.

If you want to compile for Haswell Architecture, enter the following lines in this file :

```
FCOMP = -F intel --fargs "-fPIC -O3 -xCORE-AVX2" --cargs "-fPIC"
FCOMPEXEC =  --fcompexec ftn
FORTHON = $(PYTHONUSERBASE)/bin/Forthon
INSTALLOPTIONS = --user
```

For MIC architecture, such as Intel Xeon Phi KNL, replace `-xCORE-AVX2` by `-xMIC-AVX512`.
This activates the use of AVX512 instructions for best performance on KNL.
However, note that KNL supports previous Intel instructions and your code will
work even compiled for Haswell (Cori phase 1) or Ivy Bridge (Edison) architectures.

```
FCOMP = -F intel --fargs "-fPIC -O3 -xMIC-AVX512" --cargs "-fPIC"
FCOMPEXEC =  --fcompexec ftn
FORTHON = $(PYTHONUSERBASE)/bin/Forthon
INSTALLOPTIONS = --user
```

- Then, in the directory `warp/pywarp90`, enter `make pinstall`. The compilation will take a few minutes.

### Running simulations

In order to run a simulation, create a new directory (in $SCRATCH)
and copy your Warp input script to this directory.
(The folder `scripts/examples` of the
[Warp repository](https://bitbucket.org/berkeleylab/warp/src) contains
several examples of input scripts.)

Then create a submission script named `submission_script`. Here is an
example of a typical submission script, where warp_script.py is your input file.
```
#!/bin/bash
#SBATCH --job-name=test_simulation
#SBATCH --time=00:30:00
#SBATCH -n 32
#SBATCH --partition=debug
#SBATCH -C haswell
#SBATCH -e test_simulation.err
#SBATCH -o test_simulation.out

srun -n 32 python warp_script.py -p 2 1 16
```

Submit the simulation by typing `sbatch submission_script`. The
progress of the simulation can be seen by typing ```squeue -u `whoami` ```.

# Using Shifter

You can also use Shifter
[Shifter](http://www.nersc.gov/research-and-development/user-defined-images/)
to run Warp on Cori. Shifter runs in a Linux container (similar to
Docker), and allows to easily port codes from one
architecture to another.

## Running simulations with Shifter

In order to run a simulation, create a new directory
and copy your Warp input script to this directory.
(The folder `scripts/examples/` of the
[Warp repository](https://bitbucket.org/berkeleylab/warp/src) contains
several examples of input scripts.)

Then create a submission script named `submission_script`. Here is an
example of a typical submission script, where warp_script.py is your input file.
```
#!/bin/bash
#SBATCH --job-name=test_cori_shifter
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --partition=debug
#SBATCH -e test_cori_shifter.err
#SBATCH -o test_cori_shifter.out
#SBATCH -C haswell
#SBATCH --image=docker:rlehe/warp:latest
#SBATCH --volume=<your$SCRATCH>:/home/warp_user/run

module unload python3

setenv OMP_NUM_THREADS 1
srun -n 32 -c 2 shifter python warp_script.py -p 4 1 8
```
Note that the options `--image=docker:rlehe/warp:latest` and `--
volume=<your$SCRATCH>:/home/warp_user/run` are essential
and should be copied exactly (**do not** replace `warp_user` or
`rlehe` by your username), with the exception of `<your$SCRATCH>`,
which should be replaced by the full path to your SCRATCH directory.

Then submit the simulation by typing `sbatch submission_script`.  The
progress of the simulation can be seen by typing ```squeue -u `whoami` ```.
