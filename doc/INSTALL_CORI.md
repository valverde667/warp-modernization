# Using Warp on on Cori

There are three options for using Warp on Cori.

1. Use the publicly available, pre-built installation. (Recommended)

2. Build your own installation, following the instructions below.

3. Use the Shifter version, following the instructions further down. (Recommended for jobs with a large number of processors)

# Using the publicly available installation

Warp is installed in a publicly available directory space.
To use it, load the most recent cray-python module.
```
module load cray-python/3.7.3.2
```
This version is setup using a virtual environment.
To activate the environment, run one of the following commands.
For csh and tcsh
```
source /global/cfs/cdirs/m2852/warp/warp_cori/bin/activate.csh
```
For bash
```
source /global/cfs/cdirs/m2852/warp/warp_cori/bin/activate
```
The activation will set the appropriate environment variables and execute path to use this installation.

# Installing Warp on Cori (with mpi4py)

This section describes how to install Warp on the Cori cluster, compiling the sources.

## Setting up the environnement

First, setup the modules needed.
The following module commands are recommented.
Note that the installation of Warp works better using the gnu environment. For python, use the most recent version of cray-python.
```
module swap PrgEnv-intel PrgEnv-gnu
module load cray-python/3.7.3.2
module load h5py-parallel
```

Warp should be installed within a virtual environment.
Creating the virtual environment creates the directory space where Warp will be installed.
It is recommended that this be in $SCRATCH space for optimum read access.
To create the environment:
```
python3 -m venv $SCRATCH/warp_install
```
Then activate the virtual environment.
For csh and tcsh
```
source $SCRATCH/warp_install/bin/activate.csh
```
For bash
```
source $SCRATCH/warp_install/bin/activate
```
The activation will set the appropriate environment variables and execute path to use this installation.

Several Python packages need to be installed.
```
pip install numpy
pip install Forthon
```

To install mpi4py, following the instructions on here https://docs.nersc.gov/programming/high-level-environments/python/mpi4py/.
For pygist, following the instructions on the main Warp installation page.

## Installing Warp itself

Download the source by entering `git clone https://bitbucket.org/berkeleylab/warp.git`.
This can be done anywhere within $HOME or $SCRATCH.

### Installing with gnu

- Go to the `warp/pywarp90` directory and create a file called `Makefile.local3.pympi`, adding the following lines to this file:
```
FCOMP = -F gfortran --fargs "-fPIC" --cargs "-fPIC"
FCOMPEXEC = --fcompexec ftn
```

- Then, in the directory `warp/pywarp90`, enter `make pinstall`. The compilation will take a few minutes.

For a serial version, copy `Makefile.local3.pympi` to `Makefile.local3` and run `make install`.

### Installing with the Intel compiler

- Setup the modules to use the intel compiler: `module load PrgEnv-intel`

- Go to the `warp/pywarp90` directory and create a file called `Makefile.local3.pympi`.

If you want to compile for Haswell Architecture, enter the following lines in this file:

```
FCOMP = -F intel --fargs "-fPIC -O3 -xCORE-AVX2" --cargs "-fPIC"
FCOMPEXEC = --fcompexec ftn
```

For MIC architecture, such as Intel Xeon Phi KNL, replace `-xCORE-AVX2` by `-xMIC-AVX512`.
This activates the use of AVX512 instructions for best performance on KNL.
However, note that KNL supports previous Intel instructions and your code will work even compiled for Haswell (Cori phase 1) or Ivy Bridge (Edison) architectures.

```
FCOMP = -F intel --fargs "-fPIC -O3 -xMIC-AVX512" --cargs "-fPIC"
FCOMPEXEC = --fcompexec ftn
```

- Then, in the directory `warp/pywarp90`, enter `make pinstall`. The compilation will take a few minutes.

For a serial version, copy `Makefile.local3.pympi` to `Makefile.local3` and run `make install`.

### Running simulations

In order to run a simulation, create a new directory (in $SCRATCH) and copy your Warp input script to this directory.
(The folder `scripts/examples` of the [Warp repository](https://bitbucket.org/berkeleylab/warp/src) contains several examples of input scripts.)

Then create a submission script named `submission_script`.
Here is an example of a typical submission script, where warp_script.py is your input file.
Note that $VENVHOME is either the publicly available virtual environment or the one created by you.
```
#!/bin/bash
#SBATCH --job-name=test_simulation
#SBATCH --time=00:30:00
#SBATCH -n 32
#SBATCH --partition=debug
#SBATCH -C haswell
#SBATCH -e test_simulation.err
#SBATCH -o test_simulation.out

source $VENVHOME/bin/activate
srun -n 32 python warp_script.py -p 2 1 16
```

Submit the simulation by typing `sbatch submission_script`.
The progress of the simulation can be seen by typing ```squeue -u `whoami` ```.

# Using Shifter

You can also use Shifter [Shifter](http://www.nersc.gov/research-and-development/user-defined-images/) to run Warp on Cori.
Shifter runs in a Linux container (similar to Docker), and allows to easily port codes from one architecture to another.

## Running simulations with Shifter

In order to run a simulation, create a new directory and copy your Warp input script to this directory.
(The folder `scripts/examples/` of the [Warp repository](https://bitbucket.org/berkeleylab/warp/src) contains several examples of input scripts.)

Then create a submission script named `submission_script`.
Here is an example of a typical submission script, where warp_script.py is your input file.
Note that all python modules need to be unloaded.
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

module unload python
module unload cray-python

setenv OMP_NUM_THREADS 1
srun -n 32 -c 2 shifter python warp_script.py -p 4 1 8
```
Note that the options `--image=docker:rlehe/warp:latest` and `--volume=<your$SCRATCH>:/home/warp_user/run` are essential and should be copied exactly (**do not** replace `warp_user` or `rlehe` by your username), with the exception of `<your$SCRATCH>`, which should be replaced by the full path to your SCRATCH directory.

Then submit the simulation by typing `sbatch submission_script`.
The progress of the simulation can be seen by typing ```squeue -u `whoami` ```.
