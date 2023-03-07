from ..warp import top
# *********** Constant:
# Physical constants, in one place so they stay consistent
# Values from 2018CODATA
# https://physics.nist.gov/cuu/Constants/index.html
top.amu = 1.66053906660e-27  #  [kg] # Atomic Mass Unit
top.clight = 2.99792458e+8  #  [m/s] # Speed of light in vacuum (exact)
top.echarge = 1.602176634e-19  #  [C] # Proton charge
top.emass = 9.1093837015e-31  #  [kg] # Electron mass
top.eps0 = 8.8541878128e-12  # [F/m] # Permittivity of free space
top.mu0 = 1.25663706212e-6  # [H/m] # Permeability of free space
top.boltzmann = 1.380649e-23  # [J/K]  # Boltzmann's constant
top.avogadro = 6.02214076e23  #        # Avogadro's Number
top.planck = 6.62607015e-34  # [J.s] # Planck's constant

# --- Conversion factor from joules to eV is just echarge
top.jperev = top.echarge

# --- Create python versions of the constants
amu       = top.amu
clight    = top.clight
echarge   = top.echarge
emass     = top.emass
eps0      = top.eps0
jperev    = top.jperev
mu0       = top.mu0
boltzmann = top.boltzmann
avogadro  = top.avogadro
planck    = top.planck

