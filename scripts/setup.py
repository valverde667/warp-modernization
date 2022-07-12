#!/usr/bin/env python
# To use:
#       python setup.py install
#

import os
import sys
import glob

try:
    import setuptools
except:
    raise SystemExit('Setuptools problem')

# --- Write out git versioning information
with open('__version__.py', 'w') as ff:
    ff.write('__origindate__ = "%s"\n'%os.popen('git log --branches=master --remotes=origin -n 1 --pretty=%aD').read().strip())
    ff.write('__localdate__ = "%s"\n'%os.popen('git log -n 1 --pretty=%aD').read().strip())
    ff.write('__hash__ = "%s"\n'%os.popen('git log -n 1 --pretty=%h').read().strip())
    ff.write('__fullhash__ = "%s"\n'%os.popen('git log -n 1 --pretty=%H').read().strip())

setuptools.setup(name = 'warp',
                 version = '4.6',
                 author = 'David P. Grote, Jean-Luc Vay, et. al.',
                 author_email = 'dpgrote@lbl.gov',
                 description = 'Warp PIC accelerator code',
                 long_description = """
Warp is a PIC code designed to model particle accelerators and similar
machines that are space-charge dominated.""",
                 url = 'http://warp.lbl.gov',
                 platforms = 'Linux, Unix, Windows (cygwin), Mac OSX',
                 packages = ['warp', 'warpoptions', 'warp_parallel',
                           'warp.attic',
                           'warp.data_dumping',
                           'warp.data_dumping.openpmd_diag',
                           'warp.diagnostics',
                           'warp.diagnostics.palettes',
                           'warp.envelope',
                           'warp.field_solvers',
                           'warp.field_solvers.laser',
                           'warp.GUI',
                           'warp.init_tools',
                           'warp.lattice',
                           'warp.particles',
                           'warp.run_modes',
                           'warp.utils'],
                 package_dir = {'warp': '.'},
                 package_data = {'warp': ['diagnostics/palettes/*.gs',
                                        'diagnostics/palettes/*.gp',
                                        'particles/aladdin_8.txt']},
      )
