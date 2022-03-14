from __future__ import division, absolute_import, print_function
import setuptools
from numpy.distutils.core import setup, Extension

ex1 = Extension(name = 'fcodes_fast',
                sources = ['emda/fcodes_fast.f90'])

version = {}
with open("./emda/config.py") as fp:
    exec(fp.read(), version)

setup(name='emda',
    version=version['__version__'],
    description= 'Electron Microscopy map and model manipulation tools',
    long_description='''\
        Python library for Electron Microscopy Data Analysis (EMDA). 
        EMDA supports MRC/CCP4.MAP and MTZ files as well as PDB and mmcif files. 
        EMDA offeres a range of functions for downstream data analysis and
        model building. Please refer emda.readthedocs.io for more information.''',
    url='https://www2.mrc-lmb.cam.ac.uk/groups/murshudov/content/emda/emda.html',
    author='Rangana Warshamanage, Garib N. Murshudov',
    author_email='ranganaw@mrc-lmb.cam.ac.uk, garib@mrc-lmb.cam.ac.uk',
    license='MPL-2.0',
    packages=setuptools.find_packages(),
    ext_modules =[ex1],
    install_requires=['pandas>=0.23.4','mrcfile','matplotlib','numpy','scipy','gemmi','servalcat', 'proshade'],
    test_suite='emda.tests',
    entry_points={
      'console_scripts': [
          'emda_test = emda.emda_test:main',
          'emda_test_exhaust = emda.emda_test_exhaust:main',
          'emda = emda.emda_cmd_caller:main',
                          ],
      },
    zip_safe= False)
