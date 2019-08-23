from __future__ import division, absolute_import, print_function
import setuptools
from numpy.distutils.core import setup, Extension

ex1 = Extension(name = 'fcodes',
                sources = ['emda/fcodes.f90'])

setup(name= 'emda',
    version= '1.1.0',
    description= 'Electron Microscopy map and model manipulation tools',
    long_description='''\
        Python library for Electron Microscopy map and model
        manipulations. To work with MRC/CCP4.MAP and MTZ files. Map to map
        fitting and average/difference map calculation. Map and map-model validation
        using 3D correlations.''',
    url= None,
    author= 'Rangana Warshamanage, Garib N. Murshudov',
    author_email= 'ranganaw@mrc-lmb.cam.ac.uk, garib@mrc-lmb.cam.ac.uk',
    license= 'MPL-2.0',
    packages= ['emda','emda.mapfit'],
    ext_modules = [ex1],
    install_requires=['pandas','mrcfile','matplotlib'],
    dependency_links=['git+https://github.com/project-gemmi/gemmi.git'],
    test_suite='emda.tests',
    entry_points={
      'console_scripts': [
          'emda_test = emda.emda_test:main',
          'emda_map2mtz = emda.mrc2mtz:main',
          'emda_real_space_corr = emda.realsp_corr_3d:main',
          'emda_fourier_space_corr = emda.fouriersp_corr_3d:main',
          'emda_truefsc_from_phasr = emda.fsc_true_from_phase_randomize:main',
          'emda_normalised_fscw_maps = emda.calc_normalised_and_fsc_weighted_maps:main',
                          ],
      },
    zip_safe= False)
