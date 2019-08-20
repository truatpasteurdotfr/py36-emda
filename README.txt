EMDA version 1.1
===================

EMDA is a Python and Fortran based module for Electron Microscopy map and model
manipulations. EMDA was developed in Python3 environment, but it should run in Python2 as 
well.


Dependencies
===================

EMDA has several dependencies in addition to general python modules (e.g. Numpy, Scipy).
They are 
pandas
gemmi
mrcfile
matplotlib
All these dependencies will be automatically checked and installed, if necessary during
EMDA installation.


Installing EMDA version 1.1  
===============================

For installation you may need administrator permissions, 
please consult your system administrator if needed.


Installing from binaries:
-----------------------------------------
EMDA can be easily installed using the Python package manager (pip) by executing
pip install emda==1.1
however, installing from binaries is discouraged because this may lead to missing some
important shared libraries depending on the targeted hardware and operating systems.


Installing from source
--------------------------------------------
This is the recommended method of installation of EMDA. 
If you downloaded EMDA tar file from xxx, 
follow these steps:
 - uncompress the emda-1.1.tar.gz file 
 - go to emda-1.1 directory 
 - type 'python setup.py bdist_wheel' 
 - type 'pip install dist/emda-1.1-xxx.whl'

All necessary files will be installed under pythonx.x/site-packages/
e.g.: /Users/ranganaw/anaconda3/lib/python3.6/site-packages


License
=======

EMDA-1.1 comes under Mozilla Public License Version 2.0 licence.
Please look LICENSE.txt for more details.


Citations
=========

Please cite EMDA original publication. Will be available soon.

Acknowledgments
===============

Wellcome Trust- Validation tools for Cryo EM grant (xxxxx)

This README file was last time modified on 19.08.2019

