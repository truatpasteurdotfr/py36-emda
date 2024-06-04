# logbook of manual installation:
-    enable your conda environment
-    create a emda environment (thus aoid breaking everything that you have been using so far)
-    activate this new environment
-    install the required packages
-    pip install emda 
-    fails because emda requires a fortran compiler
-    use the default gcc module (which include gfortran)
-    re-run `pip install emda`
-    test with `emda_test` 
-    save your conda recipe to be able to replay it (elsewhere/later). I am using `conda-yaml.sh`, ymmv!


```
[tru@maestro-submit ~]$ source bin/enable-miniconda3-HOME.sh #or whatever
[tru@maestro-submit ~]$ conda env list
...
[tru@maestro-submit ~]$ conda create -n py36-emda python=3.6
...
[tru@maestro-submit ~]$ conda activate py36-emda
(py36-emda) [tru@maestro-submit ~]$ pip install pandas gemmi mrcfile matplotlib
...
(py36-emda) [tru@maestro-submit ~]$ pip install emda
...
    warning: build_ext: f77_compiler=None is not available.
 
    building 'fcodes_fast' extension
    error: extension 'fcodes_fast' has Fortran sources but no Fortran compiler found
...
(py36-emda) [tru@maestro-submit ~]$ module av gcc
--------------------------------------------------------------------------------------------- /opt/gensoft/devmodules ---------------------------------------------------------------------------------------------
gcc/8.4.0  gcc/9.2.0(default)  gcc/9.3.0  gcc/9.5.0  gcc/10.1.0  gcc/10.4.0  gcc/11.3.0  gcc/12.3.0
(py36-emda) [tru@maestro-submit ~]$ module add gcc/9.2.0
(py36-emda) [tru@maestro-submit ~]$ pip install emda
...
Successfully built emda
Installing collected packages: emda
Successfully installed emda-1.1.6.post2
Installing collected packages: emda
Successfully installed emda-1.1.6.post2
(py36-emda) [tru@maestro-submit ~]$ emda_test
iotools test ... Passed
maptools test ... Passed
restools test ... Passed
 fcodes test ... Passed
```

replay (make sur you have a gfortan in your PATH):
`conda env create -n take2-emda --file 20240604-1600-py36-emda-conda-env-export.yml`
