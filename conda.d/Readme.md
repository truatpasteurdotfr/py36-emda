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

but that is not working (chicken/egg issue, emda setup.py requires numpy)
```
Pip subprocess error:
    ERROR: Command errored out with exit status 1:
     command: /home/tru/miniconda3/envs/take2-emda/bin/python -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-_e5klmrj/emda_523466f3b7e64726999272b157a762de/setup.py'"'"'; __file__='"'"'/tmp/pip-install-_e5klmrj/emda_523466f3b7e64726999272b157a762de/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base /tmp/pip-pip-egg-info-4lyy8h32
         cwd: /tmp/pip-install-_e5klmrj/emda_523466f3b7e64726999272b157a762de/
    Complete output (5 lines):
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/tmp/pip-install-_e5klmrj/emda_523466f3b7e64726999272b157a762de/setup.py", line 3, in <module>
        from numpy.distutils.core import setup, Extension
    ModuleNotFoundError: No module named 'numpy'
    ----------------------------------------
WARNING: Discarding https://files.pythonhosted.org/packages/13/a1/3d87396acd62772e026f96f26762913630cdfa214b61c5df645581e4face/emda-1.1.6.post2.tar.gz#sha256=fedd72d11484b87d0f3c3f3460e36d3b57108fd0547a88f1fe8555a8c03b27c4 (from https://pypi.org/simple/emda/). Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
ERROR: Could not find a version that satisfies the requirement emda==1.1.6.post2 (from versions: 1.0, 1.1.0, 1.1.1, 1.1.2, 1.1.2.post1, 1.1.2.post2, 1.1.2.post3, 1.1.2.post4, 1.1.2.post5, 1.1.2.post6, 1.1.2.post7, 1.1.2.post8, 1.1.2.post9, 1.1.2.post10, 1.1.2.post11, 1.1.2.post12, 1.1.2.post13, 1.1.2.post14, 1.1.2.post15, 1.1.2.post16, 1.1.2.post17, 1.1.2.post18, 1.1.2.post19, 1.1.3, 1.1.3.post1, 1.1.3.post2, 1.1.3.post3, 1.1.3.post5, 1.1.3.post6, 1.1.3.post7, 1.1.3.post8, 1.1.3.post9, 1.1.4, 1.1.5, 1.1.6.post1, 1.1.6.post2)
```

# 2 steps installation needed
- conda create --name py36-emda --file 20240604-1600-py36-emda-conda-list--explicit.yml
- conda activate py36-emda
- grep -v emda pip-freese.txt > requirements.txt
- pip install -r requirements.txt
- pip install -r pip-freese.txt

