from emda.mapfit import map_Class
from emda.mapfit import emfit_Class
from emda.config import *

def main(maplist, ncycles=10, t_init=[0.0, 0.0, 0.0], theta_init=0.0, smax=6):
    emmap1 = map_Class.EmMap(maplist)
    emmap1.load_maps()
    emmap1.calc_fsc_variance_from_halfdata()
    emmap1.make_data_for_fit(smax)
    fit = emfit_Class.emFit(emmap1)
    fit.minimizer(ncycles, t_init, theta_init)

if (__name__ == "__main__"):
    main()

