""" 
This script calculates FSC between a map and its copy
obtained by rotating about axis with a known angle
Note that angle is coded as 360/n where n is the
symmetry order.
"""

import emda.emda_methods as em
import numpy as np
from numpy.fft import fftshift, fftn
from emda.core import restools, quaternions, fsc, plotter
from emda.ext.mapfit.utils import get_FRS, set_dim_even

def fsc_per_given_axisangle(arr, unitcell, resol, proshade_axes_list, symorder_list):
    arr = set_dim_even(arr)
    f1 = fftshift(fftn(fftshift(arr)))
    nbin, res_arr, bin_idx = restools.get_resolution_array(unitcell, f1)
    resol = float(resol) * 1.1
    dist = np.sqrt((res_arr - resol) ** 2)
    ibin = np.argmin(dist)
    # cut f1 according to resolution
    f_cut, cbin_idx, cbin = restools.cut_resolution_for_linefit(f1, bin_idx, res_arr, ibin)
    res_arr = res_arr[:cbin]
    fsc_list = []
    for ax, order in zip(proshade_axes_list, symorder_list):
        axis = np.asarray(ax, 'float')
        axis = axis / np.sqrt(np.dot(axis, axis))
        theta = float(360.0 / order)
        q = quaternions.get_quaternion(
                    [[axis[2], axis[1], axis[0]], [theta]])
        rotmat = quaternions.get_RM(q)
        f_rotated = get_FRS(rotmat, f_cut, interp="linear")[:, :, :, 0]
        binfsc = fsc.anytwomaps_fsc_covariance(f_cut, f_rotated, cbin_idx, cbin)[0]
        fsc_list.append(binfsc)
    return res_arr, fsc_list

def main():
    imap = "/Users/ranganaw/MRC/REFMAC/EMD-0965/emd_0965.map"
    proshade_axes_list = [
            [+0.978,    +0.208,    +0.000],
            [+0.970,    -0.242,    -0.000],
            [+0.789,    +0.614,    +0.000],
            [+0.767,    -0.641,    +0.000],
            [+0.000,    +0.000,    +1.000],
            [+0.443,    +0.897,    -0.000],
            [-0.411,    +0.912,    +0.000],
            [+0.017,    +1.000,    -0.000],
            [-0.000,    -0.069,    +0.998],
    ]
    symorder_list = [2, 2, 2, 2, 7, 2, 2, 2, 3]
    resol = 5.0
    uc, arr, orig = em.get_data(imap)
    res_arr, fsc_list = fsc_per_given_axisangle(
                            arr=arr, 
                            unitcell=uc, 
                            resol=resol, 
                            proshade_axes_list=proshade_axes_list, 
                            symorder_list=symorder_list
                            )
    # plot
    labels = ["ax"+str(i+1) for i in range(len(proshade_axes_list))]
    plotter.plot_nlines(res_arr=res_arr,
                        mapname="fscplot.eps",
                        curve_label=labels,
                        fscline=0.,
                        list_arr=fsc_list,
                        )

if __name__=="__main__":
    main()