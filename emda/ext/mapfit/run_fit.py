"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# fitting function
from __future__ import absolute_import, division, print_function, unicode_literals
from emda import core, ext
import numpy as np
import fcodes_fast


def fsc_between_static_and_transfomed_map(
    staticmap, movingmap, bin_idx, rm, t, cell, nbin
):
    from emda.ext.mapfit import utils

    nx, ny, nz = staticmap.shape
    st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
    frt_full = utils.get_FRS(rm, movingmap * st, interp="cubic")[:, :, :, 0]
    f1f2_fsc, _ = core.fsc.anytwomaps_fsc_covariance(staticmap, frt_full, bin_idx, nbin)
    return f1f2_fsc


def get_ibin(bin_fsc):
    bin_fsc = bin_fsc[bin_fsc > 0.1]
    dist = np.sqrt((bin_fsc - 0.4) ** 2)
    ibin = np.argmin(dist) + 1
    if ibin % 2 != 0:
        ibin = ibin - 1
    ibin = min([len(dist), ibin])
    return ibin


def run_fit(emmap1, smax, rotmat, t, slf, ncycles, ifit, fobj, interp, dfs_full=None):
    from emda.ext.mapfit import emfit_class, frequency_marching, interp_derivatives

    fsc_lst = []
    for i in range(5):
        if i == 0:
            smax = smax  # A
            if emmap1.res_arr[0] < smax:
                ibin = 2
                print("Fitting starts at ", emmap1.res_arr[ibin], " (A) instead!")
            else:
                dist = np.sqrt((emmap1.res_arr - smax) ** 2)
                ibin = np.argmin(dist) + 1
                if ibin % 2 != 0:
                    ibin = ibin - 1
                ibin = min([len(dist), ibin])
                print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
            ibin_old = ibin
            f1f2_fsc = fsc_between_static_and_transfomed_map(
                staticmap=emmap1.fo_lst[0],
                movingmap=emmap1.fo_lst[ifit],
                bin_idx=emmap1.bin_idx,
                rm=rotmat,
                t=t,
                cell=emmap1.map_unit_cell,
                nbin=emmap1.nbin,
            )
            fsc_lst.append(f1f2_fsc)
        else:
            # Apply initial rotation and translation to calculate fsc
            f1f2_fsc = fsc_between_static_and_transfomed_map(
                emmap1.fo_lst[0],
                emmap1.fo_lst[ifit],
                emmap1.bin_idx,
                rotmat,
                t,
                emmap1.map_unit_cell,
                emmap1.nbin,
            )
            ibin = get_ibin(f1f2_fsc)
            if ibin_old == ibin:
                fsc_lst.append(f1f2_fsc)
                fobj.write("\n")
                fobj.write("FSC between static and moving maps\n")
                fobj.write("\n")
                fobj.write("bin#\n")
                fobj.write("resolution (A)\n")
                fobj.write("start FSC\n")
                fobj.write("end FSC\n")
                fobj.write("\n")
                print("\n***FSC between static and moving maps***\n")
                print("bin#     resolution(A)      start-FSC     end-FSC\n")
                for j in range(len(emmap1.res_arr)):
                    # print(emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[1][j])
                    print(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                            j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[1][j]
                        )
                    )
                    fobj.write(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}\n".format(
                            j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[1][j]
                        )
                    )
                break
            else:
                ibin_old = ibin
        if ibin == 0:
            print("Cannot find a solution! Stopping now...")
            exit()
        static_cutmap, cBIdx, cbin = frequency_marching.frequency_marching(
            emmap1.eo_lst[0], emmap1.bin_idx, emmap1.res_arr, bmax=ibin
        )
        moving_cutmap, _, _ = frequency_marching.frequency_marching(
            emmap1.eo_lst[ifit], emmap1.bin_idx, emmap1.res_arr, bmax=ibin
        )
        moving_cutmap_f1, _, _ = frequency_marching.frequency_marching(
            emmap1.fo_lst[ifit], emmap1.bin_idx, emmap1.res_arr, bmax=ibin
        )
        if dfs_full is not None:
            # cut dfs_full for current size
            dfs = interp_derivatives.cut_dfs4interp(dfs_full, cbin)
            #
        else:
            dfs = None
        assert static_cutmap.shape == moving_cutmap.shape
        emmap1.cfo_lst = [moving_cutmap_f1]
        emmap1.ceo_lst = [static_cutmap, moving_cutmap]
        emmap1.cbin_idx = cBIdx
        emmap1.cdim = moving_cutmap.shape
        emmap1.cbin = cbin
        fit = emfit_class.EmFit(emmap1, interp=interp, dfs=dfs)
        if ibin < slf or slf == 0:
            slf = ibin
        slf = min([ibin, slf])
        fit.minimizer(ncycles, t, rotmat, smax_lf=slf, fobj=fobj)
        ncycles = ncycles  # tweaking this you can change later # cycles
        t = fit.t_accum
        rotmat = fit.rotmat
    return rotmat, t
