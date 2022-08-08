# Overlay using half data
# This code align a pair of half maps to another pair of those.
# Overlay parameters are estimated using fullmap of each set.

import numpy as np
import fcodes_fast
import emda.emda_methods as em
from numpy.fft import fftn, fftshift, ifftshift, ifftn
from emda.ext.overlay import EmmapOverlay, run_fit
from emda import core
from emda.ext.mapfit import utils as maputils
from emda.core import quaternions, restools
import argparse

""" cmdl_parser = argparse.ArgumentParser(
description='Overlay using halfdata\n')
cmdl_parser.add_argument("--maplist", required=True, nargs="+", type=str, help="maplist for overlay")
cmdl_parser.add_argument(
    "--msk",
    required=False,
    default=None,
    nargs="+",
    type=str,
    help="masklist for overlay",
)
cmdl_parser.add_argument(
    "--tra",
    required=False,
    default=[0.0, 0.0, 0.0],
    nargs="+",
    type=float,
    help="list of translation vectors. default=[0.0, 0.0, 0.0]",
)
cmdl_parser.add_argument(
    "--rot",
    required=False,
    default=[0.0],
    nargs="+",
    type=float,
    help="list of rotations in deg. default=0.0",
)
cmdl_parser.add_argument(
    "--axr",
    required=False,
    default=[1, 0, 0],
    nargs="+",
    type=int,
    help="list of rotation axes. default=[1,0,0]",
)
cmdl_parser.add_argument(
    "--ncy",
    required=False,
    default=100,
    type=int,
    help="number of fitting cycles. default=5",
)
cmdl_parser.add_argument(
    "--fitres",
    required=False,
    default=0.0,
    type=float,
    help="last resolution to use for fitting. default= 0.0 A",
) """


def proshade_overlay(map1, map2, fitresol=7.0):
    import proshade

    ps                = proshade.ProSHADE_settings()
    ps.task           = proshade.OverlayMap
    ps.verbose        = -1                                                              
    ps.setResolution  (fitresol)                                                           
    ps.addStructure   (map1)
    ps.addStructure   (map2)
    rn                = proshade.ProSHADE_run(ps)
    eulerAngles       = rn.getEulerAngles()
    rotMatrix    = rn.getOptimalRotMat()
    return rotMatrix




if __name__ == "__main__":

    half_pair1 = [
                "/Users/ranganaw/MRC/REFMAC/Simone/Data/HalfMaps/Wt/cryosparc_P4_J14_009_volume_map_half_A.mrc",
                "/Users/ranganaw/MRC/REFMAC/Simone/Data/HalfMaps/Wt/cryosparc_P4_J14_009_volume_map_half_B.mrc"
                ]
    half_pair2 = [
                "/Users/ranganaw/MRC/REFMAC/Simone/Analysis/Magnification_correction_using_PDB/pent_half-A.mrc",
                "/Users/ranganaw/MRC/REFMAC/Simone/Analysis/Magnification_correction_using_PDB/pent_half-B.mrc"
                ]

    #0. if initial positions are too far, get the rotation matrix from proshade
    """ rotmat = proshade_overlay(map1=half_pair1[0], 
                            map2=half_pair2[0], fitresol=7) """


    #1. read in half maps
    uc, arr1, origin = em.get_data(half_pair1[0])
    uc, arr2, origin = em.get_data(half_pair1[1])
    tpixsize = [uc[i]/shape for i, shape in enumerate(arr1.shape)]

    p1_hf1 = fftshift(fftn(fftshift(arr1)))
    p1_hf2 = fftshift(fftn(fftshift(arr2)))
    p1_f = (p1_hf1 + p1_hf2) / 2


    uc2, arr3, _ = em.get_data(half_pair2[0])
    _, arr4, _ = em.get_data(half_pair2[1])
    cpixsize = [uc2[i]/shape for i, shape in enumerate(arr3.shape)]

    #2. resample p2 on p1
    newarr3 = em.resample_data(curnt_pix=cpixsize, 
                            targt_pix=tpixsize,
                            targt_dim=arr1.shape,
                            arr = arr3)
    newarr4 = em.resample_data(curnt_pix=cpixsize, 
                            targt_pix=tpixsize,
                            targt_dim=arr1.shape,
                            arr = arr4)
    p2_hf1 = fftshift(fftn(fftshift(newarr3)))
    p2_hf2 = fftshift(fftn(fftshift(newarr4)))
    p2_f = (p2_hf1 + p2_hf2) / 2

    #3. estimate fitting parameters for p2
    emmap1 = EmmapOverlay(map_list=[])
    emmap1.pixsize = [uc[i]/shape for i, shape in enumerate(arr1.shape)]
    emmap1.map_dim = [shape for shape in arr1.shape]
    emmap1.map_origin = origin
    emmap1.map_unit_cell = uc
    emmap1.com = True
    emmap1.fhf_lst = [p1_f, p2_f]
    emmap1.calc_fsc_from_maps()
    rotmat_list = []
    trans_list = []
    #if not rotmat: rotmat = np.identity(3, 'float')
    #rotmat = np.identity(3, 'float')
    #q_init = quaternions.rot2quart(rm=rotmat)
    q_init = quaternions.get_quaternion([[1,0,0],[0.]])
    t_init = [0., 0., 0.]
    t, q_final = run_fit(
            emmap1=emmap1,
            rotmat=core.quaternions.get_RM(q_init),
            t=[itm / emmap1.pixsize[i] for i, itm in enumerate(t_init)],
            ncycles=100,
            ifit=1,
            fitres=None,
        )
    rotmat = core.quaternions.get_RM(q_final)

    #4. output rotated and translated maps
    fmaps = np.stack((emmap1.fo_lst[1], p2_hf1, p2_hf2), axis=-1)
    f_static = emmap1.fo_lst[0]
    nz, ny, nx = f_static.shape

    f1f2_fsc_unaligned = core.fsc.anytwomaps_fsc_covariance(
        f_static, emmap1.fo_lst[1], emmap1.bin_idx, emmap1.nbin
    )[0]

    #output static_fullmap
    data2write = np.real(ifftshift(ifftn(ifftshift(f_static))))
    core.iotools.write_mrc(
        data2write,
        'static_fullmap.mrc',
        emmap1.map_unit_cell,
    )

    frs = maputils.get_FRS(rotmat, fmaps, interp="cubic")
    st, _, _, _ = fcodes_fast.get_st(nz, ny, nx, t)
    outnames = ['fitted_fullmap.mrc', 'fitted_halfmap1.mrc', 'fitted_halfmap2.mrc']
    for i in range(frs.shape[3]):
        frt = frs[:,:,:,i] * st
        if i == 0:
            f1f2_fsc = core.fsc.anytwomaps_fsc_covariance(f_static, frt, emmap1.bin_idx, emmap1.nbin)[0]
            core.plotter.plot_nlines(
                emmap1.res_arr,
                [f1f2_fsc_unaligned[: emmap1.nbin], f1f2_fsc[: emmap1.nbin]],
                "fsc.eps",
                ["FSC before", "FSC after"],
            )
        data2write = np.real(ifftshift(ifftn(ifftshift(frt))))     
        core.iotools.write_mrc(
            data2write,
            outnames[i],
            emmap1.map_unit_cell,
        )