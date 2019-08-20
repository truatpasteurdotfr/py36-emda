from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import argparse
import sys
from emda.iotools import read_map,write_mrc,write_3d2mtz,write_3d2mtz_refmac,mask_by_value
from emda.restools import get_resArr,remove_edge,create_soft_edged_kernel_pxl
from emda.plotter import plot_nlines
import fcodes
from emda.config import *

# Calculate normalized and FSC-weighted maps
# Date: 2019.07.08
# Modi: 2019.07.11
# Author: Rangana Warshamanage

cmdl_parser = argparse.ArgumentParser(
description='Computes 3D local correlation using maps and atomic models\n')
cmdl_parser.add_argument('-h1', '--half1_map', required=True, help='Input filename for hfmap1')
cmdl_parser.add_argument('-h2', '--half2_map', required=True, help='Input filename for hfmap2')
cmdl_parser.add_argument('-m', '--mask_map', required=False, help='Input filename for mask')
cmdl_parser.add_argument('-r', '--resol_rand', type=np.float32, required=False, help='Resolution(A) threshold for phase randomization')
cmdl_parser.add_argument('-k', '--kernel_size', type=np.int32, required=False, default=5, help='Kernel size (Pixels)')
cmdl_parser.add_argument('-n', '--norm_type', type=np.int, required=False, default=1, choices=[1,2], help='SF Normalisation type')
cmdl_parser.add_argument('-v', '--verbose', default=False,
                         help='Verbose output')

#debug_mode = 1

def get_fsc_true_from_phase_randomize_1d(uc,maplist,resol_rand):
    from emda.fsc_true_from_phase_randomize import get_randomized_sf
    from emda.fsc import halfmaps_fsc_variance
    from emda.restools import get_resolution_array
    from emda.iotools import mask_by_value
    
    half1,half2,mask = maplist
    nbin,res_arr,bin_idx = get_resolution_array(uc,half1)
    idx = np.argmin((res_arr - resol_rand)**2)
    nx,ny,nz = half1.shape
    #maxbin = np.amax(np.array([nx//2,ny//2,nz//2]))
    
    print("FSC_true calculation using phase randomization.")
    # calculate fsc
    fsc_list = []
    # unmasked fsc
    umbin_fsc,_,_,_,_,_ = halfmaps_fsc_variance(np.fft.fftshift(np.fft.fftn(half1)),
                                                    np.fft.fftshift(np.fft.fftn(half2)),
                                                    bin_idx,
                                                    nbin)
    full_fsc_unmasked = 2.0 * umbin_fsc / (1.0 + umbin_fsc)
    fsc_list.append(full_fsc_unmasked)
    # masked fsc; also get normalized SF
    bin_fsc,_,_,_,_,normalized_sf_full = halfmaps_fsc_variance(
                                                            np.fft.fftshift(np.fft.fftn(half1*mask)),
                                                            np.fft.fftshift(np.fft.fftn(half2*mask)),
                                                            bin_idx,
                                                            nbin)
    full_fsc_t = 2.0 * bin_fsc / (1.0 + bin_fsc)
    fsc_list.append(full_fsc_t)
    # randomize phases 
    hf1_randomized = get_randomized_sf(uc,half1,resol_rand)
    hf2_randomized = get_randomized_sf(uc,half2,resol_rand)
    # get phase randomized maps
    randhalf1 = np.real(np.fft.ifftn(np.fft.ifftshift(hf1_randomized)))
    randhalf2 = np.real(np.fft.ifftn(np.fft.ifftshift(hf2_randomized)))
    rbin_fsc,_,_,_,_,_ = halfmaps_fsc_variance(np.fft.fftshift(np.fft.fftn(randhalf1*mask)),
                                                   np.fft.fftshift(np.fft.fftn(randhalf2*mask)),
                                                   bin_idx,
                                                   nbin)
    full_fsc_n = 2.0 * rbin_fsc / (1.0 + rbin_fsc)
    fsc_list.append(full_fsc_n)
    # fsc_true from Richard's formula
    fsc_true = (full_fsc_t - full_fsc_n) / (1 - full_fsc_n)
    fsc_true[:idx+2] = full_fsc_t[:idx+2] # replace fsc_true with fsc_masked_full upto resol_rand_idx + 2 (RELION uses 2)
    fsc_list.append(fsc_true)
    # writing fsc_true into a file
    f = open("fsc_true.txt", "w")
    f.write("Resol  full_fsc_t   full_fsc_true\n")
    for i in range(len(res_arr)):
        f.write("{} {} {}\n".format(res_arr[i],full_fsc_t[i],fsc_true[i]))
    f.close()
    plot_nlines(res_arr,fsc_list,'fsc.eps',["unmasked","fsc_t","fsc_n","fsc_true"])
    # reading fsc_true into 3D grid
    fsc_true_grid = fcodes.read_into_grid(bin_idx,
                               mask_by_value(fsc_true),
                               nbin,
                               nx,ny,nz)
    #fsc_true_grid = fcodes.make_fsc_grid(ful_bin_idx,fsc_true,nbin,nx,ny,nz)
    write_mrc(fsc_true_grid,'corrected_1d_fsc.mrc',uc)
    return normalized_sf_full * np.sqrt(fsc_true_grid)


def get_fsc_true_from_phase_randomize_3d(uc,maplist,resol_rand,kern_sphere):
    from emda.fsc_true_from_phase_randomize import get_randomized_sf
    from emda.fouriersp_corr_3d import get_3d_fouriercorrelation,get_3dtotal_variance
    half1,half2,mask = maplist
    print("FSC_true calculation using phase randomization.")
    # calculate fsc
    # unmasked fsc
    '''hf1_um = np.fft.fftshift(np.fft.fftn(half1))
    hf2_um = np.fft.fftshift(np.fft.fftn(half2))
    full_fsc_grid_unmasked = get_3d_fouriercorrelation_full([hf1_um,hf2_um],kern_sphere)'''
    # masked fsc and normalized_fullmap
    hf1_m = np.fft.fftshift(np.fft.fftn(half1*mask))
    hf2_m = np.fft.fftshift(np.fft.fftn(half2*mask))
    _,full_fsc_grid_t,_ = get_3d_fouriercorrelation(hf1_m,hf2_m,kern_sphere)
    write_mrc(np.real(full_fsc_grid_t),'fsc_t.mrc',uc)
    normalized_sf_full = (0.5 * (hf1_m + hf2_m))/np.sqrt(get_3dtotal_variance(hf1_m,hf2_m,kern_sphere))
    # get phase randomized maps
    randhalf1 = np.real(np.fft.ifftn(np.fft.ifftshift(get_randomized_sf(uc,half1,resol_rand))))
    randhalf2 = np.real(np.fft.ifftn(np.fft.ifftshift(get_randomized_sf(uc,half2,resol_rand))))
    _,full_fsc_grid_n,_ = get_3d_fouriercorrelation(np.fft.fftshift(np.fft.fftn(randhalf1*mask)),
                                                     np.fft.fftshift(np.fft.fftn(randhalf2*mask)),
                                                     kern_sphere)
    write_mrc(np.real(full_fsc_grid_n),'fsc_n.mrc',uc)
    # fsc_true from Richard's formula
    # Negative values not allowed
    fsc_grid_true = mask_by_value(np.real(full_fsc_grid_t) - np.real(full_fsc_grid_n)) / (1 - np.real(full_fsc_grid_n))
    # Only cc > 0.3 are passed.
    fsc_grid_true = mask_by_value(fsc_grid_true, 0.3)
    write_mrc(np.real(fsc_grid_true),'corrected_3d_fsc.mrc',uc)
    return normalized_sf_full * np.sqrt(fsc_grid_true)

def normalized_fsc_weighted_map_nomask_3d(maplist,kern_sphere):
    from emda.fouriersp_corr_3d import get_3dtotal_variance
    half1,half2 = maplist
    print("FSC_true calculation using phase randomization.")
    # calculate fsc
    # unmasked fsc
    '''hf1_um = np.fft.fftshift(np.fft.fftn(half1))
        hf2_um = np.fft.fftshift(np.fft.fftn(half2))
        full_fsc_grid_unmasked = get_3d_fouriercorrelation_full([hf1_um,hf2_um],kern_sphere)'''
    # masked fsc and normalized_fullmap
    hf1_m = np.fft.fftshift(np.fft.fftn(half1))
    hf2_m = np.fft.fftshift(np.fft.fftn(half2))
    full_fsc_grid_t = get_3d_fouriercorrelation_full([hf1_m,hf2_m],kern_sphere)
    f_map = (hf1_m + hf2_m) / 2.0
    totalvar_3d = get_3dtotal_variance([hf1_m,hf2_m],kern_sphere)
    normalized_sf_full = f_map / np.sqrt(totalvar_3d)
    return normalized_sf_full * np.sqrt(full_fsc_grid_t)


def main():
    args = cmdl_parser.parse_args()
    #
    uc,half1,origin = read_map(args.half1_map) # Read half maps
    uc,half2,origin = read_map(args.half2_map)
    if args.mask_map is None: mask = 1
    else: uc,mask,origin = read_map(args.mask_map)
    nx,ny,nz = half1.shape
    fResArr = get_resArr(uc,nx) # Prepare resolution array
    fResArr = get_resArr(uc,nx)
    cut_mask = remove_edge(fResArr,fResArr[-1])
    cc_mask = np.zeros(shape=(nx,ny,nz),dtype='int')
    cx, cy, cz = cut_mask.shape
    dx = (nx - cx)//2
    dy = (ny - cy)//2
    dz = (nz - cz)//2
    cc_mask[dx:dx+cx, dy:dy+cy, dz:dz+cz] = cut_mask
    
    if args.resol_rand is None: resol_rand = fResArr[1]
    else: 
        if args.resol_rand > fResArr[1]:
            print("\nResol. entered is lower than that in 1st shell. I'll use 1st shell resol. instead.")
            resol_rand = fResArr[1]
        else:
            resol_rand = args.resol_rand

    '''# for REFMAC refinement
    f_fullmap = np.fft.fftshift(np.fft.fftn(half1*mask)) + np.fft.fftshift(np.fft.fftn(half2*mask))
    _,_,f_noisevar = get_f_fulcor(half1*mask,half2*mask,kern_sphere)
    # write out noise into mtz format to use with REFMAC
    write_3d2mtz_refmac(uc,f_fullmap,np.sqrt(f_noisevar),outfile='map_and_noise.mtz')
    exit()'''

    if args.norm_type == 2:
        normalized_fscweighted_fullmap = get_fsc_true_from_phase_randomize_1d(uc,[half1,half2,mask],resol_rand)
        outfile = 'norm2fsc-fullmap'
        write_3d2mtz(uc,normalized_fscweighted_fullmap,outfile+'.mtz')
        norm_fsc_fullmap = np.real((np.fft.ifftn(np.fft.ifftshift(normalized_fscweighted_fullmap))))
        write_mrc(norm_fsc_fullmap*cc_mask,outfile+'.mrc',uc,origin)

    elif args.norm_type == 1:
        # Prepare convolution kernel - soft-edge-kernel
        smax = args.kernel_size
        kern_sphere = create_soft_edged_kernel_pxl(smax)
        write_mrc(kern_sphere,'kern_sphere_soft_smax'+str(smax)+'.mrc',uc,origin)
        normalized_fscweighted_fullmap = get_fsc_true_from_phase_randomize_3d(uc,[half1,half2,mask],resol_rand,kern_sphere)
        # without mask
        #normalized_fscweighted_fullmap = normalized_fsc_weighted_map_nomask_3d([half1,half2],kern_sphere)
        # write out maps
        outfile = 'norm1fsc-fullmap-smax'+str(smax)
        write_3d2mtz(uc,normalized_fscweighted_fullmap,outfile+'.mtz')
        norm_fsc_fullmap = np.real((np.fft.ifftn(np.fft.ifftshift(normalized_fscweighted_fullmap))))
        write_mrc(norm_fsc_fullmap*cc_mask,outfile+'.mrc',uc,origin)


if(__name__ == "__main__"):
    main()
