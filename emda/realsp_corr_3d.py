from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sys
import argparse
from emda.iotools import read_map,write_mrc,read_mtz
from emda.maptools import mtz2map
#from .mtz2mrc import mtz2mrc,get_f,get_f_gemmi
from emda.restools import get_resArr,remove_edge,create_soft_edged_kernel_pxl
from emda.config import *

cmdl_parser = argparse.ArgumentParser(
                                      description='Computes 3D local correlation using maps\n')
cmdl_parser.add_argument('-h1', '--half1_map', required=True, help='Input filename for hfmap1')
cmdl_parser.add_argument('-h2', '--half2_map', required=True, help='Input filename for hfmap2')
cmdl_parser.add_argument('-ml', '--model', required=False, help='Input model MAP/MTZ file')
#cmdl_parser.add_argument('-m', '--mask_map', required=False, help='Input filename for mask')
cmdl_parser.add_argument('-k', '--kernel_size', type=np.int32, required=False, default=5, help='Kernel size (Pixels)')
cmdl_parser.add_argument('-v', '--verbose', default=False,
                         help='Verbose output')

def get_fullmapcor(hfmap_corr):
    import numpy.ma as ma
    hfmap_corr_ma = ma.masked_less_equal(hfmap_corr,0.0)
    ccf_ma = 2 * hfmap_corr_ma / (1.0 + hfmap_corr_ma)
    ccf = ccf_ma.filled(0.0)
    return ccf

def cc_twosimilarmaps(ccmap12,ccmap1,ccmap2,uc,origin):
    import numpy.ma as ma
    ccmap1_ma = ma.masked_less_equal(ccmap1,0.0)
    ccmap2_ma = ma.masked_less_equal(ccmap2,0.0)
    ccmap12_ma = ma.masked_less_equal(ccmap12,0.0)
    cc12_ma = ccmap12_ma * np.sqrt(ccmap1_ma) * np.sqrt(ccmap2_ma)
    cc12 = cc12_ma.filled(0.0)
    write_mrc(cc12 * cc_mask,'map12_realspacecc.mrc',uc,origin)
    return cc12

def truemap_model_cc(mapmodelcc,fullmapcc):
    import numpy.ma as ma
    from emda.iotools import mask_by_value_greater
    mapmodelcc_ma = ma.masked_less_equal(mapmodelcc,0.3) # To prevent large numbers in truemapmodelcc
    fullmapcc_ma = ma.masked_less_equal(fullmapcc,0.3) # To prevent large numbers in truemapmodelcc
    #truemapmodelcc = mask_by_value_greater(mapmodelcc / np.sqrt(fullmapcc),masking_value=1.0)
    truemapmodelcc_ma = mapmodelcc_ma / np.sqrt(fullmapcc_ma)
    truemapmodelcc = mask_by_value_greater(truemapmodelcc_ma.filled(0.0),masking_value=1.0)
    #write_mrc(truemapmodelcc * cc_mask,'truemapmodel_fancg.mrc',uc,origin)
    return truemapmodelcc

def get_3d_realspcorrelation(half1,half2,kern_sphere):
    # Full map correlation using FFT convolve
    import scipy.signal
    import numpy.ma as ma
    loc3_A = scipy.signal.fftconvolve(half1, kern_sphere, "same")
    loc3_A2 = scipy.signal.fftconvolve(half1 * half1, kern_sphere, "same")
    loc3_B = scipy.signal.fftconvolve(half2, kern_sphere, "same")
    loc3_B2 = scipy.signal.fftconvolve(half2 * half2, kern_sphere, "same")
    loc3_AB = scipy.signal.fftconvolve(half1 * half2, kern_sphere, "same")

    cov3_AB = loc3_AB - loc3_A * loc3_B
    var3_A = loc3_A2 - loc3_A**2
    var3_B = loc3_B2 - loc3_B**2
    cov3_AB_ma = ma.masked_less_equal(cov3_AB,0.0)
    var3_A_ma = ma.masked_less_equal(var3_A,0.0)
    var3_B_ma = ma.masked_less_equal(var3_B,0.0)
    cc_realsp_ma = cov3_AB_ma / np.sqrt(var3_A_ma * var3_B_ma)
    halfmaps_cc = cc_realsp_ma.filled(0.0)
    fullmap_cc = 2 * halfmaps_cc / (1.0 + halfmaps_cc)
    return halfmaps_cc,fullmap_cc

def get_3d_realspmapmodelcorrelation(map,model,kern_sphere):
    mapmodel_cc,_ = get_3d_realspcorrelation(map,model,kern_sphere)
    return mapmodel_cc


def create_mask_from_rscc(fullmapcc,threshold=0.3):
    #from math import sqrt,cos,sin
    import numpy.ma as ma
    '''fullmapcc_low_thresholded = fullmapcc > threshold
    mask = fullmapcc_low_thresholded.astype('int')
    mask = mask/np.count_nonzero(mask)'''
    # soft edged mask
    '''r1 = 0.95
    r0 = 0.91
    inner = fullmapcc >= r1
    outer = fullmapcc >= r0
    norm = 1/np.count_nonzero(outer)
    norm  = 1
    inner_norm = inner * norm
    dist1 = (outer * fullmapcc)
    dist2 = (inner * fullmapcc)
    dist = (dist1 - dist2) * fullmapcc
    dist_ma = ma.masked_less_equal(dist,0.0)
    # cosine decay
    soft_ma = (norm * (1 + np.cos(np.pi * (dist_ma - r0)/(r1 - r0))))/2
    soft = soft_ma.filled(0.0)
    mask = (soft + inner_norm)'''
    r5 = 0.5
    r4 = r5 - 0.01
    r3 = r4 - 0.01
    r2 = r3 - 0.01
    r1 = r2 - 0.01
    r0 = r1 - 0.01
    m5 = fullmapcc >= r5
    m4 = fullmapcc >= r4
    m3 = fullmapcc >= r3
    m2 = fullmapcc >= r2
    m1 = fullmapcc >= r1
    m0 = fullmapcc >= r0
    d5 = m5 * fullmapcc
    d4 = m4 * fullmapcc - d5
    d3 = m3 * fullmapcc - (d5 + d4)
    d2 = m2 * fullmapcc - (d3 + d5 + d4)
    d1 = m1 * fullmapcc - (d2 + d3 + d5 + d4)
    d0 = m0 * fullmapcc - (d1 + d2 + d3 + d5 + d4)
    d5[d5 >= r5] = 1
    d4[d4 >= r4] = r4
    d3[d3 >= r3] = r3
    d2[d2 >= r2] = r2
    d1[d1 >= r1] = r1
    d0[d0 >= r0] = r0
    norm = 1/np.count_nonzero(m0)
    norm = 1
    #mask = (d5 + d4 + d3 + d2 + d1 + d0) * norm
    xx = (d4 + d3 + d2 + d1 + d0)
    dist_ma = ma.masked_less_equal(xx,0.0)
    soft_ma = (norm * (1 + np.cos(np.pi * (dist_ma - r0)/(r5 - r0))))/2
    soft = soft_ma.filled(0.0)
    mask = d5 + soft
    return mask

def main():
    args = cmdl_parser.parse_args()
    # Halfmaps and fullmap correlation calculation
    print('Calculating 3D correlation between half maps and fullmap. Please wait...')
    uc,half1,origin = read_map(args.half1_map)
    uc,half2,origin = read_map(args.half2_map)
    nx,ny,nz = half1.shape
    fResArr = get_resArr(uc,nx)

    cut_mask = remove_edge(fResArr,fResArr[-1])
    cc_mask = np.zeros(shape=(nx,ny,nz),dtype='int')
    cx, cy, cz = cut_mask.shape
    dx = (nx - cx)//2
    dy = (ny - cy)//2
    dz = (nz - cz)//2
    print(dx,dy,dz)
    cc_mask[dx:dx+cx, dy:dy+cy, dz:dz+cz] = cut_mask

    # Creating soft-edged mask
    kern_sphere_soft = create_soft_edged_kernel_pxl(args.kernel_size) # sphere with radius of n pixles
    write_mrc(kern_sphere_soft,'kern_sphere_soft_smax'+str(args.kernel_size)+'.mrc',uc,origin)

    # Real space correlation maps
    halfmapscc, fullmapcc = get_3d_realspcorrelation(half1,half2,kern_sphere_soft)
    write_mrc(halfmapscc * cc_mask,'hfmaps_3dcc_smax'+str(args.kernel_size)+'.mrc',uc,origin)
    write_mrc(fullmapcc * cc_mask,'fullmap_3dcc_smax'+str(args.kernel_size)+'.mrc',uc,origin)
    
    # Create mask using correlation
    #mask = create_mask_from_rscc(fullmapcc*cc_mask,0.91)
    #write_mrc(mask * cc_mask,'mask.mrc',uc,origin)

    # Map-model correlation
    if args.model is not None:
        fullmap = (half1 + half2) / 2.0
        if args.model.lower().endswith(('.mrcs', '.mrc', '.map')):
            _,model,_ = read_map(args.model)
        elif args.model.lower().endswith('.mtz'):
            model = mtz2map(args.model,(nx,ny,nz))
        mapmodelcc = get_3d_realspmapmodelcorrelation(fullmap,model,kern_sphere_soft)
        write_mrc(mapmodelcc * cc_mask,'fullmapModel_3dcc.mrc',uc,origin)
        print('Map-model correlation calculated!')
        # truemap-model correlation
        truemapmodelcc = truemap_model_cc(mapmodelcc,fullmapcc)
        write_mrc(truemapmodelcc * cc_mask,'truemapModel_3dcc.mrc',uc,origin)

    # Two maps with and without ligand (or similar situation e.g. two conformations)
    #ccmap1 = loccc_realspace(half1,half2,kern_sphere)
    #ccmap2 = loccc_realspace(half3,half4,kern_sphere)
    #full1 = (half1 + half2) / 2.0
    #full2 = (half3 + half4) / 2.0
    #ccmap12 = loccc_realspace(full1,full2,kern_sphere)
    #cc12 = cc_twosimilarmaps(ccmap12,ccmap1,ccmap2,uc,origin)'''

if(__name__ == "__main__"):
    main()
