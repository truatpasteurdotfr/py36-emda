"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from emda.config import *

#debug_mode = 1

def Bf_linefit(res_arr,
               binSgnlVar,
               low_res_cutoff,
               high_res_cutoff,
               scale=1):
    from scipy import stats
    import numpy as np
    import numpy.ma as ma
    prediSigVar = np.zeros(len(res_arr), dtype='float')
    #print(res_arr)
    dist_low = np.sqrt((res_arr - low_res_cutoff)**2)
    lb = np.argmin(dist_low) + 1
    dist_high = np.sqrt((res_arr - high_res_cutoff)**2)
    ub = np.argmin(dist_high) + 1
    #print(lb,ub)
    res_arr_ma = ma.masked_equal(res_arr,0.0)
    s = 1/res_arr_ma
    x_full = (s * s) / 2.0
    x_full = x_full.filled(0.0)
    x = x_full[lb:ub]
    y = np.log(binSgnlVar[lb:ub]) # Linear range where the signal is above the noise level
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    Bfac = slope
    #print('B factor = ', Bfac, 'Intercept = ', intercept)
    ln_s = -1.0*abs(Bfac) * x_full + intercept
    prediSigVar = scale * np.exp(ln_s)
    return Bfac,prediSigVar

def estimate_map_resol(hfmap1,hfmap2):
    from emda.iotools import read_mrc
    import emda.fsc
    uc,hf1,origin = read_mrc(hfmap1)
    uc,hf2,origin = read_mrc(hfmap2)
    res_arr,bin_fsc,_,sigvar,_ = fsc.halfmaps_fsc_variance(uc,hf1,hf2)
    bin_fsc = bin_fsc[bin_fsc > 0.1]
    res_arr = res_arr[bin_fsc > 0.1]
    dist = np.sqrt((bin_fsc - 0.143)**2)
    map_resol = res_arr[np.argmin(dist)]
    return map_resol,sigvar,res_arr

def get_biso_from_model(mmcif_file):
    from emda.iotools import read_mmcif
    _,_,_,_,Biso_np = read_mmcif(mmcif_file)
    Biso = np.median(Biso_np)
    return Biso

def get_biso_from_map(halfmap1,halfmap2):
    from emda.plotter import plot_nlines_log
    map_resol,sig_var,res_arr = estimate_map_resol(halfmap1,halfmap2)
    plot_nlines_log(res_arr,[sig_var],["Signal Variance"])
    low_res_cutoff = np.float32(input('Enter low resolution cutoff: '))
    high_res_cutoff = np.float32(input('Enter high resolution cutoff: '))
    Biso,preditcted_signal = Bf_linefit(res_arr,sig_var,low_res_cutoff,high_res_cutoff)
    plot_nlines_log(res_arr,[sig_var,preditcted_signal],["Signal Variance","Predicted SV"],'Predicted.eps')
    return Biso

def apply_bfactor_to_map(mapname,bf_arr):
    import emda.iotools
    import fcodes
    uc,ar1,origin = iotools.read_map(mapname)
    hf1 = np.fft.fftshift(np.fft.fftn(ar1))
    nx, ny, nz = ar1.shape
    nbf = len(bf_arr)
    all_mapout = fcodes.apply_bfactor_to_map(hf1,bf_arr,uc,debug_mode,nx,ny,nz,nbf)
    return all_mapout

def map2mtz(mapname,mtzname='map2mtz'):
    from emda.iotools import read_map,write_3d2mtz
    import numpy as np
    uc,ar1,origin = read_map(mapname)
    hf1 = np.fft.fftshift(np.fft.fftn(ar1))
    write_3d2mtz(uc,hf1,outfile=mtzname+'.mtz')

def mtz2map(mtzname,map_size):
    from emda.iotools import read_map,read_mtz,write_mrc
    _,dataframe = read_mtz(mtzname)
    h = dataframe['H'].astype('int')
    k = dataframe['K'].astype('int')
    l = dataframe['L'].astype('int')
    f = dataframe['Fout0'] * np.exp(np.pi * 1j * dataframe['Pout0']/180.0)
    nx, ny, nz = map_size
    f3d = fcodes.mtz2_3d(h,k,l,f,nx,ny,nz,len(f))
    data2write = np.real((np.fft.ifftn(np.fft.ifftshift(f3d)))) # map center NOT moved
    #data2write = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f3d)))) # map_center MOVED by 0.5,0.5,0.5
    #write_mrc(data2write,"mtz2mrc.mrc",uc)
    return data2write




