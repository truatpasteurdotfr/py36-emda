# Author: Rangana Warshamanage
# Created: 2019.06.11

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import fcodes
from .config import *

#debug_mode = 1 # 0: no debug info, 1: debug


def halfmaps_fsc_variance(hf1,hf2,bin_idx,nbin):
    assert hf1.shape == hf2.shape
    nx,ny,nz = hf1.shape
    #maxbin = np.amax(np.array([nx//2,ny//2,nz//2]))
    #nbin,res_arr,bin_idx = fcodes.resolution_grid(uc,debug_mode,maxbin,nx,ny,nz)
    fo,eo,noisevar,signalvar,totalvar,bin_fsc = fcodes.calc_fsc_using_halfmaps(hf1,hf2,bin_idx,nbin,debug_mode,nx,ny,nz)
    '''bin_fsc = bin_fsc[:nbin]
    res_arr = res_arr[:nbin]
    noisevar = noisevar[:nbin]
    signalvar = signalvar[:nbin]
    totalvar = totalvar[:nbin]'''
    return bin_fsc,noisevar,signalvar,totalvar,fo,eo

def anytwomaps_fsc_covariance(f1,f2,bin_idx,nbin):
    assert f1.shape == f2.shape
    nx,ny,nz = f1.shape
    #maxbin = np.amax(np.array([nx//2,ny//2,nz//2]))
    #nbin,res_arr,bin_idx = fcodes.resolution_grid(uc,debug_mode,maxbin,nx,ny,nz)
    f1f2_covar,bin_fsc = fcodes.calc_covar_and_fsc_betwn_anytwomaps(f1,
                                                                f2,
                                                                bin_idx,
                                                                nbin,
                                                                debug_mode,
                                                                nx,ny,nz)
    '''bin_fsc = bin_fsc[:nbin]
    res_arr = res_arr[:nbin]
    f1f2_covar = f1f2_covar[:nbin]'''
    return bin_fsc,f1f2_covar




    
    


