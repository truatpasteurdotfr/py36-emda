"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from timeit import default_timer as timer
import sys
from emda.iotools import read_map,read_mtz,run_refmac_sfcalc,pdb2mmcif
from emda.maptools import mtz2map
from emda.map_resol_bfac import get_biso_from_model
from emda.restools import get_resolution_array
#import fcodes
import emda.fsc
from emda.plotter import *
import argparse
from emda.config import *

#debug_mode = 1 # 0: no debug info, 1: debug

cmdl_parser = argparse.ArgumentParser(description='Computes FSC between model and map\n')
cmdl_parser.add_argument('-h1', '--half1_map', required=True, help='Input filename for hfmap1')
cmdl_parser.add_argument('-h2', '--half2_map', required=True, help='Input filename for hfmap2')
cmdl_parser.add_argument('-f', '--full_map', required=True, help='Input filename for fullmap')
cmdl_parser.add_argument('-af', '--modelf_pdb', required=True, help='atomic model refined with fullmap')
cmdl_parser.add_argument('-a1', '--model1_pdb', required=True, help='atomic model refined with halfmap')
cmdl_parser.add_argument('-r', '--model_resol', type=np.float32, required=True, help='Resolution (A) for map calculation fro model')
cmdl_parser.add_argument('-v', '--verbose', default=False,
                         help='Verbose output')

def calculate_modelmap(model_pdb,dim,args):
    model_bfac = get_biso_from_model('out.cif')
    run_refmac_sfcalc(model_pdb,args.model_resol,model_bfac)
    return np.fft.fftshift(np.fft.fftn(mtz2map('sfcalc_from_crd.mtz',dim)))

def main():
    args = cmdl_parser.parse_args()
    # Read half maps
    uc, ar1, origin = read_map(args.half1_map)
    f_hf1 = np.fft.fftshift(np.fft.fftn(ar1))
    uc,ar2, origin = read_map(args.half2_map)
    f_hf2 = np.fft.fftshift(np.fft.fftn(ar2))
    uc,ar3, origin = read_map(args.full_map)
    f_ful = np.fft.fftshift(np.fft.fftn(ar3))
    dim = f_hf1.shape
    # Calculate map-model correlation
    pdb2mmcif(args.modelf_pdb)
    f_modelf = calculate_modelmap(args.modelf_pdb,dim,args)
    pdb2mmcif(args.model1_pdb)
    f_model1 = calculate_modelmap(args.model1_pdb,dim,args)
    # create res_arr and bin_idx
    nbin,res_arr,bin_idx = get_resolution_array(uc,f_hf1)
    # fsc between model1 and halfmaps
    fsc_list = []
    var_list = []
    for imap in [f_hf1,f_hf2]:
        bin_fsc,_ = fsc.anytwomaps_fsc_covariance(imap,f_model1,bin_idx,nbin)
        fsc_list.append(bin_fsc)
    # fsc between fullmap and modelf
    bin_fsc,_ = fsc.anytwomaps_fsc_covariance(f_ful,f_modelf,bin_idx,nbin)
    fsc_list.append(bin_fsc)

    plot_nlines(res_arr,fsc_list,'allmap_fsc_modelvsmap.eps',["half1-model1","half2-model1","fullmap-modelf"])
    plot_nlines2(1/res_arr,fsc_list,'allmap_fsc_modelvsmap-v2.eps',["half1-model1","half2-model1","fullmap-modelf"])

if (__name__ == "__main__"):
    main()



    
    


