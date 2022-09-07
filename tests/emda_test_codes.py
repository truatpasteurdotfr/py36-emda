import emda.emda_methods as em
import numpy as np
from emda.core import restools, fsc, plotter

'''This file contains example codes for using emda API'''

''' 1. calculating map from an atomic model '''
imap = ""
imodel = ""
# read data from map
cell, maparr, origin = em.get_data(struct=imap)
# simulate density from model
modelarr = em.model2map_gm(
    modelxyz=imodel, 
    resol=4.5, 
    dim=maparr.shape, 
    cell=cell, 
    maporigin=origin)
# create resolution grid
nbin, res_arr, bin_idx = restools.get_resolution_array(uc=cell, hf1=maparr)
# compute FSC
bin_fsc = fsc.anytwomaps_fsc_covariance(
    f1 = np.fft.fftshift(np.fft.fftn(maparr)),
    f2 = np.fft.fftshift(np.fft.fftn(modelarr)),
    bin_idx=bin_idx, 
    nbin=nbin
    )[0]
# plot FSCs
plotter.plot_nlines(
    res_arr=res_arr,
    list_arr=[bin_fsc],
    mapname='fsc.eps',
    fscline=0.5,
    )


''' 2. calculating mask from an atomic model '''
imodel = ""
imap = ""
# read data from map
cell, maparr, origin = em.get_data(struct=imap)
# calculate atomic mask in emda
mask = em.mask_from_atomic_model(
    mapname=imap, 
    modelname=imodel, 
    atmrad=5
    )
# output mask as MRC file
em.write_mrc(
    mapdata=mask, 
    filename='atomic_mask.mrc', 
    unit_cell=cell, 
    map_origin=origin
    )
