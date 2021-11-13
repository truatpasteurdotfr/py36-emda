# Difference map

import numpy as np
import emda.emda_methods as em

# read maps
imap1 = "./gro-428-433_cutOut.mrc"
imap2 = "./sample2fit-density.mrc"
uc1, arr1, orig = em.get_data(imap1)
uc2, arr2, _ = em.get_data(imap2)
# resample to 100, 100, 100
current_pix = target_pix = [uc1[i]/shape for i, shape in enumerate(arr1.shape)]
target_dim = [100, 100, 100]
arr1_new = em.resample_data(curnt_pix=current_pix,
                            targt_pix=target_pix, 
                            arr=arr1, 
                            targt_dim=target_dim)
target_uc = np.array(target_pix, 'float') * np.asarray(target_dim, dtype="int")
em.write_mrc(arr1_new, '1.mrc', target_uc, orig)

arr2_new = em.resample_data(curnt_pix=[uc2[i]/shape for i, shape in enumerate(arr2.shape)],
                            targt_pix=target_pix, 
                            arr=arr2, 
                            targt_dim=target_dim)
em.write_mrc(arr2_new, '2.mrc', target_uc, orig)

# overlay maps
from emda.ext.overlay import EmmapOverlay, run_fit
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from emda.core import quaternions
emmap1 = EmmapOverlay(map_list=[])
emmap1.mask_list = None
emmap1.modelres = None
emmap1.map_unit_cell = np.array([target_uc[0], target_uc[1], target_uc[2], 90., 90., 90.], 'float')
emmap1.map_origin = orig
emmap1.map_dim = target_dim
emmap1.pixsize = target_pix
emmap1.arr_lst = [arr1_new, arr2_new]
emmap1.msk_lst = []
emmap1.carr_lst = []
emmap1.ceo_lst = None
emmap1.cfo_lst = None
emmap1.cbin_idx = None
emmap1.cdim = None
emmap1.cbin = None
emmap1.com = False
emmap1.com1 = None
emmap1.comlist = []
emmap1.box_centr = None
emmap1.fhf_lst = [fftshift(fftn(fftshift(arr1_new))), fftshift(fftn(fftshift(arr2_new)))]
emmap1.nbin = None
emmap1.res_arr = None
emmap1.bin_idx = None
emmap1.fo_lst = None
emmap1.eo_lst = None
emmap1.calc_fsc_from_maps()

rotmat_initial = np.identity(3, 'float') # Initial rotation matrix
translation_initial = np.array([0., 0., 0.], 'float') # Intial translation in Angstroms
rotmat_list = []
trans_list = []
for ifit in range(1, len(emmap1.eo_lst)):
    t, q_final = run_fit(
        emmap1=emmap1,
        rotmat=rotmat_initial,
        t=[itm / emmap1.pixsize[i] for i, itm in enumerate(translation_initial)],
        ncycles=10,
        ifit=ifit,
        fitres=None,
    )
    rotmat = quaternions.get_RM(q_final)
    rotmat_list.append(rotmat)
    trans_list.append(t)
print("Overlay finished")

# calculate difference maps
from emda.ext.diffmap_with_fit import apply_transformation_on_f, calculate_diffmap, MapOut, mapoutput

diffmapres = 3.0 # Resolution for difference map in Ansgtroms
results = MapOut()
flist = apply_transformation_on_f(emmap1, rotmat_list, trans_list)
diffmap = calculate_diffmap(emmap1=emmap1, f_list=flist, resol=diffmapres)
# output results
list_maps = []
list_masks = emmap1.msk_lst
uc = emmap1.map_unit_cell
origin = emmap1.map_origin
for i in range(diffmap.shape[3]):
    imap = np.real(ifftshift(ifftn(ifftshift(diffmap[:, :, :, i]))))
    list_maps.append(imap)
mapoutput(list_maps=list_maps, uc=uc, origin=origin, masklist=list_masks)