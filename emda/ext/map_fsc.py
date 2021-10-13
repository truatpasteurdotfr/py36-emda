"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# This module calculates FSC between maps and model

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import emda.emda_methods as em
from emda import core, ext


def calc_fsc_mrc(hf1, hf2, bin_idx, nbin):
    bin_fsc, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(hf1, hf2, bin_idx, nbin)
    return bin_fsc


def calculate_modelmap(
    uc, model, dim, resol, lgf=None, maporigin=None
):
    #print('input params:', model, resol, dim, uc)
    """ modelmap = em.model2map(
        modelxyz=model,
        dim=dim,
        resol=resol,
        cell=uc,
        ligfile=lgf,
        maporigin=maporigin,
    ) """
    modelmap = em.model2map_gm(
        modelxyz=model,
        resol=resol,
        dim=dim,
        cell=uc,
        maporigin=maporigin,
    )
    #print('output shape:', modelmap.shape)
    f_model = np.fft.fftshift(np.fft.fftn(modelmap))
    return f_model, modelmap


def pass_mtz(mtzfile, dim):
    arr = core.maptools.mtz2map(mtzfile, dim)
    f_map = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr)))
    return f_map


def phase_randomized_fsc(arr1, arr2, mask, bin_idx, res_arr, fobj, resol_rand=None):
    from emda.ext.phase_randomize import get_randomized_sf

    fsc_list = []
    nbin = np.max(bin_idx) + 1
    fmap1 = np.fft.fftshift(np.fft.fftn(arr1 * mask))
    #fmap2 = np.fft.fftshift(np.fft.fftn(arr2 * mask))
    fmap2 = np.fft.fftshift(np.fft.fftn(arr2))
    bin_stats = core.fsc.anytwomaps_fsc_covariance(
        f1=fmap1, f2=fmap2, bin_idx=bin_idx, nbin=nbin
    )
    full_fsc_t, bin_count = bin_stats[0], bin_stats[2]
    fsc_list.append(full_fsc_t)
    if resol_rand is None:
        # resol_rand is taken as the point where half-fsc_masked falls below 0.8 - RELION
        idx = np.argmin((full_fsc_t - 0.8) ** 2)
        resol_rand = res_arr[idx]
    else:
        idx = np.argmin((res_arr - resol_rand) ** 2)
    # phase randomization
    print('phase randomize resolution: ', res_arr[idx])
    fobj.write('phase randomize resolution: %6.2f\n' % res_arr[idx])
    f1_randomized = get_randomized_sf(bin_idx, arr1, idx)
    f2_randomized = get_randomized_sf(bin_idx, arr2, idx)
    # get phase randomized maps
    randarr1 = np.real(np.fft.ifftn(np.fft.ifftshift(f1_randomized)))
    randarr2 = np.real(np.fft.ifftn(np.fft.ifftshift(f2_randomized)))
    full_fsc_n, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(
        np.fft.fftshift(np.fft.fftn(randarr1 * mask)),
        #np.fft.fftshift(np.fft.fftn(randarr2 * mask)),
        np.fft.fftshift(np.fft.fftn(randarr2)),
        bin_idx,
        nbin,
    )
    fsc_list.append(full_fsc_n)
    fsc_true = (full_fsc_t - full_fsc_n) / (1 - full_fsc_n)
    fsc_true[: idx + 2] = full_fsc_t[: idx + 2]
    fsc_list.append(fsc_true)
    core.plotter.plot_nlines(
        res_arr=res_arr,
        list_arr=fsc_list,
        mapname="fsc_phaserand.eps",
        curve_label=["total", "noise", "true"],
        fscline=0.5,
    )
    return fsc_list, bin_count


def map_model_fsc(
    half1_map,
    half2_map,
    modelf_pdb,
    bfac=0.0,
    lig=True,
    norm_mask=False,
    model1_pdb=None,
    mask_map=None,
    model_resol=None,
    lgf=None,
):
    import os

    fobj = open("mapmodel_validate.txt", "+w")
    fsc_list = []
    if model1_pdb is not None:
        model_list = [modelf_pdb, model1_pdb]
    else:
        model_list = [modelf_pdb]

    # if maps are in MRC format
    if half1_map.endswith((".mrc", ".map")):
        uc, arr1, origin = core.iotools.read_map(half1_map)
        uc, arr2, _ = core.iotools.read_map(half2_map)
    # mask taking into account
    if mask_map is not None:
        uc, msk, _ = core.iotools.read_map(mask_map)
    else:
        msk = 1
    f_hf1 = np.fft.fftshift(np.fft.fftn(arr1 * msk))
    f_hf2 = np.fft.fftshift(np.fft.fftn(arr2 * msk))
    f_ful = (f_hf1 + f_hf2) / 2.0

    nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, f_hf1)
    bin_fsc, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(f_hf1, f_hf2, bin_idx, nbin)
    fsc_list.append(bin_fsc)
    """ if model_resol is None:
        # determine map resolution using hfmap FSC
        dist = np.sqrt((bin_fsc - 0.143) ** 2)
        map_resol = res_arr[np.argmin(dist)]
    else:
        map_resol = model_resol """
    map_resol = res_arr[-2] # up to Nyquest
    print("Modelmap is calcuated upto Nyquist resolution.")
    # Calculate maps from models
    fmodel_list = []
    dim = f_hf1.shape
    for model in model_list:
        if model.endswith((".pdb", ".ent", ".cif")):
            f_model, modelmap = calculate_modelmap(
                uc=uc,
                model=model,
                dim=dim,
                resol=map_resol,
                lgf=lgf,
                maporigin=origin,
            )
        else:
            raise SystemExit("Accpetable model types: .pdb, .ent, .cif")
        fmodel_list.append(f_model)
        # output modelmap
        em.write_mrc(modelmap, 'last_modelmap.mrc', uc, origin)

    if len(fmodel_list) == 2:
        # FSC between halfmaps and model1
        for imap in [f_hf1, f_hf2]:
            print(imap.shape, fmodel_list[1].shape)
            bin_fsc = calc_fsc_mrc(imap, fmodel_list[1], bin_idx, nbin)
            fsc_list.append(bin_fsc)

    # FSC between fullmap and modelf
    bin_fsc = calc_fsc_mrc(f_ful, fmodel_list[0], bin_idx, nbin)
    fsc_list.append(bin_fsc)
    # write out data into txt file
    fobj.write("halfmap1 file: %s\n" % os.path.abspath(half1_map))
    fobj.write("halfmap2 file: %s\n" % os.path.abspath(half2_map))
    fobj.write("model refined against fullmap : %s\n" % os.path.abspath(modelf_pdb))
    fobj.write("model refined against half1 : %s\n" % os.path.abspath(model1_pdb))
    fobj.write("\n")
    fobj.write("***** Unmasked FSC *****\n")
    fobj.write("\n")
    fobj.write("bin # \n")
    fobj.write("resolution (Ang.) \n")
    fobj.write("FSC half1 vs half2 \n")
    fobj.write("FSC half1 vs model1 \n")
    fobj.write("FSC half2 vs model1 \n")
    fobj.write("FSC fullmap vs model \n")
    i = -1
    for i, resol in enumerate(res_arr):
        fobj.write("{:-3d} {:-6.2f} {:-14.4f} {:-14.4f} {:-14.4f} {:-14.4f}\n".format(
                i, resol, fsc_list[0][i], fsc_list[1][i], fsc_list[2][i], fsc_list[3][i]
            ))
    # output plots
    if len(fsc_list) == 4:
        core.plotter.plot_nlines(
            res_arr,
            fsc_list,
            "allmap_fsc_modelvsmap.eps",
            ["half1-half2", "half1-model1", "half2-model1", "fullmap-model"],
        )
        core.plotter.plot_nlines2(
            1 / res_arr,
            fsc_list,
            "allmap_fsc_modelvsmap-2.eps",
            ["half1-half2", "half1-model1", "half2-model1", "fullmap-model"],
        )
    elif len(fsc_list) == 2:
        core.plotter.plot_nlines(
            res_arr, fsc_list, "fsc_modelvsmap.eps", ["half1-half2", "fullmap-model"]
        )
        core.plotter.plot_nlines2(
            1 / res_arr, fsc_list, "fsc_modelvsmap-2.eps", ["half1-half2", "fullmap-model"]
        )
    return fsc_list


def map_model_fsc_phaserand(
    half1_map,
    half2_map,
    modelf_pdb,
    bfac=0.0,
    lig=True,
    norm_mask=False,
    model1_pdb=None,
    mask_map=None,
    model_resol=None,
    lgf=None,
    phaserand=True
):

    fobj = open("mapmodel_validate.txt", "+w")
    fsc_list = []
    if model1_pdb is not None:
        model_list = [modelf_pdb, model1_pdb]
    else:
        model_list = [modelf_pdb]

    # if maps are in MRC format
    if half1_map.endswith((".mrc", ".map")):
        uc, arr1, origin = core.iotools.read_map(half1_map)
        uc, arr2, _ = core.iotools.read_map(half2_map)
    # mask taking into account
    if mask_map is not None:
        uc, msk, _ = core.iotools.read_map(mask_map)
    else:
        # calculate mask from model
        mask = em.mask_from_atomic_model(mapname=half1_map, 
                                         modelname=model_list[0], atmrad=5)
    f_hf1 = np.fft.fftn(arr1)
    f_hf2 = np.fft.fftn(arr2)
    f_ful = (f_hf1 + f_hf2) / 2.0
    nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, f_hf1)

    # get mask corrected FSC
    from emda.ext.phase_randomize import phase_randomized_fsc
    fsc_all, _, resol_phs = phase_randomized_fsc(arr1=arr1, 
                                    arr2=arr2, 
                                    mask=msk, 
                                    bin_idx=bin_idx, 
                                    res_arr=res_arr, 
                                    fobj=fobj, 
                                    resol_rand=None)
    fsc_list.append(fsc_all[-1])
    # Calculate maps from models
    map_resol = res_arr[-2] # up to Nyquest
    dim = f_hf1.shape
    modelarr_list = []
    for model in model_list:
        if model.endswith((".pdb", ".ent", ".cif")):
            _, modelarr = calculate_modelmap(
                uc=uc,
                model=model,
                dim=dim,
                resol=map_resol,
                lgf=lgf,
                maporigin=origin,
            )
        else:
            raise SystemExit("Accpetable model types: .pdb, .ent, .cif")
        modelarr_list.append(modelarr)

    if len(modelarr_list) == 2:
        # FSC between halfmaps and model1
        for imap in [arr1, arr2]:
            fsc_all = phase_randomized_fsc(arr1=imap, 
                                            arr2=modelarr_list[-1], 
                                            mask=msk, 
                                            bin_idx=bin_idx, 
                                            res_arr=res_arr, 
                                            fobj=fobj, 
                                            resol_rand=resol_phs)[0]            
            fsc_list.append(fsc_all[-1])

    # FSC between fullmap and modelf
    fsc_all = phase_randomized_fsc(arr1=np.real(np.fft.ifftn(f_ful)), 
                                    arr2=modelarr_list[0], 
                                    mask=msk, 
                                    bin_idx=bin_idx, 
                                    res_arr=res_arr, 
                                    fobj=fobj, 
                                    resol_rand=resol_phs)[0] 
    fsc_list.append(fsc_all[-1])
    # output data into text file
    negative = False
    cref_arr = np.empty(np.size(res_arr, axis=0), dtype='float')
    cref_arr.fill(np.nan)
    fsc_full = 2 * fsc_list[0] / (1.0 + fsc_list[0])
    import math
    for i, _ in enumerate(fsc_list[0]):
        if fsc_list[0][i] > 0.: 
            if not(negative):
                cref_arr[i] = math.sqrt(fsc_full[i])
        else:
            negative = True
        print(res_arr[i], fsc_list[0][i], fsc_list[1][i], fsc_list[2][i], fsc_list[3][i], cref_arr[i])
    fsc_list.append(cref_arr)
    # output plots
    if len(fsc_list) == 5:
        core.plotter.plot_nlines(
            res_arr,
            fsc_list,
            "allmap_fsc_modelvsmap.eps",
            ["hf1-hf2", "half1-model1", "half2-model1", "fullmap-model", "sqrt(fsc_full)"],
        )
        core.plotter.plot_nlines2(
            1 / res_arr,
            fsc_list,
            "allmap_fsc_modelvsmap-2.eps",
            ["hf1-hf2", "half1-model1", "half2-model1", "fullmap-model", "sqrt(fsc_full)"],
        )
    elif len(fsc_list) == 2:
        core.plotter.plot_nlines(
            res_arr, fsc_list, "fsc_modelvsmap.eps", ["hf1-hf2", "fullmap-model"]
        )
        core.plotter.plot_nlines2(
            1 / res_arr, fsc_list, "fsc_modelvsmap-2.eps", ["hf1-hf2", "fullmap-model"]
        )
    return fsc_list


def fsc_mapmodel(
    map1,
    model,
    fobj,
    model_resol=5.0,
    bfac=0.0,
    phaserand=False,
    lig=True,
    mask_map=None,
    lgf=None,
):
    import os

    fobj.write("Map file: %s\n" % os.path.abspath(map1))
    fobj.write("Model file: %s\n" % os.path.abspath(model))
    fobj.write("\n")
    uc, arr1, orig = em.get_data(map1)
    if model.endswith((".pdb", ".ent", ".cif")):
        modelmap = em.model2map(
            modelxyz=model,
            dim=arr1.shape,
            resol=model_resol,
            cell=uc,
            ligfile=lgf,
            maporigin=orig,
        )
        em.write_mrc(modelmap, 'modelmap.mrc', uc, orig)
    elif model.endswith((".mrc", ".map")):
        _, modelmap, _ = em.get_data(model)
    fsc_list = []
    nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, arr1)
    print("Resolution grid Done.")
    print("Calculating unmasked FSC...")
    # unmask FSC
    f_map = np.fft.fftshift(np.fft.fftn(arr1))
    f_model = np.fft.fftshift(np.fft.fftn(modelmap))
    bin_stats = core.fsc.anytwomaps_fsc_covariance(
        f1=f_map, f2=f_model, bin_idx=bin_idx, nbin=nbin
    )
    print("FSC calculation Done.")
    bin_fsc, bin_count = bin_stats[0], bin_stats[2]
    fsc_list.append(bin_fsc)
    if mask_map is not None:
        print("mask file: %s is used.\n" % os.path.abspath(mask_map))
        fobj.write("mask file: %s\n" % os.path.abspath(mask_map))
        _, mask, _ = em.get_data(mask_map)
    else:
        print("No mask is given. EMDA calculated mask from atomic model is used.\n")
        fobj.write("No mask is given. EMDA calculated mask from atomic model is used.\n")
        #mask = em.mask_from_map(uc, modelmap, kern=9, itr=5)
        mask = em.mask_from_atomic_model(mapname=map1, modelname=model, atmrad=3)
    if phaserand:
        print("Phase randomization is carrying out...")
        fobj.write("Phase randomization is carrying out...\n")
        #nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, arr1)
        #print("Resolution grid Created.")
        fsclist, bin_count = phase_randomized_fsc(arr1, modelmap, mask, bin_idx, res_arr, fobj)
        bin_fsc = fsclist[-1]
        fsc_list.append(bin_fsc)
    else:
        f_map = np.fft.fftshift(np.fft.fftn(arr1 * mask))
        #f_model = np.fft.fftshift(np.fft.fftn(modelmap))
        #nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, arr1)
        #print("Resolution grid Done.")
        print("Calculating FSC...")
        bin_stats = core.fsc.anytwomaps_fsc_covariance(
            f1=f_map, f2=f_model, bin_idx=bin_idx, nbin=nbin
        )
        print("FSC calculation Done.")
        bin_fsc, bin_count = bin_stats[0], bin_stats[2]
        fsc_list.append(bin_fsc)
    # output data into a file
    fobj.write("\n")
    fobj.write("bin number \n")
    fobj.write("resolution (Ang.) \n")
    fobj.write("reci. resolution (1/Ang.) \n")
    fobj.write("unmask-FSC \n")
    fobj.write("mask-FSC \n")
    fobj.write("number of reflections \n")
    fobj.write("\n")
    print()
    print("Bin#    Resolution(A)  reci.Resol.(1/A)  unmask-FSC   mask-FSC     #reflx")
    i = -1
    for umfsci, mfsci, nfc in zip(fsc_list[0], fsc_list[1], bin_count):
        i += 1
        print(
            "{:-3d} {:-8.3f} {:-8.3f} {:-14.4f} {:-14.4f} {:-10d}".format(
                i, res_arr[i], 1.0/res_arr[i],umfsci, mfsci, nfc
            )
        )
        fobj.write(
            "{:-3d} {:-8.3f} {:-8.3f} {:-14.4f} {:-14.4f} {:-10d}\n".format(
                i, res_arr[i], 1.0/res_arr[i], umfsci, mfsci, nfc
            )
        )
    core.plotter.plot_nlines(
        res_arr=res_arr,
        list_arr=fsc_list,
        mapname="fsc_modelmap.eps",
        curve_label=["unmask-FSC", "mask-FSC"],
        fscline=0.5,
        plot_title="Map-model FSC"
    )
    print("FSC plotted into fsc_modelmap.eps")
    fobj.write("\n")
    fobj.write("FSC plotted into fsc_modelmap.eps")
    return res_arr, bin_fsc
