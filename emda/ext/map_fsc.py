"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# This module calculates FSC between maps and model

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from emda import core, ext


def calc_fsc_mrc(hf1, hf2, bin_idx, nbin):
    bin_fsc, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(hf1, hf2, bin_idx, nbin)
    return bin_fsc


def calculate_modelmap(uc, model, dim, resol, bfac=0.0, lig=False, lgf=None):
    import emda.emda_methods as em

    modelmap = em.model2map(
        modelxyz=model, dim=dim, resol=resol, cell=uc, lig=lig, ligfile=lgf, bfac=bfac
    )
    f_model = np.fft.fftshift(np.fft.fftn(modelmap))
    return f_model


def pass_mtz(mtzfile, dim):
    arr = core.maptools.mtz2map(mtzfile, dim)
    f_map = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr)))
    return f_map


def map_model_fsc(
    half1_map,
    half2_map,
    modelf_pdb,
    bfac=0.0,
    lig=False,
    norm_mask=False,
    model1_pdb=None,
    mask_map=None,
    model_resol=None,
    lgf=None
):
    fsc_list = []
    if model1_pdb is not None:
        model_list = [modelf_pdb, model1_pdb]
    else:
        model_list = [model1_pdb]

    # if maps are in MRC format
    if half1_map.endswith((".mrc", ".mrcs", ".map")):
        uc, arr1, _ = core.iotools.read_map(half1_map)
        uc, arr2, _ = core.iotools.read_map(half2_map)
    # mask taking into account
    if mask_map is not None:
        uc, msk, _ = core.iotools.read_map(mask_map)
    if mask_map is None:
        if norm_mask:
            nm = ext.realsp_local.NormalizedMaps(
                hf1=np.fft.fftshift(np.fft.fftn(arr1)),
                hf2=np.fft.fftshift(np.fft.fftn(arr2)),
                cell=uc,
            )
            nm.get_normdata()
            obj_maskmap = ext.maskmap_class.MaskedMaps()
            obj_maskmap.generate_mask(nm.normmap1, nm.normmap2)
            msk = obj_maskmap.mask
        else:
            # creating ccmask from half data
            obj_maskmap = ext.maskmap_class.MaskedMaps()
            obj_maskmap.generate_mask(arr1, arr2)
            msk = obj_maskmap.mask
        core.iotools.write_mrc(msk, "ccmask.mrc", uc)
    f_hf1 = np.fft.fftshift(np.fft.fftn(arr1 * msk))
    f_hf2 = np.fft.fftshift(np.fft.fftn(arr2 * msk))
    f_ful = (f_hf1 + f_hf2) / 2.0

    # if maps are in MTZ format
    # if half1_map.endswith((".mtz")):
    #    if map_size is None:
    #        print("Need map dimensions.")
    #        exit()
    #    dim = map_size
    #    if len(dim) < 3:
    #        print("Need three values space delimited")
    #        exit()
    #    if len(dim) > 3:
    #        dim = dim[:3]
    #    f_hf1 = pass_mtz(half1_map, dim)
    #    f_hf2 = pass_mtz(half2_map, dim)

    nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, f_hf1)
    bin_fsc, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(f_hf1, f_hf2, bin_idx, nbin)
    fsc_list.append(bin_fsc)
    # fullmap_fsc = 2.0*bin_fsc/(bin_fsc + 1.0)
    # fsc_list.append(fullmap_fsc)
    if model_resol is None:
        # determine map resolution using hfmap FSC
        dist = np.sqrt((bin_fsc - 0.143) ** 2)
        map_resol = res_arr[np.argmin(dist)]
    else:
        map_resol = model_resol

    # Calculate maps from models
    fmodel_list = []
    dim = f_hf1.shape
    for model in model_list:
        if model.endswith((".pdb", ".ent", ".cif")):
            f_model = calculate_modelmap(
                uc=uc,
                model=model,
                dim=dim,
                resol=map_resol,
                bfac=bfac,
                lig=lig,
                lgf=lgf
            )
        else:
            raise SystemExit('Accpetable model types: .pdb, .ent, .cif')
        fmodel_list.append(f_model)

    if len(fmodel_list) == 2:
        # FSC between halfmaps and model1
        for imap in [f_hf1, f_hf2]:
            bin_fsc = calc_fsc_mrc(imap, fmodel_list[1], bin_idx, nbin)
            fsc_list.append(bin_fsc)

    # FSC between fullmap and modelf
    bin_fsc = calc_fsc_mrc(f_ful, fmodel_list[0], bin_idx, nbin)
    fsc_list.append(bin_fsc)
    # output plots
    if len(fsc_list) == 4:
        core.plotter.plot_nlines(
            res_arr,
            fsc_list,
            "allmap_fsc_modelvsmap.eps",
            ["hf1-hf2", "half1-model1", "half2-model1", "fullmap-model"],
        )
        core.plotter.plot_nlines2(
            1 / res_arr,
            fsc_list,
            "allmap_fsc_modelvsmap-2.eps",
            ["hf1-hf2", "half1-model1", "half2-model1", "fullmap-model"],
        )
    elif len(fsc_list) == 2:
        core.plotter.plot_nlines(
            res_arr, fsc_list, "fsc_modelvsmap.eps", ["hf1-hf2", "fullmap-model"]
        )
        core.plotter.plot_nlines2(
            1 / res_arr, fsc_list, "fsc_modelvsmap-2.eps", ["hf1-hf2", "fullmap-model"]
        )
    return fsc_list