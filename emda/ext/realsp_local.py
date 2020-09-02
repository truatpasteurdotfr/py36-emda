"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# local correlation in real space

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import sys
from emda import core, ext
import argparse


def get_fullmapcor(hfmap_corr):
    import numpy.ma as ma

    hfmap_corr_ma = ma.masked_less_equal(hfmap_corr, 0.0)
    ccf_ma = 2 * hfmap_corr_ma / (1.0 + hfmap_corr_ma)
    ccf = ccf_ma.filled(0.0)
    return ccf


def cc_twosimilarmaps(ccmap12, ccmap1, ccmap2, uc, origin):
    import numpy.ma as ma

    ccmap1_ma = ma.masked_less_equal(ccmap1, 0.0)
    ccmap2_ma = ma.masked_less_equal(ccmap2, 0.0)
    ccmap12_ma = ma.masked_less_equal(ccmap12, 0.0)
    cc12_ma = ccmap12_ma * np.sqrt(ccmap1_ma) * np.sqrt(ccmap2_ma)
    cc12 = cc12_ma.filled(0.0)
    return cc12


def truemap_model_cc(mapmodelcc, fullmapcc):
    import numpy.ma as ma

    # To prevent large numbers in truemapmodelcc
    mapmodelcc_ma = ma.masked_less_equal(mapmodelcc, 0.3)
    fullmapcc_ma = ma.masked_less_equal(fullmapcc, 0.3)
    truemapmodelcc_ma = mapmodelcc_ma / np.sqrt(fullmapcc_ma)
    truemapmodelcc = core.iotools.mask_by_value_greater(
        truemapmodelcc_ma.filled(0.0), masking_value=1.0
    )
    return truemapmodelcc


def get_3d_realspcorrelation(half1, half2, kern):
    # Full map correlation using FFT convolve
    import scipy.signal

    loc3_A = scipy.signal.fftconvolve(half1, kern, "same")
    loc3_A2 = scipy.signal.fftconvolve(half1 * half1, kern, "same")
    loc3_B = scipy.signal.fftconvolve(half2, kern, "same")
    loc3_B2 = scipy.signal.fftconvolve(half2 * half2, kern, "same")
    loc3_AB = scipy.signal.fftconvolve(half1 * half2, kern, "same")
    cov3_AB = loc3_AB - loc3_A * loc3_B
    var3_A = loc3_A2 - loc3_A ** 2
    var3_B = loc3_B2 - loc3_B ** 2
    cov3_AB = np.where(var3_A * var3_B <= 0.0, 0.0, cov3_AB)
    var3_A = np.where(var3_A <= 0.0, 0.1, var3_A)
    var3_B = np.where(var3_B <= 0.0, 0.1, var3_B)
    halfmaps_cc = cov3_AB / np.sqrt(var3_A * var3_B)
    halfmaps_cc = np.where(cov3_AB <= 0.0, 0.0, halfmaps_cc)
    fullmap_cc = 2 * halfmaps_cc / (1.0 + halfmaps_cc)
    return halfmaps_cc, fullmap_cc


def get_3d_realspmapmodelcorrelation(map, model, kern_sphere):
    mapmodel_cc, _ = get_3d_realspcorrelation(map, model, kern_sphere)
    return mapmodel_cc


def calculate_modelmap(modelxyz, dim, resol, uc, bfac=0.0, lig=False, lgf=None):
    import emda.emda_methods as em

    modelmap = em.model2map(
        modelxyz=modelxyz,
        dim=dim,
        resol=resol,
        cell=uc,
        lig=lig,
        ligfile=lgf,
        bfac=bfac,
    )
    return modelmap


def rcc(
    half1_map,
    half2_map,
    kernel_size,
    norm=False,
    lig=False,
    model=None,
    model_resol=None,
    mask_map=None,
    lgf=None,
):
    hf1, hf2 = None, None
    bin_idx = None
    print(
        "Calculating 3D correlation between half maps and fullmap. \
            Please wait..."
    )
    uc, arr1, origin = core.iotools.read_map(half1_map)
    uc, arr2, origin = core.iotools.read_map(half2_map)
    # mask taking into account
    if mask_map is not None:
        _, cc_mask, _ = core.iotools.read_map(mask_map)
    if mask_map is None:
        # creating ccmask from half data
        obj_maskmap = ext.maskmap_class.MaskedMaps()
        obj_maskmap.generate_mask(arr1, arr2)
        cc_mask = obj_maskmap.mask
    #
    nx, ny, nz = arr1.shape
    # Creating soft-edged mask
    kern_sphere_soft = core.restools.create_soft_edged_kernel_pxl(kernel_size)
    # full map from half maps
    f_hf1 = np.fft.fftn(arr1)
    f_hf2 = np.fft.fftn(arr2)
    fullmap = np.real(np.fft.ifftn((f_hf1 + f_hf2) / 2.0))
    if norm:
        hf1, hf2 = np.fft.fftshift(f_hf1), np.fft.fftshift(f_hf2)
        print("Normalising maps...")
        nm = NormalizedMaps(hf1=hf1, hf2=hf2, cell=uc)
        nm.get_normdata()
        nbin, bin_idx = nm.nbin, nm.bin_idx
        halffsc, res_arr = nm.binfsc, nm.res_arr
        normfull = nm.normfull
        # Real space correlation maps
        print("Calculating 3D correlation using normalized maps...")
        halfmapscc, fullmapcc = get_3d_realspcorrelation(
            half1=nm.normmap1, half2=nm.normmap2, kern=kern_sphere_soft
        )
    if not norm:
        # Real space correlation maps
        print("Calculating 3D correlation...")
        halfmapscc, fullmapcc = get_3d_realspcorrelation(
            half1=arr1, half2=arr2, kern=kern_sphere_soft
        )
    print("Writing out correlation maps")
    core.iotools.write_mrc(
        mapdata=halfmapscc,
        filename="rcc_halfmap_smax" + str(kernel_size) + ".mrc",
        unit_cell=uc,
        map_origin=origin,
        label=True,
    )
    core.iotools.write_mrc(
        mapdata=fullmapcc * cc_mask,
        filename="rcc_fullmap_smax" + str(kernel_size) + ".mrc",
        unit_cell=uc,
        map_origin=origin,
        label=True,
    )
    # Map-model correlation
    if model is not None:
        print("\nMap-model correlation!\n")
        dim = [nx, ny, nz]
        if model.endswith((".pdb", ".ent", ".cif")):
            if model_resol is None:
                if norm:
                    # determine map resolution using hfmap FSC
                    dist = np.sqrt((halffsc - 0.143) ** 2)
                    map_resol = res_arr[np.argmin(dist)]
                    print("map will be calculated upto " + str(map_resol) + "A")
                if not norm:
                    hf1, hf2 = np.fft.fftshift(f_hf1), np.fft.fftshift(f_hf2)
                    nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, hf1)
                    halffsc, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(
                        hf1, hf2, bin_idx, nbin
                    )
                    dist = np.sqrt((halffsc - 0.143) ** 2)
                    map_resol = res_arr[np.argmin(dist)]
                    print("map will be calculated upto " + str(map_resol) + "A")
            else:
                map_resol = model_resol
                print("map will be calculated up to " + str(map_resol) + "A")
            print("Calculating map from model using REFMAC!")
            model_arr = calculate_modelmap(
                modelxyz=model, dim=dim, resol=map_resol, uc=uc, lig=lig, lgf=lgf
            )
        elif model.lower().endswith((".mrcs", ".mrc", ".map")):
            _, model_arr, _ = core.iotools.read_map(model)
        elif model.lower().endswith(".mtz"):
            model_arr = core.maptools.mtz2map(model, (nx, ny, nz))
        if norm:
            # normalisation
            normmodel = normalized(map=model_arr, bin_idx=bin_idx, nbin=nbin)
            print("Calculating model-map correlation...\n")
            mapmodelcc = get_3d_realspmapmodelcorrelation(
                map=normfull, model=normmodel, kern_sphere=kern_sphere_soft
            )
        if not norm:
            print("Calculating model-map correlation...\n")
            mapmodelcc = get_3d_realspmapmodelcorrelation(
                map=fullmap, model=model_arr, kern_sphere=kern_sphere_soft
            )
        core.iotools.write_mrc(
            mapdata=mapmodelcc * cc_mask,
            filename="rcc_mapmodel_smax" + str(kernel_size) + ".mrc",
            unit_cell=uc,
            map_origin=origin,
            label=True,
        )
        print("Calculating truemap-model correlation...")
        # truemap-model correlation
        truemapmodelcc = truemap_model_cc(mapmodelcc, fullmapcc)
        core.iotools.write_mrc(
            mapdata=truemapmodelcc * cc_mask,
            filename="rcc_truemapmodel_smax" + str(kernel_size) + ".mrc",
            unit_cell=uc,
            map_origin=origin,
            label=True,
        )
        print("Map-model correlation calculated!")


def scale_model2map(fullmap, model_arr, uc):
    from emda.ext.scale_maps import scale_twomaps_by_power, transfer_power

    f1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(fullmap)))
    f2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(model_arr)))
    nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, f1)
    scale_grid = transfer_power(
        bin_idx,
        res_arr,
        scale_twomaps_by_power(f1, f2, bin_idx=bin_idx, res_arr=res_arr),
    )
    scaled_model = np.real(np.fft.ifftn(np.fft.ifftshift(f2 * scale_grid)))
    return scaled_model


def mapmodel_rcc(
    fullmap, model, resol, kernel_size=9, lig=False, trim_px=1, mask_map=None, lgf=None
):

    print(
        "Calculating 3D correlation between map and model. \
            Please wait..."
    )
    uc, arr1, origin = core.iotools.read_map(fullmap)
    nx, ny, nz = arr1.shape
    if mask_map is not None:
        _, cc_mask, _ = core.iotools.read_map(mask_map)
    else:
        nbin = arr1.shape[0] // 2 - trim_px
        obj_maskmap = ext.maskmap_class.MaskedMaps()
        edge_mask = obj_maskmap.create_edgemask(nbin)
        cc_mask = np.zeros(shape=(nx, ny, nz), dtype="bool")
        cx, cy, cz = edge_mask.shape
        dx = (nx - cx) // 2
        dy = (ny - cy) // 2
        dz = (nz - cz) // 2
        print(dx, dy, dz)
        cc_mask[dx : dx + cx, dy : dy + cy, dz : dz + cz] = edge_mask

    assert arr1.shape == cc_mask.shape
    print("\nMap-model correlation!\n")
    dim = [nx, ny, nz]
    if model.endswith((".pdb", ".ent", ".cif")):
        print("map will be calculated up to " + str(resol) + "A")
        print("Calculating map from model using REFMAC!")
        model_arr = calculate_modelmap(
            modelxyz=model, dim=dim, resol=resol, uc=uc, lig=lig, lgf=lgf
        )
    elif model.lower().endswith((".mrcs", ".mrc", ".map")):
        _, model_arr, _ = core.iotools.read_map(model)
    elif model.lower().endswith(".mtz"):
        model_arr = core.maptools.mtz2map(model, (nx, ny, nz))
    core.iotools.write_mrc(
        mapdata=model_arr, filename="modelmap.mrc", unit_cell=uc, map_origin=origin
    )
    print("Calculating model-map correlation...\n")
    kern_sphere_soft = core.restools.create_soft_edged_kernel_pxl(kernel_size)
    mapmodelcc = get_3d_realspmapmodelcorrelation(arr1, model_arr, kern_sphere_soft)
    core.iotools.write_mrc(
        mapdata=mapmodelcc * cc_mask,
        filename="rcc_mapmodel.mrc",
        unit_cell=uc,
        map_origin=origin,
        label=True,
    )
    print("Calculating truemap-model correlation...")
    print("Map-model correlation calculated!")


def normalized(map, bin_idx=None, nbin=None, uc=None):
    # normalise in resol bins
    if bin_idx is None:
        if uc is not None:
            nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, map)
    hf1 = np.fft.fftshift(np.fft.fftn(map))
    _, _, _, totalvar, _, eo = core.fsc.halfmaps_fsc_variance(hf1, hf1, bin_idx, nbin)
    norm_map = np.real(np.fft.ifftn(np.fft.ifftshift(eo)))
    return norm_map


def hfdata_normalized(hf1, hf2, bin_idx=None, nbin=None, uc=None):
    # normalise in resol bins
    import fcodes_fast as fc

    if bin_idx is None:
        if uc is not None:
            nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, hf1)
    binfsc, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(hf1, hf2, bin_idx, nbin)
    _, _, _, _, _, e1 = core.fsc.halfmaps_fsc_variance(hf1, hf1, bin_idx, nbin)
    _, _, _, _, _, e2 = core.fsc.halfmaps_fsc_variance(hf2, hf2, bin_idx, nbin)
    nx, ny, nz = hf1.shape
    fsc3d = fc.read_into_grid(bin_idx, binfsc, nbin, nx, ny, nz)
    norm_map1 = np.real(np.fft.ifftn(np.fft.ifftshift(e1 * fsc3d)))
    norm_map2 = np.real(np.fft.ifftn(np.fft.ifftshift(e2 * fsc3d)))
    return norm_map1, norm_map2


class NormalizedMaps:
    def __init__(self, hf1=None, hf2=None, bin_idx=None, nbin=None, cell=None):
        self.hf1 = hf1
        self.hf2 = hf2
        self.e0 = None
        self.e1 = None
        self.e2 = None
        self.normmap1 = None
        self.normmap2 = None
        self.normfull = None
        self.bin_idx = bin_idx
        self.cell = cell
        self.nbin = nbin
        self.binfsc = None
        self.res_arr = None

    def get_parameters(self):
        if self.bin_idx is None:
            if self.cell is not None:
                (
                    self.nbin,
                    self.res_arr,
                    self.bin_idx,
                ) = core.restools.get_resolution_array(self.cell, self.hf1)

    def get_normdata(self):
        import fcodes_fast as fc

        if self.bin_idx is None:
            self.get_parameters()
        self.binfsc, _, _, _, _, self.e0 = core.fsc.halfmaps_fsc_variance(
            self.hf1, self.hf2, self.bin_idx, self.nbin
        )
        _, _, _, _, _, self.e1 = core.fsc.halfmaps_fsc_variance(
            self.hf1, self.hf1, self.bin_idx, self.nbin
        )
        _, _, _, _, _, self.e2 = core.fsc.halfmaps_fsc_variance(
            self.hf2, self.hf2, self.bin_idx, self.nbin
        )
        nx, ny, nz = self.hf1.shape
        fsc3d = fc.read_into_grid(self.bin_idx, self.binfsc, self.nbin, nx, ny, nz)
        self.normmap1 = np.real(np.fft.ifftn(np.fft.ifftshift(self.e1 * fsc3d)))
        self.normmap2 = np.real(np.fft.ifftn(np.fft.ifftshift(self.e2 * fsc3d)))
        self.normfull = np.real(np.fft.ifftn(np.fft.ifftshift(self.e0 * fsc3d)))
