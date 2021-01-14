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


def get_3d_realspmapmodelcorrelation(imap, model, kern_sphere):
    mapmodel_cc, _ = get_3d_realspcorrelation(imap, model, kern_sphere)
    return mapmodel_cc


def calculate_modelmap(modelxyz, dim, resol, uc, bfac=0.0, lig=True, lgf=None):
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
    lig=True,
    model=None,
    model_resol=None,
    mask_map=None,
    lgf=None,
):
    import emda.emda_methods as em

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
        print("Mask is not given. EMDA will generate cc_mask.mrc and be used.")
        print("Please take a look at this automatic mask. It may be suboptimal.")
        cc_mask = em.mask_from_halfmaps(uc, arr1, arr2, radius=7, thresh=0.65)
        em.write_mrc(cc_mask, "cc_mask.mrc", uc, origin)
    cc_mask = cc_mask * (cc_mask > 0.0)
    nx, ny, nz = arr1.shape
    # Creating soft-edged mask
    kern_sphere_soft = core.restools.create_soft_edged_kernel_pxl(kernel_size)
    # full map from half maps
    f_hf1, f_hf2 = np.fft.fftn(arr1), np.fft.fftn(arr2)
    f_fullmap = (f_hf1 + f_hf2) / 2.0
    fullmap = np.real(np.fft.ifftn(f_fullmap))
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
            half1=nm.normmap1 * cc_mask,
            half2=nm.normmap2 * cc_mask,
            kern=kern_sphere_soft,
        )
    if not norm:
        # Real space correlation maps
        print("Calculating 3D correlation...")
        halfmapscc, fullmapcc = get_3d_realspcorrelation(
            half1=arr1 * cc_mask, half2=arr2 * cc_mask, kern=kern_sphere_soft
        )
    halfmapscc = halfmapscc * (halfmapscc > 0.0)
    fullmapcc = fullmapcc * (fullmapcc > 0.0)
    print("Writing out correlation maps")
    core.iotools.write_mrc(
        mapdata=halfmapscc * cc_mask,
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
        """ if mask_map is None:
            # generate mask from the model for map-model correlation
            modelmask = em.mask_from_map(uc, model_arr, kern=9, itr=5)
            cc_mask = modelmask """
        em.write_mrc(model_arr, "modelmap.mrc", uc, origin)
        if norm:
            # normalisation
            normmodel = normalized_modelmap(
                modelmap=model_arr, f_fullmap=f_fullmap, bin_idx=bin_idx, nbin=nbin
            )
            em.write_mrc(normmodel, "normmodel.mrc", uc, origin)
            print("Calculating model-map correlation...\n")
            mapmodelcc = get_3d_realspmapmodelcorrelation(
                imap=normfull * cc_mask,
                model=normmodel * cc_mask,
                kern_sphere=kern_sphere_soft,
            )
        if not norm:
            print("Calculating model-map correlation...\n")
            mapmodelcc = get_3d_realspmapmodelcorrelation(
                imap=fullmap * cc_mask, model=model_arr, kern_sphere=kern_sphere_soft
            )
        mapmodelcc = mapmodelcc * (mapmodelcc > 0.0)
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
        truemapmodelcc = truemapmodelcc * (truemapmodelcc > 0.0)
        core.iotools.write_mrc(
            mapdata=truemapmodelcc * cc_mask,
            filename="rcc_truemapmodel_smax" + str(kernel_size) + ".mrc",
            unit_cell=uc,
            map_origin=origin,
            label=True,
        )
        print("Map-model correlation calculated! Maps were writted!")

def make_c_B_table(s_max, dat_out=None):
    import scipy.special
    t = lambda B: np.sqrt(np.abs(B)/2)*s_max
    w = lambda z: np.exp(-z**2)*(1-1j*scipy.special.erfi(-z))
    fp = lambda x: 3/2*(np.sqrt(np.pi)/2*scipy.special.erf(t(x))-t(x)*np.exp(-t(x)**2))/t(x)**3
    fn = lambda x: 3/4*np.exp(t(x)**2)*(2*t(x)-np.sqrt(np.pi)*np.imag(w(t(x))))/t(x)**3

    Bn = np.arange(-300,0)
    cn = fn(Bn)
    Bp = np.arange(1, 600)
    cp = fp(Bp)

    Bs = np.concatenate((Bn, [0.], Bp))
    cs = np.concatenate((cn, [1.], cp))

    order = np.argsort(cs)
    Bs, cs = Bs[order], cs[order]

    if dat_out:
      with open(dat_out, "w") as ofs:
        ofs.write("# s_max= {}\n".format(s_max))
        ofs.write("c B\n")
        for i in range(len(Bs)):
          ofs.write("{:.5e} {:.1f}\n".format(cs[i], Bs[i]))

    return Bs, cs

def bfromcc(
    half1_map,
    half2_map,
    kernel_size,
    resol,
    mask_map=None,
):
    import emda.emda_methods as em

    hf1, hf2 = None, None
    bin_idx = None
    print(
        "Calculating 3D correlation between half maps. \
            Please wait..."
    )
    uc, arr1, origin = core.iotools.read_map(half1_map)
    uc, arr2, origin = core.iotools.read_map(half2_map)

    # mask taking into account
    if mask_map is not None:
        _, cc_mask, _ = core.iotools.read_map(mask_map)
    else:
        # creating ccmask from half data
        print("Mask is not given. EMDA will generate cc_mask.mrc and be used.")
        print("Please take a look at this automatic mask. It may be suboptimal.")
        cc_mask = em.mask_from_halfmaps(uc, arr1, arr2, radius=7, thresh=0.65)
        em.write_mrc(cc_mask, "cc_mask.mrc", uc, origin)

    cc_mask = cc_mask * (cc_mask > 0.0)
    # Creating soft-edged mask
    kern_sphere_soft = core.restools.create_soft_edged_kernel_pxl(kernel_size)

    # Lowpass
    f_hf1, arr1 = em.lowpass_map(uc=uc, arr1=arr1, resol=resol)
    f_hf2, arr2 = em.lowpass_map(uc=uc, arr1=arr2, resol=resol)

    # Real space correlation maps
    print("Calculating 3D correlation...")
    halfmapscc, fullmapcc = get_3d_realspcorrelation(
        half1=arr1 * cc_mask, half2=arr2 * cc_mask, kern=kern_sphere_soft
    )

    overall_cc = np.corrcoef((arr1*cc_mask).flatten(), (arr2*cc_mask).flatten())[1,0]
    print("Overall CC=", overall_cc)

    c = (1.-overall_cc)/overall_cc * halfmapscc/(1.-halfmapscc)
    print("c range: {:.4e} to {:.4e}".format(c.min(), c.max()))
    core.iotools.write_mrc(c,'c_for_localB'+str(kernel_size)+'.mrc',uc,origin)

    table_B, table_c = make_c_B_table(1./resol, dat_out="c_B.dat")
    print("c to B table made for c= {:.4e} to {:.4e} and B= {:.1f} to {:.1f}".format(min(table_c), max(table_c), min(table_B), max(table_B)))
    B = np.interp(c, table_c, table_B)
    print("Writing out local B map")
    core.iotools.write_mrc(mapdata=B,filename='localB'+str(kernel_size)+'.mrc',unit_cell=uc,map_origin=origin)

    print("Writing out correlation maps")
    core.iotools.write_mrc(
        mapdata=halfmapscc * cc_mask,
        filename="rcc_halfmap_smax" + str(kernel_size) + ".mrc",
        unit_cell=uc,
        map_origin=origin,
        label=True,
    )


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
    fullmap,
    model,
    resol,
    kernel_size=9,
    lig=True,
    norm=False,
    nomask=False,
    #trim_px=1,
    mask_map=None,
    lgf=None,
):
    import emda.emda_methods as em
    import fcodes_fast as fc

    print(
        "Calculating 3D correlation between map and model. \
            Please wait..."
    )
    uc, arr1, origin = core.iotools.read_map(fullmap)
    nx, ny, nz = arr1.shape
    mask = 1
    generate_mask = not(nomask)
    if mask_map is not None:
        _, mask, _ = core.iotools.read_map(mask_map)
        generate_mask=False
    """ else:
        nbin = arr1.shape[0] // 2 - trim_px
        obj_maskmap = ext.maskmap_class.MaskedMaps()
        edge_mask = obj_maskmap.create_edgemask(nbin)
        cc_mask = np.zeros(shape=(nx, ny, nz), dtype="bool")
        cx, cy, cz = edge_mask.shape
        dx = (nx - cx) // 2
        dy = (ny - cy) // 2
        dz = (nz - cz) // 2
        print(dx, dy, dz)
        cc_mask[dx : dx + cx, dy : dy + cy, dz : dz + cz] = edge_mask """
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
    if generate_mask:
        # generate mask from the model
        mask = em.mask_from_map(uc, model_arr, kern=9, itr=5)
    if np.isscalar(mask):
        print("Correlation is calculated without a mask")
    else:
        print("Correlation is calculated using a mask")
    print("Calculating model-map correlation...\n")
    kern_sphere_soft = core.restools.create_soft_edged_kernel_pxl(kernel_size)
    if norm:
        # using normalised maps
        f_mdl = np.fft.fftshift(np.fft.fftn(model_arr))
        f_map = np.fft.fftshift(np.fft.fftn(arr1))
        nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, arr1)
        eo, _, _, _, binfsc, _ = fc.get_normalized_sf(
            f_mdl, f_map, bin_idx, nbin, 0, nx, ny, nz
        )
        e_mdl = eo[:, :, :, 0]
        e_map = eo[:, :, :, 1]
        fsc_frid = fc.read_into_grid(bin_idx, binfsc, nbin, nx, ny, nz)
        norm_mdl = np.real(np.fft.ifftn(np.fft.ifftshift(e_mdl * fsc_frid)))
        norm_map = np.real(np.fft.ifftn(np.fft.ifftshift(e_map * fsc_frid)))
        mapmodelcc = get_3d_realspmapmodelcorrelation(
            norm_map * mask, norm_mdl, kern_sphere_soft
        )
    if not norm:
        mapmodelcc = get_3d_realspmapmodelcorrelation(
            arr1 * mask, model_arr, kern_sphere_soft
        )
    core.iotools.write_mrc(
        mapdata=mapmodelcc * mask,
        filename="rcc_mapmodel.mrc",
        unit_cell=uc,
        map_origin=origin,
        label=True,
    )
    print("Map-model correlation calculated! Maps were written!")


def normalized(map, bin_idx=None, nbin=None, uc=None):
    import fcodes_fast as fc

    # normalise in resol bins
    if bin_idx is None:
        if uc is not None:
            nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, map)
    hf1 = np.fft.fftshift(np.fft.fftn(map))
    binfsc, _, _, totalvar, _, eo = core.fsc.halfmaps_fsc_variance(
        hf1, hf1, bin_idx, nbin
    )
    binfsc = binfsc * (totalvar > 1e-5)
    nx, ny, nz = eo.shape
    fsc_frid = fc.read_into_grid(bin_idx, binfsc, nbin, nx, ny, nz)
    norm_map = np.real(np.fft.ifftn(np.fft.ifftshift(eo * fsc_frid)))
    return norm_map


def normalized_modelmap(modelmap, f_fullmap, bin_idx=None, nbin=None, uc=None):
    import fcodes_fast as fc

    # normalise in resol bins
    if bin_idx is None:
        if uc is not None:
            nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, modelmap)
    f1 = np.fft.fftshift(np.fft.fftn(modelmap))
    f_fullmap = np.fft.fftshift(f_fullmap)
    """ _, _, _, _, _, eo = core.fsc.halfmaps_fsc_variance(f1, f1, bin_idx, nbin)
    binfsc, _, _, _, _, _ = core.fsc.halfmaps_fsc_variance(f1, f_fullmap, bin_idx, nbin) """
    nx, ny, nz = f1.shape
    eo, _, _, _, binfsc, _ = fc.get_normalized_sf(
        f1, f_fullmap, bin_idx, nbin, 0, nx, ny, nz
    )
    e1 = eo[:, :, :, 0]
    fsc_frid = fc.read_into_grid(bin_idx, binfsc, nbin, nx, ny, nz)
    norm_map = np.real(np.fft.ifftn(np.fft.ifftshift(e1 * fsc_frid)))
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
        """ self.binfsc, _, _, _, _, self.e0 = core.fsc.halfmaps_fsc_variance(
            self.hf1, self.hf2, self.bin_idx, self.nbin
        )
        _, _, _, _, _, self.e1 = core.fsc.halfmaps_fsc_variance(
            self.hf1, self.hf1, self.bin_idx, self.nbin
        )
        _, _, _, _, _, self.e2 = core.fsc.halfmaps_fsc_variance(
            self.hf2, self.hf2, self.bin_idx, self.nbin
        ) """
        nx, ny, nz = self.hf1.shape
        eo, _, _, _, self.binfsc, _ = fc.get_normalized_sf(
            self.hf1, self.hf2, self.bin_idx, self.nbin, 0, nx, ny, nz
        )
        self.e1 = eo[:, :, :, 0]
        self.e2 = eo[:, :, :, 1]
        self.e0 = eo[:, :, :, 2]
        fsc_ful = 2 * self.binfsc / (1 + self.binfsc)
        fsc_frid = fc.read_into_grid(self.bin_idx, fsc_ful, self.nbin, nx, ny, nz)
        fsc_grid_filtered = np.where(fsc_frid < 0.0, 0.0, fsc_frid)
        fsc_grid_str = np.sqrt(fsc_grid_filtered)
        self.normmap1 = np.real(np.fft.ifftn(np.fft.ifftshift(self.e1 * fsc_grid_str)))
        self.normmap2 = np.real(np.fft.ifftn(np.fft.ifftshift(self.e2 * fsc_grid_str)))
        self.normfull = np.real(np.fft.ifftn(np.fft.ifftshift(self.e0 * fsc_grid_str)))
