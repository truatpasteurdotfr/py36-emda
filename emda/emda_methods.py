"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# Caller script
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import fcodes_fast
from emda.config import debug_mode
from emda.core import iotools, maptools, restools, plotter, fsc, quaternions


def read_map(mapname):
    """Reads CCP4 type map (.map) or MRC type map.

    Arguments:
        Inputs:
            mapname: string
                CCP4/MRC map file name

        Outputs:
            uc: float, 1D array
                Unit cell
            arr: float, 3D array
                Map values as Numpy array
            origin: list
                Map origin list
    """
    uc, arr, origin = iotools.read_map(mapname)
    return uc, arr, origin


def read_mtz(mtzfile):
    """Reads mtzfile and returns unit_cell and data in Pandas DataFrame.

    Arguments:
        Inputs:
            mtzfile: string
                MTZ file name

        Outputs:
            uc: float, 1D array
                Unit cell
            df: Pandas data frame
                Map values in Pandas Dataframe
    """
    uc, df = iotools.read_mtz(mtzfile)
    return uc, df


def write_mrc(mapdata, filename, unit_cell, map_origin=None):
    """Writes 3D Numpy array into MRC file.

    Arguments:
        Inputs:
            mapdata: float, 3D array
                Map values to write
            filename: string
                Output file name
            unit_cell: float, 1D array
                Unit cell params
            map_origin: list, optional
                map origin. Default is [0.0, 0.0, 0.0]

        Outputs:
            Output MRC file
    """
    iotools.write_mrc(
        mapdata=mapdata, filename=filename, unit_cell=unit_cell, map_origin=map_origin
    )


def write_mtz(uc, arr, outfile="map2mtz.mtz"):
    """Writes 3D Numpy array into MTZ file.

    Arguments:
        Inputs:
            uc: float, 1D array
                Unit cell params.
            arr: complex, 3D array
                Map values to write.

        Outputs:
            outfile: string
            Output file name. Default is map2mtz.mtz.
    """
    iotools.write_3d2mtz(unit_cell=uc, mapdata=arr, outfile=outfile)


def resample_data(curnt_pix, targt_pix, targt_dim, arr):
    """Resamples a 3D array.

    Arguments:
        Inputs:
            curnt_pix: float
                Current pixel size.
            targt_pix: float
                Target pixel size.
            targt_dim: list
                List of three integer values.
            arr: float
                3D array of map values.

        Outputs:
            new_arr: float, 3D array
                Resampled 3D array.
    """
    new_arr = iotools.resample2staticmap(
        curnt_pix=curnt_pix, targt_pix=targt_pix, targt_dim=targt_dim, arr=arr
    )
    return new_arr


def estimate_map_resol(hfmap1name, hfmap2name):
    """Estimates map resolution.

    Arguments:
        Inputs:
            hfmap1name: string
                Halfmap 1 name.
            hfmap2name: string
                Halfmap 2 name.

        Outputs:
            map_resol: float
                Map resolution determined by the halfmap FSC.
    """
    map_resol = maptools.estimate_map_resol(hfmap1=hfmap1name, hfmap2=hfmap2name)
    return map_resol


def get_map_power(mapname):
    """Calculates the map power spectrum.

    Arguments:
        Inputs:
            mapname: string
                Map name.

        Outputs:
            res_arr: float
                Resolution array.
            power_spectrum: float
                Map power spectrum.
    """
    res_arr, power_spectrum = maptools.get_map_power(mapname)
    return res_arr, power_spectrum


def get_biso_from_model(mmcif_file):
    """Calculates the isotropic B-value of the model.

    Arguments:
        Inputs:
            mmcif_file: string
                mmCIF file name.

        Outputs:
            biso: float
                Model B-iso value.
    """
    biso = maptools.get_biso_from_model(mmcif_file)
    return biso


def get_biso_from_map(halfmap1, halfmap2):
    """Calculates the isotropic B-value of the map.

    Arguments:
        Inputs:
            halfmap1: string
                Halfmap 1 file name.
            halfmap2: string
                Halfmap 2 file name.

        Outputs:
            biso: float
                Map B-iso value.
    """
    biso = maptools.get_biso_from_map(halfmap1=halfmap1, halfmap2=halfmap2)
    return biso


def apply_bfactor_to_map(mapname, bf_arr, mapout):
    """Applies an array of B-factors on the map.

    Arguments:
        Inputs:
            mapname: string
                Map file name.
            bf_arr: float, 1D array
                An array/list of B-factors.
            mapout: bool
                If True, map for each B-factor will be output.

        Outputs:
            all_mapout: complex, ndarray
                4D array containing Fourier coefficients of all maps.
                e.g. all_mapout[:,:,:,i], where i represents map number
                corresponding to the B-factor in bf_arr.
    """
    all_mapout = maptools.apply_bfactor_to_map(
        mapname=mapname, bf_arr=bf_arr, mapout=mapout
    )
    return all_mapout


def map2mtz(mapname, mtzname="map2mtz.mtz"):
    """Converts a map into MTZ format.

    Arguments:
        Inputs:
            mapname: string
                Map file name.
            mtzname: string
                Output MTZ file name. Default is map2mtz.mtz

        Outputs:
            Outputs MTZ file.
    """
    maptools.map2mtz(mapname=mapname, mtzname=mtzname)


def map2mtzfull(uc, arr1, arr2, mtzname="halfnfull.mtz"):
    """Writes several 3D Numpy arrays into an MTZ file.

    This function accepts densities of two half maps as 3D numpy arrays
    and outputs an MTZ file containing amplitudes of half1, half2 and
    full map. The outfile data labels are H, K, L, Fout1, Fout2, Foutf, Poutf.
    The last four labels correspond to amplitude of halfmap1, amplitudes of
    halfmap2, amplitudes of fullmap and the phase values of fullmap, respectively.

    Arguments:
        Inputs:
            uc: float, 1D array
                Unit cell params.
            arr1: float, 3D array
                Half1 map values.
            arr2: float, 3D array
                Half2 map values.
            mtzname: string, optional
                Output MTZ file name. Default is halfnfull.mtz

        Outputs:
            Outputs an MTZ file containing amplitudes of half maps and
            full map.
    """
    hf1 = np.fft.fftshift(np.fft.fftn(arr1))
    hf2 = np.fft.fftshift(np.fft.fftn(arr2))
    iotools.write_3d2mtz_full(unit_cell=uc, hf1data=hf1, hf2data=hf2, outfile=mtzname)


def mtz2map(mtzname, map_size):
    """Converts an MTZ file into MRC format.

    This function converts data in an MTZ file into a 3D Numpy array.
    It combines amplitudes and phases with "Fout0" and
    "Pout0" labels to form Fourier coefficients. If the MTZ contains
    several aplitude columns, only the one corresponding to "Fout0"
    will be used.

    Arguments:
        Inputs:
            mtzname: string
                MTZ file name.
            map_size: list
                Shape of output 3D Numpy array as a list of three integers.

        Outputs:
            outarr: float
            3D Numpy array of map values.
    """
    data2write = maptools.mtz2map(mtzname=mtzname, map_size=map_size)
    return data2write


def lowpass_map(uc, arr1, resol, filter="ideal", order=4):
    """Lowpass filters a map to a specified resolution.

    This function applies a lowpass filter on a map to a specified resolution.
    This operations is carried out in the Fourier space. Note that lowpass map
    will have the same shape as input data.

    Arguments:
        Inputs:
            uc: float, 1D array
                Unit cell params.
            arr1: float, 3D array
                3D Numpy array containing map values.
            resol: float
                Resolution cutoff for lowpass filtering in Angstrom units.
            filter: string, optional
                Fiter type to use in truncating Fourier coefficients.
                Currently, only 'ideal' or 'butterworth' filters can be employed.
                Default type is ideal.
            order: integer, optional
                Order of the Butterworth filter. Default is 4.

        Outputs:
            fmap1: complex, 3D array
                Lowpass filtered Fourier coefficeints.
            map1: float, 3D array
                Lowpass filtered map in image/real space
    """
    import emda.ext.lowpass_map as lp

    if filter == "ideal":
        fmap1, map1 = lp.lowpass_map(uc, arr1, resol)
    if filter == "butterworth":
        fmap1, map1 = lp.butterworth(uc, arr1, resol, order)
    return fmap1, map1


def half2full(half1name, half2name, outfile="fullmap.mrc"):
    """Combines half maps to generate full map.

    Arguments:
        Inputs:
            half1name: string
                Name of half map 1.
            half2name: string
                name of half map 2.
            outfile: string, optional
                Name of the output file. Default is fullmap.mrc

        Outputs:
            fullmap: float, 3D array
                3D Numpy array of floats.
    """
    import emda.ext.half2full as h2f

    uc, arr1, origin = read_map(half1name)
    uc, arr2, origin = read_map(half2name)
    fullmap = h2f.half2full(arr1, arr2)
    write_mrc(fullmap, outfile, uc, origin)
    return fullmap


def map_transform(mapname, tra, rot, axr, outname="transformed.mrc"):
    """Imposes a transformation on a map.

    Imposes a transformation (i.e. translation and rotation) on a map
    and returns the transformed map.

    Arguments:
        Inputs:
            mapname: string
                Name of the input map.
            tra: list of three floats values
                Translation vector as a list in Angstrom units.
            rot: float
                Rotation to apply in degrees.
            axr: list of three integers
                Axis to rotation. e.g [1, 0, 0]
            outname: string, optional
                Name of the transformed map. Default is transformed.mrc.

        Outputs:
            transformed_map: float, 3D array
                3D Numpy array of floats.
    """
    from emda.ext import transform_map

    transformed_map = transform_map.map_transform(
        mapname, tra, rot, tuple(axr), outname
    )
    return transformed_map


def halfmap_fsc(half1name, half2name, filename=None, maskname=None):
    """Computes Fourier Shell Correlation (FSC) using half maps.

    Computes Fourier Shell Correlation (FSC) using half maps.
    FSC is not corrected for mask effect in this implementation.

    Arguments:
        Inputs:
            half1name: string
                Name of the half map 1.
            half2name: string
                Name of the half map 2.
            filename: string
                If present, statistics will be printed into this file.
            maskname: String
                If present, input maps will be masked before computing FSC.

        Outputs:
            res_arr: float, 1D array
                Linear array of resolution in Angstrom units.
            bin_fsc: float, 1D array
                Linear array of FSC in each resolution bin.
    """
    import os

    uc, arr1, _ = iotools.read_map(half1name)
    uc, arr2, _ = iotools.read_map(half2name)
    if maskname is not None:
        _, mask, _ = read_map(maskname)
        arr1 = arr1 * mask
        arr2 = arr2 * mask
    hf1 = np.fft.fftshift(np.fft.fftn(arr1))
    hf2 = np.fft.fftshift(np.fft.fftn(arr2))
    nbin, res_arr, bin_idx = restools.get_resolution_array(uc, hf1)
    (
        _,
        _,
        noisevar,
        signalvar,
        totalvar,
        bin_fsc,
        bincount,
    ) = fcodes_fast.calc_fsc_using_halfmaps(
        hf1, hf2, bin_idx, nbin, debug_mode, hf1.shape[0], hf1.shape[1], hf1.shape[2]
    )
    if filename is not None:
        tdata = open(filename, "w")
        tdata.write("halfmap1 file: %s\n" % os.path.abspath(half1name))
        tdata.write("halfmap2 file: %s\n" % os.path.abspath(half2name))
        tdata.write("\n")
        tdata.write("bin # \n")
        tdata.write("resolution (Ang.) \n")
        tdata.write("signal variance \n")
        tdata.write("noise variance \n")
        tdata.write("total variance \n")
        tdata.write("halfmap fsc \n")
        tdata.write("# reflx \n")
        i = -1
        for sv, nv, tv, fsci, nfc in zip(
            signalvar, noisevar, totalvar, bin_fsc, bincount
        ):
            i += 1
            tdata.write(
                "{:-3d} {:-6.2f} {:-14.4f} {:-14.4f} {:-14.4f} {:-14.4f} {:-10d}\n".format(
                    i, res_arr[i], sv, nv, tv, fsci, nfc
                )
            )
    print("Bin    Resolution     FSC")
    for i in range(len(res_arr)):
        print("{:5d} {:6.2f} {:14.4f}".format(i, res_arr[i], bin_fsc[i]))
    return res_arr, bin_fsc


def get_variance(half1name, half2name, filename=None, maskname=None):
    """Returns noise and signal variances of half maps.

    Returns noise and signal variances of half maps. Return values are not
    corrected for full map.

    Arguments:
        Inputs:
            half1name: string
                Name of the half map 1.
            half2name: string
                Name of the half map 2.
            filename: string
                If present, statistics will be printed into this file.
            maskname: String
                If present, input maps will be masked before computing variances.

        Outputs:
            res_arr: float, 1D array
                Linear array of resolution in Angstrom units.
            noisevar: float, 1D array
                Linear array of noise variance in each resolution bin.
            signalvar: float, 1D array
                Linear array of signal variance in each resolution bin.
    """
    import os

    uc, arr1, _ = iotools.read_map(half1name)
    uc, arr2, _ = iotools.read_map(half2name)
    if maskname is not None:
        _, mask, _ = read_map(maskname)
        arr1 = arr1 * mask
        arr2 = arr2 * mask
    hf1 = np.fft.fftshift(np.fft.fftn(arr1))
    hf2 = np.fft.fftshift(np.fft.fftn(arr2))
    nbin, res_arr, bin_idx = restools.get_resolution_array(uc, hf1)
    (
        _,
        _,
        noisevar,
        signalvar,
        totalvar,
        bin_fsc,
        bincount,
    ) = fcodes_fast.calc_fsc_using_halfmaps(
        hf1, hf2, bin_idx, nbin, debug_mode, hf1.shape[0], hf1.shape[1], hf1.shape[2]
    )
    if filename is not None:
        tdata = open(filename, "w")
        tdata.write("halfmap1 file: %s\n" % os.path.abspath(half1name))
        tdata.write("halfmap2 file: %s\n" % os.path.abspath(half2name))
        tdata.write("\n")
        tdata.write("bin # \n")
        tdata.write("resolution (Ang.) \n")
        tdata.write("signal variance \n")
        tdata.write("noise variance \n")
        tdata.write("total variance \n")
        tdata.write("halfmap fsc \n")
        tdata.write("# reflx \n")
        i = -1
        for sv, nv, tv, fsci, nfc in zip(
            signalvar, noisevar, totalvar, bin_fsc, bincount
        ):
            i += 1
            tdata.write(
                "{:-3d} {:-6.2f} {:-14.4f} {:-14.4f} {:-14.4f} {:-14.4f} {:-10d}\n".format(
                    i, res_arr[i], sv, nv, tv, fsci, nfc
                )
            )
    return res_arr, noisevar, signalvar


def twomap_fsc(map1name, map2name, fobj=None, xmlobj=None):
    """Returns Fourier Shell Correlation (FSC) between any two maps.

    Computes Fourier Shell Correlation (FSC) using any two maps.

    Arguments:
        Inputs:
            map1name: string
                Name of the map 1.
            map2name: string
                Name of the map 2.
            fobj: file object for logging
                If present, statistics will be printed into this file.
            xmlobj: xml object
                If present, statistics will be printed into an XML file.

        Outputs:
            res_arr: float, 1D array
                Linear array of resolution in Angstrom units.
            bin_fsc: float, 1D array
                Linear array of FSC in each resolution bin.
    """
    import os

    uc, arr1, _ = iotools.read_map(map1name)
    uc, arr2, _ = iotools.read_map(map2name)
    f1 = np.fft.fftshift(np.fft.fftn(arr1))
    f2 = np.fft.fftshift(np.fft.fftn(arr2))
    nbin, res_arr, bin_idx = restools.get_resolution_array(uc, f1)
    bin_fsc, _ = fsc.anytwomaps_fsc_covariance(f1, f2, bin_idx, nbin)
    if xmlobj is not None:
        xmlobj.map1path = os.path.abspath(map1name)
        xmlobj.map2path = os.path.abspath(map2name)
        xmlobj.res_arr = res_arr
        xmlobj.fsc = bin_fsc
        # xmlobj.outmap = "fullmap.mtz" # this file may not present
        xmlobj.write_xml()
    if fobj is not None:
        fobj.write("map1 file: %s\n" % os.path.abspath(map1name))
        fobj.write("map2 file: %s\n" % os.path.abspath(map2name))
        fobj.write("\n")
        fobj.write("bin # \n")
        fobj.write("resolution (Ang.) \n")
        fobj.write("fsc \n")
        for ibin, fsc1 in enumerate(bin_fsc):
            fobj.write("{:5d} {:6.2f} {:6.3f}\n".format(ibin, res_arr[ibin], fsc1))
    print("Bin      Resolution     FSC")
    for ibin, fsc2 in enumerate(bin_fsc):
        print("{:5d} {:6.2f} {:6.3f}".format(ibin, res_arr[ibin], fsc2))
    return res_arr, bin_fsc


def balbes_data(map1name, map2name, fsccutoff=0.5, mode="half"):
    """Returns data required for Balbes pipeline.

    Required data is output with their references in EMDA.xml.

    Arguments:
        Inputs:
            map1name: string
                Name of the map 1.
            map2name: string
                Name of the map 2.
            fsccutoff: float, optional
                FSC of desired resolution. Defualt is 0.5
            mode: string
                Mode can be either 'half' or 'any'. If the input maps are
                the half maps, mode should be 'half'. Otherwise, mode should be 'any'.
                Default mode is half.

        Outputs:
            Outputs EMDA.xml containing data and references to other data.
            No return variables.
    """
    from emda.ext import xmlclass

    if mode == "half":
        prepare_refmac_data(hf1name=map1name, hf2name=map2name, fsccutoff=fsccutoff)
    else:
        xml = xmlclass.Xml()
        res_arr, bin_fsc = twomap_fsc(map1name=map1name, map2name=map2name, xmlobj=xml)


def singlemap_fsc(map1name, knl=3):
    """Returns Fourier Shell Correlation (FSC) of a map.

    Computes Fourier Shell Correlation (FSC) between a map and its
    reconstituted other half from neighbough Fourier coefficients.
    This method can be used to estimate FSC based resolution. However,
    results seem to be reliable when an unfiltered map is used.

    Arguments:
        Inputs:
            map1name: string
                Name of the map.
            knl: integer, optional
                Radius of the integrating kernel.

        Outputs:
            res_arr: float, 1D array
                Linear array of resolution in Angstrom units.
            bin_fsc: float, 1D array
                Linear array of FSC in each resolution bin.
            Outputs reconstituted map as 'fakehalf.mrc'
    """
    from scipy import interpolate
    from emda.ext import fakehalf
    from emda.ext.mapfit.mapaverage import set_array

    uc, arr1, origin = iotools.read_map(map1name)
    f1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr1)))
    f2 = fakehalf.fakehalf(f_map=f1, knl=knl)
    data2write = np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(f2))))
    write_mrc(data2write, "fakehalf.mrc", uc, origin)
    nbin, res_arr, bin_idx = restools.get_resolution_array(uc, f1)
    bin_fsc, _, _, _, _, _ = fsc.halfmaps_fsc_variance(f1, f2, bin_idx, nbin)
    print("Resolution bin     FSC")
    for i, _ in enumerate(res_arr):
        print("{:.2f} {:.4f}".format(res_arr[i], bin_fsc[i]))
    # deciding resolution
    bin_fsc_trunc = set_array(bin_fsc, thresh=0.15)
    dist05 = np.sqrt((bin_fsc_trunc - 0.5) ** 2)
    indx = fakehalf.get_index(dist05)
    lim1 = bin_fsc[indx]
    res1 = res_arr[indx]
    if lim1 > 0.5:
        lim2 = bin_fsc[indx + 1]
        res2 = res_arr[indx + 1]
        fsc_seq = [lim1, lim2]
        res_seq = [res1, res2]
    elif lim1 < 0.5:
        lim2 = bin_fsc[indx - 1]
        res2 = res_arr[indx - 1]
        fsc_seq = [lim2, lim1]
        res_seq = [res2, res1]
    f = interpolate.interp1d(fsc_seq, res_seq)
    map_resol = f(0.5)
    print("Map resolution (A): ", map_resol)
    return res_arr, bin_fsc, map_resol


def mask_from_halfmaps(uc, half1, half2, radius=9, norm=False, iter=1, thresh=None):
    """Generates a mask from half maps.

    Generates a mask from half maps based on real space local correlation.

    Arguments:
        Inputs:
            uc: float, 1D array
                Unit cell parameters.
            half1: float, 3D array
                Half map 1 data.
            half2: float, 3D array
                Half map 2 data.
            radius: integer, optional
                Radius of integrating kernel in voxels. Default is 9.
            norm: bool, optional
                If true, normalized maps will be used to generate correlation mask.
                Default is False.
            iter: integer,optional
                Number of dilation cycles. Default is 1 cycle.
            thresh: float, optional
                Correlation cutoff for mask generation. Program automatically
                decides the best value, however, user can overwrite this.

        Outputs:
            mask: float, 3D array
                3D Numpy array of correlation mask.
    """
    from emda.ext import realsp_local, maskmap_class

    arr1, arr2 = half1, half2
    # normalized maps
    if norm:
        hf1 = np.fft.fftshift(np.fft.fftn(arr1))
        hf2 = np.fft.fftshift(np.fft.fftn(arr2))
        arr1, arr2 = realsp_local.hfdata_normalized(hf1=hf1, hf2=hf2, uc=uc)
        write_mrc(arr1, "normarr1.mrc", uc)
    obj_maskmap = maskmap_class.MaskedMaps()
    obj_maskmap.generate_mask(arr1, arr2, smax=radius, iter=iter, threshold=thresh)
    mask = obj_maskmap.mask
    return mask


def mask_from_map(
    uc,
    arr,
    kern=5,
    resol=15,
    filter="butterworth",
    order=1,
    prob=0.99,
    itr=3,
    orig=None,
):
    """Generates a mask from a map.

    Generates a mask from a map.

    Arguments:
        Inputs:
            uc: float, 1D array
                Unit cell parameters.
            arr: float, 3D array
                Map data.
            half2: float, 3D array
                Half map 2 data.
            kern: integer, optional
                Radius of integrating kernel in voxels. Default is 5.
            resol: float, optional
                Resolution cutoff for lowpass filtering in Angstrom units.
                Default is 15 Angstrom.
            filter: string,optional
                Filter type to use with lowpass filtering. Default is butterworth.
            order: integer, optional
                Butterworth filter order. Default is 1.
            prob: float, optional
                Cumulative probability cutoff to decide the density threshold.
                Default value is 0.99.
            itr: integer, optional
                Number of dilation cycles. Default is 3 cycles.
            orig: list of three integer values.
                Map origin. e.g. [0, 0, 0]

        Outputs:
            mask: float, 3D array
                3D Numpy array of the mask.
            Outputs lowpass.mrc and mapmask.mrc files.
    """
    from emda.ext import maskmap_class

    _, arrlp = lowpass_map(uc, arr, resol, filter, order=order)
    write_mrc(arrlp, "lowpass.mrc", uc, orig)
    mask = maskmap_class.mapmask(arr=arrlp, uc=uc, kern_rad=kern, prob=prob, itr=itr)
    write_mrc(mask, "mapmask.mrc", uc, orig)
    return mask


def sphere_kernel_softedge(radius=5):
    """Generates a soft-edged spherical kernel.

    Arguments:
        Inputs:
            radius: integer, optional
                Radius of integrating kernel in voxels. Default is 5.

        Outputs:
            kernel: float, 3D array
                3D Numpy array of spherical kernel.
    """
    kernel = restools.create_soft_edged_kernel_pxl(radius)
    return kernel


def overlay_maps(
    maplist,
    rot=0.0,
    ncy=5,
    res=6,
    interp="linear",
    hfm=False,
    masklist=None,
    tra=None,
    axr=None,
    fobj=None,
):
    """Superimposes several maps.

    Superimposes several maps using a likelihood function. All maps are
    overlaid on the first map.

    Arguments:
        Inputs:
            maplist: list
                List of maps to overlay.
            masklist: list
                List of masks to apply on maps.
            rot: float, optional
                Initial rotation in degrees. Default is 0.0.
            axr: list, optional
                Rotation axis. Default is [1, 0, 0].
            tra: list, optional
                Translation vector in fractional units. Default is [0.0, 0.0, 0.0]
            res: float, optional
                Fit start resolution in Angstrom units. Default is 6.0 Angstrom.
            ncy: integer, optional
                Number of fitting cycles. Default is 5.
            interp: string, optional
                Interpolation type either "linear" or "cubic".
                Default is linear.
            hfm: bool, optional
                If True, overlay will be carried out on half maps. In this case,
                maplist will contain half maps.
                e.g. [map1_half1.mrc, map1_half2.mrc, map2_half1.mrc, map2_half2.mrc, ...].
                masklist will contain masks for each map. e.g. [map1_mask.mrc, map2_mask.mrc].
                The length of masklist should be equal to half the length of maplist.
                If False, uses full maps for overlay. Default is False.
            fobj: string
                File object for logging. If None given, EMDA_overlay.txt will be output.

        Outputs:
            Outputs a series of overlaid maps (fitted_map_?.mrc).
    """
    from emda.ext.mapfit import mapoverlay

    if axr is None:
        axr = [1, 0, 0]
    if tra is None:
        tra = [0.0, 0.0, 0.0]
    if fobj is None:
        fobj = open("EMDA_overlay.txt", "w")
    theta_init = [tuple(axr), rot]
    mapoverlay.main(
        maplist=maplist,
        masklist=masklist,
        ncycles=ncy,
        t_init=tra,
        theta_init=theta_init,
        smax=res,
        fobj=fobj,
        interp=interp,
        halfmaps=hfm,
    )


def average_maps(
    maplist,
    rot=0.0,
    ncy=5,
    res=6,
    interp="linear",
    fit=True,
    tra=None,
    axr=None,
    fobj=None,
    masklist=None,
):
    """Calculates the best average maps using Bayesian principles.

    Calculates the best average map using Bayesian principles. This is done in two steps;
    1. Parameter estimation using a likelihood function, 2. Best map calculation.
    Parameter estimation is similar to map overlay where each map is brought onto
    static map by maximizing the overlap. The best maps are calculated using
    superimposed maps.

    Arguments:
        Inputs:
            maplist: list
                List of half maps to average.
            masklist: list, optional
                List of masks to apply on maps. len(masklist) == len(maplist) // 2
            rot: float, optional
                Initial rotation in degrees. Default is 0.0.
            axr: list, optional
                Rotation axis. Default is [1, 0, 0].
            tra: list, optional
                Translation vector in fractional units. Default is [0.0, 0.0, 0.0]
            res: float, optional
                Fit start resolution in Angstrom units. Default is 6.0 Angstrom.
            ncy: integer, optional
                Number of fitting cycles. Default is 5.
            interp: string, optional
                Interpolation type either "linear" or "cubic".
                Default is linear.
            fobj: string
                File object for logging. If None given, EMDA_average.txt will be output.
            fit: bool, optional
                If True, map fitting will be carried out before average map calculation.
                Default is True.

        Outputs:
            Outputs a series of overlaid maps (fitted_map_?.mrc).
            Also, outputs a series of average maps (avgmap_?.mrc)
    """
    from emda.ext.mapfit import mapaverage

    if axr is None:
        axr = [1, 0, 0]
    if tra is None:
        tra = [0.0, 0.0, 0.0]
    if fobj is None:
        fobj = open("EMDA_average.txt", "w")
    theta_init = [tuple(axr), rot]
    mapaverage.main(
        maplist=maplist,
        masklist=masklist,
        ncycles=ncy,
        t_init=tra,
        theta_init=theta_init,
        smax=res,
        fobj=fobj,
        interp=interp,
        fit=True,
    )


def realsp_correlation(
    half1map,
    half2map,
    kernel_size=5,
    norm=False,
    lig=False,
    model=None,
    model_resol=None,
    mask_map=None,
    lgf=None,
):
    """Calculates local correlation in real/image space.

    Arguments:
        Inputs:
            half1map: string
                Name of half map 1.
            half1map: string
                Name of half map 2.
            kernel_size: integer, optional
                Radius of integration kernal in pixels. Default is 5.
            norm: bool, optional
                If True, correlation will be carried out on normalized maps.
                Default is False.
            model: string, optional
                An argument for model based map calculation using REFMAC.
                Name of model file (cif/pdb). If present, map-model local
                correlation will be calculated.
            model_resol: float, optional
                An argument for model based map calculation using REFMAC.
                Resolution to calculate model based map. If absent, FSC based
                resolution cutoff will be employed.
            mask_map: string, optional
                Mask file to apply on correlation maps. If not given, correlation based
                mask will be employed.
            lig: bool, optional
                An argument for model based map calculation using REFMAC.
                Set True, if there is a ligand in the model, but no description.
                Default is False.
            lgf: string, optional
                An argument for model based map calculation using REFMAC.
                Ligand description file (cif).

        Outputs:
            Following maps are written out:
            rcc_halfmap_smax?.mrc - reals space half map local correlation.
            rcc_fullmap_smax?.mrc - correlation map corrected to full map
                using the formula 2 x FSC(half) / (1 + FSC(half)).
            If a model included, then
            rcc_mapmodel_smax?.mrc - local correlation map between model and
                full map.
            rcc_truemapmodel_smax?.mrc - truemap-model correaltion map for
                validation purpose.
    """
    from emda.ext import realsp_local

    realsp_local.rcc(
        half1_map=half1map,
        half2_map=half2map,
        kernel_size=kernel_size,
        norm=norm,
        lig=lig,
        model=model,
        model_resol=model_resol,
        mask_map=mask_map,
        lgf=lgf,
    )


def realsp_correlation_mapmodel(
    fullmap, model, resol, kernel_size=5, lig=False, trimpx=1, mask_map=None, lgf=None
):
    """Calculates real space local correlation between map and model.

    Arguments:
        Inputs:
            fullmap: string
                Name of the map.
            model: string
                An argument for model based map calculation using REFMAC.
                Name of model file (cif/pdb/ent/mtz/mrc).
            resol: float
                An argument for model based map calculation using REFMAC.
                Resolution to calculate model based map.
            kernel_size: integer, optional
                Radius of integration kernal in pixels. Default is 5.
            mask_map: string, optional
                Mask file to apply on correlation maps. If not given, a spherical mask
                will be employed. Default radius = (map.shape[0] // 2) - trimpx
            trimpx: integer, optional
                Parameter to adjust the radius (in pixels) of the spherical mask
                in the absence of mask_map argument. default is 1.
            lig: bool, optional
                An argument for model based map calculation using REFMAC.
                Set True, if there is a ligand in the model, but no description.
                Default is False.
            lgf: string, optional
                An argument for model based map calculation using REFMAC.
                Ligand description file (cif).

        Outputs:
            Following maps are written out:
            modelmap.mrc - model based map.
            rcc_mapmodel.mrc - real space local correlation map.
    """
    from emda.ext import realsp_local

    realsp_local.mapmodel_rcc(
        fullmap=fullmap,
        model=model,
        kernel_size=kernel_size,
        lig=lig,
        resol=resol,
        mask_map=mask_map,
        lgf=lgf,
        trim_px=trimpx,
    )


def fouriersp_correlation(half1_map, half2_map, kernel_size=5, mask=None):
    """Calculates Fourier space local correlation using half maps.

    Arguments:
        Inputs:
            half1_map: string
                Name of half map 1.
            half2_map: string
                Name of half map 2.
            kernel_size: integer, optional
                Radius of integration kernal. Default is 5.

        Outputs:
            Following maps are written out:
            fouriercorr3d_halfmaps.mrc - local correlation in half maps.
            fouriercorr3d_fullmap.mrc - local correlation in full map
                using the formula 2 x FSC(half) / (1 + FSC(half)).
            fouriercorr3d_truemap.mrc - local correlation in true map.
                Useful for validation purpose.
    """
    from emda.ext import fouriersp_local

    fouriersp_local.fcc(
        half1_map=half1_map, half2_map=half2_map, kernel_size=kernel_size, maskmap=mask
    )


def map_model_validate(
    half1map,
    half2map,
    modelfpdb,
    bfac=0.0,
    lig=False,
    model1pdb=None,
    mask=None,
    modelresol=None,
    lgf=None,
):
    """Calculates various FSCs for maps and model validation.

    Arguments:
        Inputs:
            half1map: string
                Name of half map 1.
            half2map: string
                Name of half map 2.
            modelfpdb: string
                Name of the model refined against full map in cif/pdb/ent
                formats.
            model1pdb: string, optional
                Name of the model refined against one half map in cif/pdb/ent
                formats. If included, FSC between that and half maps will be
                calculated.
            mask: string, optional
                Name of the mask file. It will apply on half maps before
                computing FSC. If not included, a correlation based masked will
                employed.
            modelresol: float, optional
                An argument for model based map calculation using REFMAC.
                Resolution to calculate model based map. If not specified, an FSC
                based cutoff will be used.
            bfac: float, optional
                An overall B-factor for model map. Default is 0.0
            lig: bool, optional
                An argument for model based map calculation using REFMAC.
                Set True, if there is a ligand in the model, but no description.
                Default is False.
            lgf: string, optional
                An argument for model based map calculation using REFMAC.
                Ligand description file (cif).

        Outputs:
            fsc_list: list
                List of FSCs is returned. If len(fsc_list) is 4,
                FSC lables are as follows:
                0 - half maps FSC
                1 - half1map - model1 FSC
                2 - half2map - model1 FSC
                3 - fullmap-fullmodel FSC
                If len(fsc_list) is 2, only 0 and 3 contains.
            Outputs FSCs in allmap_fsc_modelvsmap.eps
    """
    from emda.ext import map_fsc

    fsc_list = map_fsc.map_model_fsc(
        half1_map=half1map,
        half2_map=half2map,
        modelf_pdb=modelfpdb,
        bfac=bfac,
        lig=lig,
        model1_pdb=model1pdb,
        mask_map=mask,
        model_resol=modelresol,
        lgf=lgf,
    )
    return fsc_list


def difference_map(maplist, masklist, smax, mode="ampli"):
    """Calculates difference map.

    Arguments:
        Inputs:
            maplist: string
                List of map names to calculate difference maps.
            masklist: string
                List of masks to apply on maps.
            smax: float
                Resolution to which difference map be calculated.
            mode: string, optional
                Different modes to scale maps. Three difference modes are supported.
                'ampli' - scale between maps is based on amplitudes [Default].
                'power' - scale between maps is based on powers (intensities).
                'norm' - normalized maps are used to calculate difference map.

        Outputs:
            Outputs diffence maps and initial maps after scaling in MRC format.
    """
    from emda.ext import difference

    uc, arr1, origin = read_map(maplist[0])
    _, arr2, _ = iotools.read_map(maplist[1])
    _, msk1, _ = iotools.read_map(masklist[0])
    _, msk2, _ = iotools.read_map(masklist[1])
    f1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr1)))  # * msk1)))
    f2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr2)))  # * msk2)))
    if mode == "power":
        dm1_dm2, dm2_dm1 = difference.diffmap_scalebypower(
            f1=f1, f2=f2, cell=uc, origin=origin, smax=smax
        )
        # calculate map rmsd
        masked_mean = np.sum(dm1_dm2 * msk1) / np.sum(msk1)
        diff = (dm1_dm2 - masked_mean) * msk1
        rmsd = np.sqrt(np.sum(diff * diff) / np.sum(msk1))
        print("rmsd: ", rmsd)

    if mode == "norm":
        diffmap = difference.diffmap_normalisedsf(
            f1=f1, f2=f2, cell=uc, origin=origin, smax=smax
        )
        list_maps = []
        for i in range(diffmap.shape[3]):
            map = np.real(
                np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(diffmap[:, :, :, i])))
            )
            list_maps.append(map)
        # calculate map rmsd
        masked_mean = np.sum(list_maps[0] * msk1) / np.sum(msk1)
        diff = (list_maps[0] - masked_mean) * msk1
        rmsd = np.sqrt(np.sum(diff * diff) / np.sum(msk1))
        print("rmsd: ", rmsd)
        masked_mean = np.sum(list_maps[1] * msk2) / np.sum(msk2)
        diff = (list_maps[1] - masked_mean) * msk2
        rmsd = np.sqrt(np.sum(diff * diff) / np.sum(msk2))
        print("rmsd of diffmap_m1-m2_amp: ", rmsd)

        iotools.write_mrc(list_maps[0] * msk1, "diffmap_m1-m2_nrm.mrc", uc, origin)
        iotools.write_mrc(list_maps[1] * msk2, "diffmap_m2-m1_nrm.mrc", uc, origin)
        iotools.write_mrc(list_maps[2] * msk1, "map1.mrc", uc, origin)
        iotools.write_mrc(list_maps[3] * msk2, "map2.mrc", uc, origin)

    if mode == "ampli":
        diffmap = difference.diffmap_scalebyampli(
            f1=f1, f2=f2, cell=uc, origin=origin, smax=smax
        )
        list_maps = []
        for i in range(diffmap.shape[3]):
            map = np.real(
                np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(diffmap[:, :, :, i])))
            )
            list_maps.append(map)
        # calculate map rmsd
        masked_mean = np.sum(list_maps[0] * msk1) / np.sum(msk1)
        diff = (list_maps[0] - masked_mean) * msk1
        rmsd = np.sqrt(np.sum(diff * diff) / np.sum(msk1))
        print("rmsd of diffmap_m1-m2_amp: ", rmsd)
        masked_mean = np.sum(list_maps[1] * msk2) / np.sum(msk2)
        diff = (list_maps[1] - masked_mean) * msk2
        rmsd = np.sqrt(np.sum(diff * diff) / np.sum(msk2))
        print("rmsd of diffmap_m2-m1_amp: ", rmsd)
        # difference map output
        iotools.write_mrc(list_maps[0] * msk1, "diffmap_m1-m2_amp.mrc", uc, origin)
        iotools.write_mrc(list_maps[1] * msk2, "diffmap_m2-m1_amp.mrc", uc, origin)
        iotools.write_mrc(list_maps[2] * msk1, "map1.mrc", uc, origin)
        iotools.write_mrc(list_maps[3] * msk2, "map2.mrc", uc, origin)


def applymask(mapname, maskname, outname):
    uc, arr1, origin = iotools.read_map(mapname)
    _, mask, _ = read_map(maskname)
    iotools.write_mrc(arr1 * mask, outname, uc, origin)


def scale_map2map(staticmap, map2scale, outfile):
    # this need further options
    from emda.ext.scale_maps import scale_twomaps_by_power, transfer_power

    uc, arr1, origin = iotools.read_map(staticmap)
    uc, arr2, origin = iotools.read_map(map2scale)
    f1 = np.fft.fftshift(np.fft.fftn(arr1))
    f2 = np.fft.fftshift(np.fft.fftn(arr2))
    nbin, res_arr, bin_idx = restools.get_resolution_array(uc, f1)
    scale_grid = transfer_power(
        bin_idx,
        res_arr,
        scale_twomaps_by_power(f1, f2, bin_idx=bin_idx, res_arr=res_arr),
    )
    data2write = np.real(np.fft.ifftn(np.fft.ifftshift(f2 * scale_grid)))
    iotools.write_mrc(data2write, outfile, uc, origin)
    return data2write


def bestmap(hf1name, hf2name, outfile, mode=1, knl=5, mask=None):
    from emda.ext import bestmap

    if mask is None:
        msk = 1.0
    else:
        _, msk, _ = read_map(mask)
    uc, arr1, origin = iotools.read_map(hf1name)
    uc, arr2, origin = iotools.read_map(hf2name)
    if mask:
        print("mask is not included in FSC calculation")
    f1 = np.fft.fftshift(np.fft.fftn(arr1))  # * msk))
    f2 = np.fft.fftshift(np.fft.fftn(arr2))  # * msk))
    if mode == 1:
        nbin, res_arr, bin_idx = restools.get_resolution_array(uc, f1)
        f_map = bestmap.bestmap(f1=f1, f2=f2, bin_idx=bin_idx, nbin=nbin, mode=mode)
    elif mode == 2:
        f_map = bestmap.bestmap(f1=f1, f2=f2, mode=mode, kernel_size=knl)
    data2write = np.real(np.fft.ifftn(np.fft.ifftshift(f_map))) * msk
    iotools.write_mrc(data2write, outfile, uc, origin)


def predict_fsc(hf1name, hf2name, nparticles=None, bfac=None, mask=None):
    uc, arr1, _ = iotools.read_map(hf1name)
    uc, arr2, _ = iotools.read_map(hf2name)
    if mask is not None:
        _, msk, _ = read_map(mask)
    else:
        msk = 1.0
    if nparticles is None and bfac is None:
        print("Either nparticles or bfac needs to be given!")
        exit()
    f1 = np.fft.fftshift(np.fft.fftn(arr1 * msk))
    f2 = np.fft.fftshift(np.fft.fftn(arr2 * msk))
    nbin, res_arr, bin_idx = restools.get_resolution_array(uc, f1)
    if nparticles is not None:
        bfac = None
        nparticles = 1.0 / np.asarray(nparticles, dtype="float")
        fsc_lst = fsc.predict_fsc(
            hf1=f1,
            hf2=f2,
            bin_idx=bin_idx,
            nbin=nbin,
            nparticles=nparticles,
            res_arr=res_arr,
        )
        labels = [str(i) for i in nparticles]
    if bfac is not None:
        fsc_lst = fsc.predict_fsc(
            hf1=f1, hf2=f2, bin_idx=bin_idx, nbin=nbin, bfac=bfac, res_arr=res_arr
        )
        labels = [str(i) for i in bfac]
    labels.append("reference")
    plotter.plot_nlines(
        res_arr, fsc_lst, curve_label=labels, mapname="fsc_predicted.eps"
    )
    return fsc_lst, res_arr, bin_idx, nbin


def prepare_refmac_data(
    hf1name,
    hf2name,
    outfile="fullmap.mtz",
    bfac=None,
    maskname=None,
    xmlobj=None,
    fsccutoff=None,
):
    import os
    from emda.ext import xmlclass

    xml = xmlclass.Xml()
    uc, arr1, origin = iotools.read_map(hf1name)
    uc, arr2, origin = iotools.read_map(hf2name)
    xml.map1path = os.path.abspath(hf1name)
    xml.map2path = os.path.abspath(hf2name)
    if maskname is None:
        msk = 1.0
    if maskname is not None:
        _, msk, _ = read_map(maskname)
        if arr1.shape != msk.shape:
            print("mask dim and map dim do not match!")
            print("map dim:", arr1.shape, "mask dim:", msk.shape)
            exit()
    if bfac is None:
        bfac = np.array([0.0], dtype="float")
    else:
        bfac = np.asarray(bfac, dtype="float")
        bfac = np.insert(bfac, 0, 0.0)
    hf1 = np.fft.fftshift(np.fft.fftn(arr1 * msk))
    hf2 = np.fft.fftshift(np.fft.fftn(arr2 * msk))
    # stats from half maps
    nx, ny, nz = hf1.shape
    maxbin = np.amax(np.array([nx // 2, ny // 2, nz // 2]))
    nbin, res_arr, bin_idx, sgrid = fcodes_fast.resolution_grid(
        uc, debug_mode, maxbin, nx, ny, nz
    )
    res_arr = res_arr[:nbin]
    _, _, nvar, svar, tvar, binfsc, bincount = fcodes_fast.calc_fsc_using_halfmaps(
        hf1, hf2, bin_idx, nbin, debug_mode, hf1.shape[0], hf1.shape[1], hf1.shape[2]
    )
    xml.res_arr = res_arr
    xml.fsc = binfsc
    xml.outmap = outfile
    # determine resolution
    bin_fsc = binfsc[binfsc > 0.1]
    if len(bin_fsc) > 0:
        if fsccutoff is None:
            fsccutoff = 0.5
        dist500 = np.sqrt((bin_fsc - float(fsccutoff)) ** 2)
        dist143 = np.sqrt((bin_fsc - 0.143) ** 2)
        xml.fsccutoff1 = float(0.143)
        xml.fsccutoff2 = float(fsccutoff)
        xml.mapres1 = res_arr[np.argmin(dist143)]
        xml.mapres2 = res_arr[np.argmin(dist500)]
    xml.write_xml()

    tdata = open("table_variances.txt", "w")
    tdata.write("halfmap1 file: %s\n" % os.path.abspath(hf1name))
    tdata.write("halfmap2 file: %s\n" % os.path.abspath(hf2name))
    tdata.write("\n")
    tdata.write("bin # \n")
    tdata.write("resolution (Ang.) \n")
    tdata.write("signal variance \n")
    tdata.write("noise variance \n")
    tdata.write("total variance \n")
    tdata.write("halfmap fsc \n")
    tdata.write("# reflx \n")
    for i in range(len(res_arr)):
        sv = svar[i]
        nv = nvar[i]
        tv = tvar[i]
        fsc = binfsc[i]
        nfc = bincount[i]
        tdata.write(
            "{:-3d} {:-6.2f} {:-14.4f} {:-14.4f} {:-14.4f} {:-14.4f} {:-10d}\n".format(
                i, res_arr[i], sv, nv, tv, fsc, nfc
            )
        )
    print("Bin    Resolution     FSC")
    for i in range(len(res_arr)):
        print("{:5d} {:6.2f} {:14.4f}".format(i, res_arr[i], binfsc[i]))
    # output mtz file
    iotools.write_3d2mtz_refmac(
        uc, sgrid, (hf1 + hf2) / 2.0, (hf1 - hf2), bfac, outfile=outfile
    )


def overall_cc(map1name, map2name, space="real", maskname=None):
    from emda.ext import cc

    uc, arr1, origin = iotools.read_map(map1name)
    uc, arr2, origin = iotools.read_map(map2name)
    if maskname is not None:
        uc, msk, origin = read_map(maskname)
        arr1 = arr1 * msk
        arr2 = arr2 * msk
    if space == "fourier":
        print("Overall CC calculation in Fourier space")
        f1 = np.fft.fftn(arr1)
        f2 = np.fft.fftn(arr2)
        occ, hocc = cc.cc_overall_fouriersp(f1=f1, f2=f2)
        print("Overall Correlation in Fourier space= ", occ)
    else:
        print("Overall CC calculation in Real/Image space")
        occ, hocc = cc.cc_overall_realsp(map1=arr1, map2=arr2)
        print("Overall Correlation in real space= ", occ)
    return occ, hocc


def mirror_map(mapname):
    # gives the inverted copy of the map
    uc, arr, origin = iotools.read_map(mapname)
    data = np.real(np.fft.ifftn(np.conjugate(np.fft.fftn(arr))))
    iotools.write_mrc(data, "mirror.mrc", uc, origin)


def model2map(modelxyz, dim, resol, cell, bfac=0.0, lig=False, ligfile=None):
    import gemmi as gm

    # check for valid sampling:
    if np.any(np.mod(dim, 2)) != 0:
        dim = dim + 1
    # check for minimum sampling
    min_pix_size = resol / 2  # in Angstrom
    min_dim = np.asarray(cell[:3], dtype="float") / min_pix_size
    min_dim = np.ceil(min_dim).astype(int)
    if np.any(np.mod(min_dim, 2)) != 0:
        min_dim = min_dim + 1
    if min_dim[0] > dim[0]:
        print("Minimum dim should be: ", min_dim)
        exit()
    # replace/add cell and write model.cif
    a, b, c = cell[:3]
    structure = gm.read_structure(modelxyz)
    structure.cell.set(a, b, c, 90.0, 90.0, 90.0)
    structure.make_mmcif_document().write_file("model.cif")
    # run refmac using model.cif just created
    iotools.run_refmac_sfcalc("./model.cif", resol, bfac, lig=lig, ligfile=ligfile)
    modelmap = maptools.mtz2map("./sfcalc_from_crd.mtz", dim)
    return modelmap


def read_atomsf(atom, fpath=None):
    import subprocess

    if fpath is None:
        CMD = "echo $CLIBD"
        p = subprocess.Popen(
            CMD, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
        )
        list_of_strings = [
            x.decode("utf-8").rstrip("\n") for x in iter(p.stdout.readlines())
        ]
        fpath = list_of_strings[0] + "/atomsf.lib"
    z, ne, a, b, ier = iotools.read_atomsf(atom, fpath=fpath)
    return z, ne, a, b, ier


def compositemap(maps, masks):
    from emda.ext import composite

    composite.main(mapslist=maps, masklist=masks)


def mapmagnification(maplist, rmap):
    from emda.ext import magnification

    # magnification refinement
    maplist.append(rmap)
    magnification.main(maplist=maplist)
