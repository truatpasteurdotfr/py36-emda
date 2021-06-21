"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
   
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy as np
import fcodes_fast
import gemmi
import pandas
import mrcfile as mrc
from emda.config import debug_mode


def test():
    """ Tests iotools module installation. """

    print("iotools test ... Passed")


def read_map(mapname, fid=None):
    """Reads CCP4 type map (.map) or MRC type map.

    Arguments:
        Inputs:
            mapname: string
                CCP4/MRC map file name
        Outputs:
            unit_cell: float, 1D array
                Unit cell
            arr: float, 3D array
                Map values as Numpy array
            origin: list
                Map origin list
     """
    import mrcfile
    import numpy as np

    try:
        file = mrcfile.open(mapname)
        order = (file.header.mapc-1, file.header.mapr-1, file.header.maps-1)
        axes_order = "".join(["ZYX"[i] for i in order])
        arr = np.asarray(file.data, dtype="float")
        arr = np.moveaxis(a=arr, source=(0,1,2), destination=order)
        """ if file.header.mapc == 1:
            if file.header.mapr == 2 and file.header.maps == 3:
                axes_order = 'ZYX'
                arr = np.asarray(file.data, dtype="float")
            elif file.header.mapr == 3 and file.header.maps == 2:
                axes_order = 'ZXY'
                arr = np.asarray(file.data, dtype="float")
                arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-3,-1,-2])                
        elif file.header.mapc == 2:
            if file.header.mapr == 1 and file.header.maps == 3:
                axes_order = 'YZX'
                arr = np.asarray(file.data, dtype="float")
                arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-2,-3,-1])                
            elif file.header.mapr == 3 and file.header.maps == 1:
                axes_order = 'YXZ'
                arr = np.asarray(file.data, dtype="float")
                arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-2,-1,-3])
        elif file.header.mapc == 3:
            if file.header.mapr == 1 and file.header.maps == 2:
                axes_order = 'XZY'
                arr = np.asarray(file.data, dtype="float")
                arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-1,-3,-2])
            elif file.header.mapr == 2 and file.header.maps == 1:
                axes_order = 'XYZ'
                arr = np.asarray(file.data, dtype="float")
                arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-1,-2,-3])
        else:
            raise SystemExit("Wrong axes order. Stopping now...") """
        if fid is not None:
            fid.write('Axes order: %s\n' % (axes_order))
        unit_cell = np.zeros(6, dtype='float')
        cell = file.header.cella[['x', 'y', 'z']]
        unit_cell[:3] = cell.view(('f4', 3))
        # swapping a and c to compatible with ZYX convension
        unit_cell[0], unit_cell[2] = unit_cell[2], unit_cell[0]
        unit_cell[3:] = float(90)
        origin = [
                1 * file.header.nxstart,
                1 * file.header.nystart,
                1 * file.header.nzstart,
            ]
        file.close()
        print(mapname, arr.shape, unit_cell[:3])
        return unit_cell, arr, origin
    except FileNotFoundError as e:
        print(e)


def change_axesorder(arr, uc, axes_order, maporig):
    import numpy as np

    # Change Mmap axes order
    # input arr has ZYX order (EMDA convension)
    if axes_order == 'ZXY':
        arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-3,-1,-2])
        # cba --> cab
        uc[0], uc[1], uc[2] = uc[0], uc[2], uc[1]
        maporig[0], maporig[1], maporig[2] = maporig[0], maporig[2], maporig[2]
    if axes_order == 'YZX':
        arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-2,-3,-1]) 
        # cba --> bca
        uc[0], uc[1], uc[2] = uc[1], uc[0], uc[2]
        maporig[0], maporig[1], maporig[2] = maporig[1], maporig[0], maporig[2]              
    if axes_order == 'YXZ':
        arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-2,-1,-3])
        # cba --> bac
        uc[0], uc[1], uc[2] = uc[1], uc[2], uc[0]
        maporig[0], maporig[1], maporig[2] = maporig[1], maporig[2], maporig[0]
    if axes_order == 'XZY':
        arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-1,-3,-2])
        # cba --> acb
        uc[0], uc[1], uc[2] = uc[2], uc[0], uc[1]
        maporig[0], maporig[1], maporig[2] = maporig[2], maporig[0], maporig[1]
    if axes_order == 'XYZ':
        arr = np.moveaxis(a=arr, source=[0,1,2], destination=[-1,-2,-3])
        # cba --> abc
        uc[0], uc[1], uc[2] = uc[2], uc[1], uc[0]
        maporig[0], maporig[1], maporig[2] = maporig[2], maporig[1], maporig[0]
    else:
        raise SystemExit("Wrong axes order. Stopping now...")
    return arr, uc, maporig


def write_mrc(mapdata, filename, unit_cell, map_origin=None, label=False, axesorder='ZYX'):
    """ Writes 3D Numpy array into MRC file.

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
            label: bool, optional
                If True a text label is written out in the map header.
                Default is False
            axesorder: string, optional
                Axes order can be specified for the data to be written.
                By defualt EMDA write data in ZXY convension.
                With this argument, the axes order can be changed.

        Outputs:
            Outputs the MRC file
    """
    import numpy as np
    import mrcfile as mrc

    if map_origin is None:
        map_origin = [0.0, 0.0, 0.0]
    if axesorder != 'ZYX':
        mapdata, unit_cell, map_origin = change_axesorder(
            arr=mapdata, uc=unit_cell, axes_order=axesorder, maporig=map_origin)
    file = mrc.new(
        name=filename, data=np.float32(mapdata), compression=None, overwrite=True
    )
    file.header.cella.x = unit_cell[0]
    file.header.cella.y = unit_cell[1]
    file.header.cella.z = unit_cell[2]
    file.header.nxstart = map_origin[0]
    file.header.nystart = map_origin[1]
    file.header.nzstart = map_origin[2]
    if label:
        # correlation maps are written with effective correlation
        # range in the header for visualisation
        maxcc = np.max(mapdata)
        label1 = "{}{:5.3f}{}{:5.3f}{}".format(
            "EMDA_Color:linear: (#901010, 0.000) (#109010, ",
            maxcc / 2.0,
            ") (#101090, ",
            maxcc,
            ")",
        )
        file.header.label = label1
    file.close()


def read_mtz(mtzfile):
    """ Reads mtzfile and returns unit_cell and data in Pandas DataFrame.

    Arguments:
        Inputs:
            mtzfile: string
                MTZ file name
        Outputs:
            unit_cell: float, 1D array
                Unit cell
            data_frame: Pandas data frame
                Map values in Pandas Dataframe
    """
    try:
        mtz = gemmi.read_mtz_file(mtzfile)
        unit_cell = np.zeros(6, dtype="float")
        unit_cell[0] = mtz.dataset(0).cell.a
        unit_cell[1] = mtz.dataset(0).cell.b
        unit_cell[2] = mtz.dataset(0).cell.c
        unit_cell[3:] = float(90)
        all_data = np.array(mtz, copy=False)
        data_frame = pandas.DataFrame(data=all_data, columns=mtz.column_labels())
        return unit_cell, data_frame
    except FileNotFoundError as e:
        print(e)


def write_3d2mtz(unit_cell, mapdata, outfile="map2mtz.mtz", resol=None):
    """ Writes 3D Numpy array into MTZ file.

    Arguments:
        Inputs:
            unit_cell: float, 1D array
                Unit cell params.
            mapdata: complex, 3D array
                Map values to write.
            resol: float, optional
                Map will be output for this resolution.
                Default is up to Nyquist.

        Outputs: 
            outfile: string
            Output file name. Default is map2mtz.mtz.
    """
    from emda.core import restools

    nx, ny, nz = mapdata.shape
    assert nx == ny == nz
    nbin, res_arr, bin_idx = restools.get_resolution_array(unit_cell, mapdata)
    if resol is None:
        cbin = nx // 2
    else:
        dist = np.sqrt((res_arr - resol) ** 2)
        cbin = np.argmin(dist) + 1
        if cbin % 2 != 0: cbin += 1
        if nbin < cbin: cbin = nbin
    mtz = gemmi.Mtz()
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.cell.set(unit_cell[0], unit_cell[1], unit_cell[2], 90, 90, 90)
    mtz.add_dataset("HKL_base")
    for label in ["H", "K", "L"]:
        mtz.add_column(label, "H")
    mtz.add_column("Fout0", "F")
    mtz.add_column("Pout0", "P")
    # calling fortran function
    h, k, l, ampli, phase = fcodes_fast.prepare_hkl(mapdata, bin_idx, cbin, debug_mode, nx, ny, nz)
    print("hmin, hmax ", np.amin(h), np.amax(h))
    print("kmin, kmax ", np.amin(k), np.amax(k))
    print("lmin, lmax ", np.amin(l), np.amax(l))
    nrows = len(h)
    ncols = 5  # Change this according to what columns to write
    data = np.ndarray(shape=(nrows, ncols), dtype=object)
    data[:, 0] = -l.astype(int)
    data[:, 1] = -k.astype(int)
    data[:, 2] = -h.astype(int)
    data[:, 3] = ampli.astype(np.float32)
    data[:, 4] = phase.astype(np.float32)
    mtz.set_data(data)
    mtz.write_to_file(outfile)


def write_3d2mtz_full(unit_cell, hf1data, hf2data, outfile="halfnfull.mtz"):
    """ Writes several 3D Numpy arrays into an MTZ file.

    Arguments:
        Inputs:
            unit_cell: float, 1D array
                Unit cell params.
            hf1data: complex, 3D array
                Halfmap1 values to write.
            hf2data: complex, 3D array
                Halfmap1 values to write.

        Outputs:
            outfile: string
            Output file name. Default is halfnfull.mtz.
    """
    # mtz containing hf data and full data
    hf1, hf2 = hf1data, hf2data
    assert hf1.shape == hf2.shape
    nx, ny, nz = hf1.shape
    mtz = gemmi.Mtz()
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.cell.set(unit_cell[0], unit_cell[1], unit_cell[2], 90, 90, 90)
    mtz.add_dataset("HKL_base")
    for label in ["H", "K", "L"]:
        mtz.add_column(label, "H")
    for label in ["Fout" + str(i) for i in ["1", "2", "f"]]:
        mtz.add_column(label, "F")
    mtz.add_column("Poutf", "P")
    # calling fortran function
    f_arr = np.zeros((nx, ny, nz, 2), dtype=np.complex)
    for i, f in enumerate([hf1, hf2]):
        f_arr[:, :, :, i] = f
    hkl, ampli, phase = fcodes_fast.prepare_hkl2(f_arr, 1, nx, ny, nz, 2)
    nrows = ampli.shape[0]
    ncols = 5 + ampli.shape[1]  # Change this according to what columns to write
    data = np.ndarray(shape=(nrows, ncols), dtype=object)
    data[:, 0] = -hkl[:, 2].astype(int)
    data[:, 1] = -hkl[:, 1].astype(int)
    data[:, 2] = -hkl[:, 0].astype(int)
    for i in range(ampli.shape[1]):
        data[:, i + 3] = ampli[:, i].astype(np.float32)
    data[:, -2] = ((ampli[:, 0] + ampli[:, 1]) / 2.0).astype(np.float32)
    data[:, -1] = phase.astype(np.float32)
    # print(data.shape)
    mtz.set_data(data)
    mtz.write_to_file(outfile)


def write_3d2mtz_refmac(unit_cell, sgrid, fdata, fnoise, bfac, outfile="output.mtz"):
    """Prepares an MTZ file for REFMAC5 refinement.

    Arguments:
        Inputs:
            unit_cell: float, 1D array
                Unit cell params.
            sgrid: float, 3D array
                Resolution grid (unit - inverse Angstrom).
            fdata: complex, 3D array
                Fullmap values to write.
            fnoise: complex, 3D array
                Noise values to write.
            bfac: float, 1D array
                Array of B-factors to apply on Fourier Coefficients.

        Outputs:
            outfile: string
            Output file name. Default is output.mtz.
    """
    assert fdata.shape == fnoise.shape
    nx, ny, nz = fdata.shape
    mtz = gemmi.Mtz()
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.cell.set(unit_cell[0], unit_cell[1], unit_cell[2], 90, 90, 90)
    mtz.add_dataset("HKL_base")
    for label in ["H", "K", "L"]:
        mtz.add_column(label, "H")
    for label in ["Fout" + str(int(i)) for i in bfac]:
        mtz.add_column(label, "F")
    for label in ["SigF" + str(int(i)) for i in bfac]:
        mtz.add_column(label, "Q")
    mtz.add_column("Pout0", "P")
    h, k, l, ampli, noise, phase = fcodes_fast.prepare_hkl_bfac(
        sgrid, fdata, fnoise, bfac, 1, nx, ny, nz, len(bfac)
    )
    nrows = ampli.shape[0]
    ncols = 4 + 2 * ampli.shape[1]
    data = np.ndarray(shape=(nrows, ncols), dtype=object)
    data[:, 0] = -l.astype(int)
    data[:, 1] = -k.astype(int)
    data[:, 2] = -h.astype(int)
    for i in range(ampli.shape[1]):
        data[:, i + 3] = ampli[:, i].astype(np.float32)
        data[:, i + 3 + ampli.shape[1]] = noise[:, i].astype(np.float32)
    data[:, -1] = phase.astype(np.float32)
    print(data.shape)
    mtz.set_data(data)
    mtz.write_to_file(outfile)


""" def write_mtz2_3d(h, k, l, f, nx, ny, nz):
    # Output mtz data into numpy 3D array
    # NEED CAREFUL TEST
    arr = np.zeros((nx * ny * nx), dtype=np.complex)
    xv, yv, zv = np.meshgrid(h, k, l)
    xvf = xv.flatten(order="F")
    lb = (len(arr) - len(xvf)) // 2
    ub = lb + len(xvf)
    print(lb, ub)
    arr[lb:ub] = f
    f3d = arr.reshape(nx, ny, nz)
    mapdata = np.fft.ifftn(np.fft.fftshift(f3d))
    return mapdata """


def write_mtz2_3d_gemmi(mtzfile, map_size):
    # Writing out mtz data into ccp4.map format
    # Note that write_ccp4_map only supports numbers with factors 2, 3 and 5
    mtz = gemmi.read_mtz_file(mtzfile)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = mtz.transform_f_phi_to_map("Fout0", "Pout0", map_size)
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map("output.ccp4")
    return


#def resample2staticmap(curnt_pix, targt_pix, targt_dim, arr, sf=False, fobj=None):
#    """Resamples a 3D array.
#
#    Arguments:
#        Inputs:
#            curnt_pix: float, Current pixel size.
#            targt_pix: float, Target pixel size.
#            targt_dim: list, List of three integer values.
#            arr: float, 3D array of map values.
#            sf: bool, optional
#                If True, returns a complex array. Otherwise, float array
#            fobj: optional. Logger file object
#
#        Outputs:
#            new_arr: float, 3D array
#                Resampled 3D array. If sf was used, return is a complex array
#    """
#    # Resamling arr into an array having target_dim
#    tnx, tny, tnz = targt_dim
#    curnt_dim = arr.shape
#    nx, ny, nz = arr.shape
#    tnx, tny, tnz = targt_dim
#    print("pixel size [current, target]: ", curnt_pix, targt_pix)
#    if fobj is not None:
#        fobj.write(
#            "pixel size [current, target]: "
#            + str(curnt_pix)
#            + " "
#            + str(targt_pix)
#            + " \n"
#        )
#    if abs(curnt_pix - targt_pix) < 10e-3:
#        if targt_dim[0] == curnt_dim[0]:
#            print("No change of dims")
#            if fobj is not None:
#                fobj.write("No change of dims \n")
#            new_arr = arr
#        elif curnt_dim[0] < targt_dim[0]:
#            print("Padded with zeros")
#            if fobj is not None:
#                fobj.write("Padded with zeros \n")
#            dx = abs(tnx - nx) // 2
#            new_arr = np.zeros((tnx, tny, tnz), arr.dtype)
#            new_arr[dx: nx + dx, dx: nx + dx, dx: nx + dx] = arr
#        elif targt_dim[0] < curnt_dim[0]:
#            print("Cropped image")
#            if fobj is not None:
#                fobj.write("Cropped image \n")
#            dx = abs(nx - tnx) // 2
#            new_arr = np.zeros((targt_dim), arr.dtype)
#            new_arr = arr[dx: dx + tnx, dx: dx + tnx, dx: dx + tnx]
#    elif abs(curnt_pix - targt_pix) > 10e-3:
#        newsize = []
#        for i in range(3):
#            ns = int(round(arr.shape[i] * (curnt_pix / targt_pix)))
#            if ns % 2 != 0: ns -= 1
#            newsize.append(ns)
#        print("Resizing in Fourier space and transforming back")
#        if fobj is not None:
#            fobj.write("Resizing in Fourier space and transforming back \n")
#        new_arr = resample(arr, newsize, sf)
#        if newsize[0] < targt_dim[0]:
#            new_arr = padimage(new_arr, targt_dim)
#        elif newsize[0] > targt_dim[0]:
#            new_arr = cropimage(new_arr, targt_dim)
#    return new_arr


def resample2staticmap(curnt_pix, targt_pix, targt_dim, arr, sf=False, fobj=None):
    """Resamples a 3D array.

    Arguments:
        Inputs:
            curnt_pix: float list, Current pixel sizes along c, b, a.
            targt_pix: float list, Target pixel sizes along c, b a.
            targt_dim: int list, Target sampling along z, y, x.
            arr: float, 3D array of map values.
            sf: bool, optional
                If True, returns a complex array. Otherwise, float array
            fobj: optional. Logger file object

        Outputs:
            new_arr: float, 3D array
                Resampled 3D array. If sf was used, return is a complex array
    """
    # Resamling arr into an array having target_dim
    tnz, tny, tnx = targt_dim
    nz, ny, nx = arr.shape
    if len(curnt_pix) < 3:
        curnt_pix.append(curnt_pix[0])
        curnt_pix.append(curnt_pix[0])
    if len(targt_pix) < 3:
        targt_pix.append(targt_pix[0])
        targt_pix.append(targt_pix[0])
    print("Current pixel size: ", curnt_pix)
    print("Target pixel size: ", targt_pix)
    if fobj is not None:
        fobj.write(
            "pixel size [current, target]: "
            + str(curnt_pix)
            + " "
            + str(targt_pix)
            + " \n"
        )
    #if abs(curnt_pix - targt_pix) < 10e-3:
    if np.all(abs(np.array(curnt_pix) - np.array(targt_pix)) < 1e-3):
        dx = (tnx - nx) // 2
        dy = (tny - ny) // 2
        dz = (tnz - nz) // 2
        if dx == dy == dz == 0:
            print("No change of dims")
            if fobj is not None:
                fobj.write("No change of dims \n")
            new_arr = arr
        if np.any(np.array([dx, dy, dz]) > 0):
            print("Padded with zeros")
            if fobj is not None:
                fobj.write("Padded with zeros \n")
            new_arr = padimage(arr, targt_dim)
        if np.any(np.array([dx, dy, dz]) < 0):
            print("Cropped image")
            if fobj is not None:
                fobj.write("Cropped image \n")
            new_arr = cropimage(arr, targt_dim)
    else:
        newsize = []
        print("arr.shape: ", arr.shape)
        for i in range(3):
            ns = int(round(arr.shape[i] * (curnt_pix[i] / targt_pix[i])))
            newsize.append(ns)
        print("Resizing in Fourier space and transforming back")
        if fobj is not None:
            fobj.write("Resizing in Fourier space and transforming back \n")
        new_arr = resample(arr, newsize, sf)
        if np.any(np.array(new_arr.shape) < np.array(targt_dim)):
            print("pading image...")
            new_arr = padimage(new_arr, targt_dim)
        elif np.any(np.array(new_arr.shape) > np.array(targt_dim)):
            print("cropping image...")
            new_arr = cropimage(new_arr, targt_dim)
    return new_arr

def resample_on_anothermap(uc1, uc2, arr1, arr2):
    # arr1 is taken as reference and arr2 is resampled on arr1
    tpix1 = uc1[0] / arr1.shape[0]
    tpix2 = uc1[1] / arr1.shape[1]
    tpix3 = uc1[2] / arr1.shape[2]
    dim1 = int(round(uc2[0] / tpix1))
    dim2 = int(round(uc2[1] / tpix2))
    dim3 = int(round(uc2[2] / tpix3))
    target_dim = arr1.shape
    arr2 = resample(arr2, [dim1, dim2, dim3], sf=False)
    if np.any(np.array(arr2.shape) < np.array(target_dim)):
        arr2 = padimage(arr2, target_dim)
    elif np.any(np.array(arr2.shape) > np.array(target_dim)):
        arr2 = cropimage(arr2, target_dim)
    return arr2


def padimage(arr, tdim):
    import emda.emda_methods as em

    if len(tdim) == 3:
        tnz, tny, tnx = tdim
    elif len(tdim) < 3:
        tnz = tny = tnx = tdim[0]
    else:
        raise SystemExit("More than 3 dimensions given. Cannot handle")
    print("current shape: ", arr.shape)
    print("target shape: ", tdim)
    nz, ny, nx = arr.shape
    assert tnx >= nx
    assert tny >= ny
    assert tnz >= nz
    # previous padding
    """ dx = abs(tnx - nx) // 2; modz = abs(tnx - nx) % 2
    dy = abs(tny - ny) // 2; mody = abs(tny - ny) % 2
    dz = abs(tnz - nz) // 2; modx = abs(tnz - nz) % 2 
    image = np.zeros((tnz, tny, tnx), arr.dtype)
    image[dz: nz + dz, dy: ny + dy, dx: nx + dx] = arr """
    com1 = np.asarray(em.center_of_mass_density(arr))
    #print('com1', com1)
    com2 = (com1/np.array(arr.shape)) * np.asarray(tdim)
    #print('com2: ', com2)
    dx = int(com2[2] - com1[2]) + int(com2[2] - com1[2] > 0.9)
    dy = int(com2[1] - com1[1]) + int(com2[1] - com1[1] > 0.9)
    dz = int(com2[0] - com1[0]) + int(com2[0] - com1[0] > 0.9)
    print(dz, dy, dx)
    image = np.zeros((tnz, tny, tnx), arr.dtype)
    image[-(nz + dz):-dz, -(ny + dy):-dy, -(nx + dx):-dx] = arr
    return image

def cropimage(arr, tdim):
    if len(tdim) == 3:
        tnz, tny, tnx = tdim
    elif len(tdim) < 3:
        tnz = tny = tnx = tdim[0]
    else:
        raise SystemExit("More than 3 dimensions given. Cannot handle")
    nz, ny, nx = arr.shape
    assert tnx <= nx
    assert tny <= ny
    assert tnz <= nz
    dx = abs(nx - tnx) // 2
    dy = abs(ny - tny) // 2
    dz = abs(nz - tnz) // 2
    return arr[dz: tdim[0] + dz, dy: tdim[1] + dy, dx: tdim[2] + dx]


def resample(x, newshape, sf):
    xshape = list(x.shape)
    for i in range(3):
        if x.shape[i] % 2 != 0:
            xshape[i] += 1
        if newshape[i] % 2 != 0:
            newshape[i] += 1
    temp = np.zeros(xshape, x.dtype)
    temp[:x.shape[0], :x.shape[1], :x.shape[2]] = x
    x = temp
    print(np.array(x.shape) - np.array(newshape))
    # nosampling
    if np.all((np.array(x.shape) - np.array(newshape)) == 0):
        print('no sampling')
        return x
    # Forward transform
    X = np.fft.fftn(x)
    X = np.fft.fftshift(X)
    # Placeholder array for output spectrum
    Y = np.zeros(newshape, X.dtype)
    # upsampling
    dx = []
    if np.any((np.array(x.shape) - np.array(newshape)) < 0):
        print('upsampling...')
        for i in range(3):
            dx.append(abs(newshape[i] - X.shape[i]) // 2)
        Y[dx[0]: dx[0] + X.shape[0], 
          dx[1]: dx[1] + X.shape[1], 
          dx[2]: dx[2] + X.shape[2]] = X
    # downsampling
    if np.any((np.array(x.shape) - np.array(newshape)) > 0):
        print('downsampling...')
        for i in range(3):
            dx.append(abs(newshape[i] - X.shape[i]) // 2)
        Y[:, :, :] = X[
                    dx[0]: dx[0] + newshape[0], 
                    dx[1]: dx[1] + newshape[1], 
                    dx[2]: dx[2] + newshape[2]
                    ]
    if sf:
        return Y
    Y = np.fft.ifftshift(Y)
    return (np.fft.ifftn(Y)).real


#def resample(x, num, sf):
#    xshape = list(x.shape)
#    print(xshape, num)
#    for i in range(3):
#        if x.shape[i] % 2 != 0:
#            xshape[i] = xshape[i] + 1
#    temp = np.zeros(xshape, x.dtype)
#    #temp[:-1, :-1, :-1] = x
#    temp[:x.shape[0], :x.shape[1], :x.shape[2]] = x
#    x = temp
#    """ # Check if dims are even
#    if x.shape[0] % 2 != 0:
#        xshape = list(x.shape)
#        xshape[0] = xshape[0] + 1
#        xshape[1] = xshape[1] + 1
#        xshape[2] = xshape[2] + 1
#        temp = np.zeros(xshape, x.dtype)
#        temp[:-1, :-1, :-1] = x
#        x = temp """
#    newshape = list(x.shape)
#    newshape[0] = num[0]
#    newshape[1] = num[1]
#    newshape[2] = num[2]
#    if num[0] % 2 != 0:
#        newshape[0] = num[0] + 1
#        newshape[1] = num[1] + 1
#        newshape[2] = num[2] + 1
#    # no-sampling
#    if x.shape[0] == newshape[0]:
#        return x
#    print(x.shape)
#    print(newshape)
#    # Forward transform
#    X = np.fft.fftn(x)
#    X = np.fft.fftshift(X)
#    # Placeholder array for output spectrum
#    Y = np.zeros(newshape, X.dtype)
#    # upsampling
#    if X.shape[0] < newshape[0]:
#        dx = abs(newshape[0] - X.shape[0]) // 2
#        dy = abs(newshape[1] - X.shape[1]) // 2
#        dz = abs(newshape[2] - X.shape[2]) // 2
#        #Y[dx: dx + X.shape[0], dx: dx + X.shape[0], dx: dx + X.shape[0]] = X
#        Y[dx: dx + X.shape[0], dy: dy + X.shape[1], dz: dz + X.shape[2]] = X
#    # downsampling
#    if newshape[0] < X.shape[0]:
#        dx = abs(newshape[0] - X.shape[0]) // 2
#        Y[:, :, :] = X[
#                     dx: dx + newshape[0], dx: dx + newshape[0], dx: dx + newshape[0]
#                     ]
#    if sf:
#        return Y
#    Y = np.fft.ifftshift(Y)
#    return (np.fft.ifftn(Y)).real


def read_mmcif(mmcif_file):
    # Reading mmcif using gemmi and output as numpy 1D arrays
    # from gemmi import cif
    doc = gemmi.cif.read_file(mmcif_file)
    block = doc.sole_block()  # cif file as a single block
    a = block.find_value("_cell.length_a")
    b = block.find_value("_cell.length_b")
    c = block.find_value("_cell.length_c")
    alf = block.find_value("_cell.angle_alpha")
    bet = block.find_value("_cell.angle_beta")
    gam = block.find_value("_cell.angle_gamma")
    cell = np.array([a, b, c, alf, bet, gam], dtype="float")
    # Reading X coordinates in all atoms
    col_x = block.find_values("_atom_site.Cartn_x")
    col_y = block.find_values("_atom_site.Cartn_y")
    col_z = block.find_values("_atom_site.Cartn_z")
    # Reading B_iso values
    col_Biso = block.find_values("_atom_site.B_iso_or_equiv")
    # Casting gemmi.Columns into a numpy array
    x_np = np.array(col_x, dtype="float", copy=False)
    y_np = np.array(col_y, dtype="float", copy=False)
    z_np = np.array(col_z, dtype="float", copy=False)
    Biso_np = np.array(col_Biso, dtype="float", copy=False)
    return cell, x_np, y_np, z_np, Biso_np


def run_refmac_sfcalc(filename, resol, lig=True, bfac=None, ligfile=None):
    import os
    import os.path
    import subprocess

    # get current path
    current_path = os.getcwd()
    # get the path to filename
    filepath = os.path.abspath(os.path.dirname(filename)) + '/'
    # navigate to filepath
    os.chdir(filepath)
    fmtz = filename[:-4] + ".mtz"
    cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz]
    if ligfile is not None:
        cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz, "lib_in", ligfile]
        lig = False
    # Creating the sfcalc.inp with custom parameters (resol, Bfac)
    sfcalc_inp = open(filepath+"sfcalc.inp", "w+")
    sfcalc_inp.write("mode sfcalc\n")
    sfcalc_inp.write("sfcalc cr2f\n")
    if lig:
        sfcalc_inp.write("make newligand continue\n")
    sfcalc_inp.write("resolution %f\n" % resol)
    if bfac is not None and bfac > 0.0:
        sfcalc_inp.write("temp set %f\n" % bfac)
    sfcalc_inp.write("source em mb\n")
    sfcalc_inp.write("make hydrogen yes\n")
    sfcalc_inp.write("end")
    sfcalc_inp.close()
    # Read in sfcalc_inp
    PATH = filepath+"sfcalc.inp"
    logf = open(filepath+"sfcalc.log", "w+")
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print("sfcalc.inp exists and is readable")
        inp = open(filepath+"sfcalc.inp", "r")
        # Run the command with parameters from file f2mtz.inp
        subprocess.call(cmd, stdin=inp, stdout=logf)
        logf.close()
        inp.close()
    else:
        raise SystemExit("File is either missing or not readable")
    os.chdir(current_path)


def read_atomsf(atm, fpath=None):
    # outputs A and B coefficients corresponding to atom(atm)
    found = False
    with open(fpath) as myFile:
        for num, line in enumerate(myFile, 1):
            if line.startswith(atm):
                found = True
                break
    A = np.zeros(5, dtype=np.float)
    B = np.zeros(5, dtype=np.float)
    if found:
        ier = 0
        f = open(fpath)
        all_lines = f.readlines()
        for i in range(4):
            if i == 0:
                Z = all_lines[num + i].split()[0]
                NE = all_lines[num + i].split()[1]
                A[i] = all_lines[num + i].split()[-1]
                B[i] = 0.0
            elif i == 1:
                A[1:] = np.asarray(all_lines[num + i].split(), dtype=np.float)
            elif i == 2:
                B[1:] = np.asarray(all_lines[num + i].split(), dtype=np.float)
        f.close()
    else:
        ier = 1
        Z = 0
        NE = 0.0
        print(atm, "is not found!")
    return int(Z), float(NE), A, B, ier


def pdb2mmcif(filename_pdb):
    structure = gemmi.read_structure(filename_pdb)
    structure.setup_entities()
    structure.assign_label_seq_id()
    mmcif = structure.make_mmcif_document()
    mmcif.write_file("out.cif")


def mask_by_value(array, masking_value=0.0, filling_value=0.0):
    # Array values less_or_equal to masking_value are filled with
    # given filling value
    import numpy.ma as ma

    array_ma = ma.masked_less_equal(array, masking_value)
    array_masked_filled = array_ma.filled(filling_value)
    return array_masked_filled


def mask_by_value_greater(array, masking_value=0.0, filling_value=0.0):
    # Array values greater than masking_value are filled with
    # given filling value
    import numpy.ma as ma

    array_ma = ma.masked_greater(array, masking_value)
    array_masked_filled = array_ma.filled(filling_value)
    return array_masked_filled


def output_to_table(dataframe, filename="data_emda.txt"):
    with open(filename, 'a') as f:
        f.write(
            dataframe.to_string(header=True, index=False)
        )
    print("data_emda.txt file was written!")


def apply_transformation_on_model(mmcif_file, rotmat=None, trans=None, outfilename=None):
    import gemmi
    import numpy as np

    doc = gemmi.cif.read_file(mmcif_file)
    st = gemmi.read_structure(mmcif_file)
    model = st[0]
    com = model.calculate_center_of_mass()
    # print(com)
    block = doc.sole_block()  # cif file as a single block
    col_x = block.find_values("_atom_site.Cartn_x")
    col_y = block.find_values("_atom_site.Cartn_y")
    col_z = block.find_values("_atom_site.Cartn_z")
    a = block.find_value("_cell.length_a")
    b = block.find_value("_cell.length_b")
    c = block.find_value("_cell.length_c")
    alf = block.find_value("_cell.angle_alpha")
    bet = block.find_value("_cell.angle_beta")
    gam = block.find_value("_cell.angle_gamma")
    cell = np.array([a, b, c, alf, bet, gam], dtype="float")
    if rotmat is None:
        rotmat = np.identity(3)
    if trans is None:
        trans = np.zeros(3, dtype='float')
    if outfilename is None:
        outfilename = "emda_transformed_model.cif"
    vec = np.zeros(3, dtype='float')
    for n, _ in enumerate(col_x):
        vec[0] = float(col_x[n]) - com.x
        vec[1] = float(col_y[n]) - com.y
        vec[2] = float(col_z[n]) - com.z
        vec_rot = np.dot(rotmat, vec)
        col_x[n] = str(vec_rot[0] + com.x + trans[2])
        col_y[n] = str(vec_rot[1] + com.y + trans[1])
        col_z[n] = str(vec_rot[2] + com.z + trans[0])
    x_np = np.array(col_x, dtype="float", copy=False)
    y_np = np.array(col_y, dtype="float", copy=False)
    z_np = np.array(col_z, dtype="float", copy=False)
    doc.write_file(outfilename)
    return cell, x_np, y_np, z_np


def model_transform_gm(mmcif_file, rotmat=None, trans=None, outfilename=None, mapcom=None):
    if rotmat is None:
        rotmat = np.identity(3)
    if outfilename is None:
        outfilename = "gemmi_transformed_model.cif"
    st = gemmi.read_structure(mmcif_file)
    com = gemmi.Vec3(*st[0].calculate_center_of_mass())
    print("Model COM: ", com)
    if mapcom is not None:
        com = gemmi.Vec3(mapcom[2], mapcom[1], mapcom[0])
        """ comoffset = np.zeros(3, 'float')
        comoffset[0] = float(com.z) - mapcom[0]
        comoffset[1] = float(com.y) - mapcom[1]
        comoffset[2] = float(com.x) - mapcom[2]
        trans += comoffset """ 
    t = gemmi.Vec3(trans[2], trans[1], trans[0]) # ZYX --> XYZ
    mat33 = gemmi.Mat33(rotmat)
    trans = com - mat33.multiply(com) + t
    tr = gemmi.Transform(mat33, trans)
    st[0].transform(tr)
    st.make_mmcif_document().write_file(outfilename)


##### below function are not frequently used.#####
def read_mrc(mapname11):
    file11 = mrc.open(mapname11)
    cell1 = np.array(file11.header.cella)
    a1 = cell1["x"]
    b1 = cell1["y"]
    c1 = cell1["z"]
    uc = np.asarray([a1 * 1.0, b1 * 1.0, c1 * 1.0, 90.0, 90.0, 90.0])
    origin = [
        1 * file11.header.nxstart,
        1 * file11.header.nystart,
        1 * file11.header.nzstart,
    ]
    ar11 = np.asarray(file11.data, dtype="float")
    ar11_centered = np.fft.fftshift(ar11)  # CENTERED IMAGE
    nx, ny, nz = ar11_centered.shape
    print(mapname11, nx, ny, nz, uc[:3])
    file11.close()
    hf11 = np.fft.fftshift(np.fft.fftn(ar11_centered))  # CENTERED FFT OF CENTERED IMAGE
    return uc, hf11, origin

# def write_3d2mtz_full(uc, arr, outfile="output.mtz"):
#    # Write numpy 3D array into MTZ file.
#    # Issue: It write out full sphere data
#    import numpy.fft as fft
#
#    print(arr.shape)
#    nx, ny, nz = arr.shape
#    mtz = gemmi.Mtz()
#    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
#    mtz.cell.set(uc[0], uc[1], uc[2], 90, 90, 90)
#    mtz.add_dataset("HKL_base")
#    for label in ["H", "K", "L"]:
#        mtz.add_column(label, "H")
#    mtz.add_column("Fout0", "F")
#    mtz.add_column("Pout0", "P")
#    # Add more columns if you need
#    x = fft.fftshift(fft.fftfreq(arr.shape[0]) * nx)
#    y = fft.fftshift(fft.fftfreq(arr.shape[1]) * ny)
#    z = fft.fftshift(fft.fftfreq(arr.shape[2]) * nz)
#    xv, yv, zv = np.meshgrid(x, y, z)
#    xvf = xv.flatten(order="F")
#    yvf = yv.flatten(order="F")
#    zvf = zv.flatten(order="F")
#
#    arr_real = np.real(arr)
#    arr_imag = np.imag(arr)
#    ampli = np.sqrt(np.power(arr_real, 2) + np.power(arr_imag, 2))
#    phase = np.arctan2(arr_imag, arr_real) * 180 / np.pi
#
#    ampli_1d = ampli.flatten(order="F")
#    phase_1d = phase.flatten(order="F")
#
#    nrows = len(xvf)
#    ncols = 5  # Change this according to what columns to write
#    data = np.ndarray(shape=(nrows, ncols), dtype=object)
#    data[:, 0] = zvf.astype(int)
#    data[:, 1] = xvf.astype(int)
#    data[:, 2] = yvf.astype(int)
#    data[:, 3] = ampli_1d.astype(np.float32)
#    data[:, 4] = phase_1d.astype(np.float32)
#    print(data.shape)
#    mtz.set_data(data)
#    mtz.write_to_file(outfile)
