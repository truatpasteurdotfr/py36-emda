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
import gemmi, pandas
import mrcfile as mrc
from emda.config import debug_mode


def test():
    """ Tests iotools module installation. """
    
    print("iotools test ... Passed")


def read_map(mapname):
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

    try:
        file = mrc.open(mapname)
        cell = np.array(file.header.cella)
        unit_cell = np.zeros(6, dtype='float')
        cell = file.header.cella[['x', 'y', 'z']]
        unit_cell[:3] = cell.view(('f4', 3))
        # Swap cell parameters a, b, c to c, b, a
        tmp = unit_cell[:3]
        unit_cell[0],  unit_cell[2] = tmp[2], tmp[0]
        #
        unit_cell[3:] = float(90)
        origin = [
            1 * file.header.nxstart,
            1 * file.header.nystart,
            1 * file.header.nzstart,
        ]
        arr = np.asarray(file.data, dtype="float")
        file.close()
        print(mapname, arr.shape, unit_cell[:3])
        return unit_cell, arr, origin
    except FileNotFoundError:
        print("File Not Found!")
        exit()


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

       
def write_mrc(mapdata, filename, unit_cell, map_origin=None, factor=1.0, label=False):
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
            factor: float, optional
                A factor to apply on unit cell params. Default is 1.0
            label: bool, optional
                If True a text label is written out in the map header.
                Default is False

        Outputs:
            Outputs the MRC file
    """
    if map_origin is None:
        map_origin = [0.0, 0.0, 0.0]
    file = mrc.new(
        name=filename, data=np.float32(mapdata), compression=None, overwrite=True
    )
    file.header.cella.x = unit_cell[0] * factor
    file.header.cella.y = unit_cell[1] * factor
    file.header.cella.z = unit_cell[2] * factor
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
    #file.update_header_from_data()
    file.close()


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
    h,k,l,ampli,phase = fcodes_fast.prepare_hkl(mapdata,bin_idx,cbin,debug_mode,nx,ny,nz)
    #h, k, l, ampli, phase = fcodes_fast.prepare_hkl(mapdata, debug_mode, nx, ny, nz)
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
    #print("data shape: ", data.shape)
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


def resample2staticmap(curnt_pix, targt_pix, targt_dim, arr, sf=False, fobj=None):
    """Resamples a 3D array.

    Arguments:
        Inputs:
            curnt_pix: float, Current pixel size.
            targt_pix: float, Target pixel size.
            targt_dim: list, List of three integer values.
            arr: float, 3D array of map values.
            sf: bool, optional
                If True, returns a complex array. Otherwise, float array
            fobj: optional. Logger file object

        Outputs:
            new_arr: float, 3D array
                Resampled 3D array. If sf was used, return is a complex array
    """
    # Resamling arr into an array having target_dim
    tnx, tny, tnz = targt_dim
    curnt_dim = arr.shape
    nx, ny, nz = arr.shape
    tnx, tny, tnz = targt_dim
    assert tnx == tny == tnz
    assert nx == ny == nz
    print("pixel size [current, target]: ", curnt_pix, targt_pix)
    if fobj is not None:
        fobj.write(
            "pixel size [current, target]: "
            + str(curnt_pix)
            + " "
            + str(targt_pix)
            + " \n"
        )
    # new_arr = resample(arr, targt_dim, sf)
    if abs(curnt_pix - targt_pix) < 10e-3:
        if targt_dim[0] == curnt_dim[0]:
            print("No change of dims")
            if fobj is not None:
                fobj.write("No change of dims \n")
            new_arr = arr
        elif curnt_dim[0] < targt_dim[0]:
            print("Padded with zeros")
            if fobj is not None:
                fobj.write("Padded with zeros \n")
            dx = abs(tnx - nx) // 2
            new_arr = np.zeros((tnx, tny, tnz), arr.dtype)
            new_arr[dx : nx + dx, dx : nx + dx, dx : nx + dx] = arr
        elif targt_dim[0] < curnt_dim[0]:
            print("Cropped image")
            if fobj is not None:
                fobj.write("Cropped image \n")
            dx = abs(nx - tnx) // 2
            new_arr = np.zeros((targt_dim), arr.dtype)
            new_arr = arr[dx : dx + tnx, dx : dx + tnx, dx : dx + tnx]
    elif curnt_pix != targt_pix:
        print("Resizing in Fourier space and transforming back")
        if fobj is not None:
            fobj.write("Resizing in Fourier space and transforming back \n")
        new_arr = resample(arr, targt_dim, sf)
    return new_arr


def resample(x, num, sf):
    # Check if dims are even
    if x.shape[0] % 2 != 0:
        xshape = list(x.shape)
        xshape[0] = xshape[0] + 1
        xshape[1] = xshape[1] + 1
        xshape[2] = xshape[2] + 1
        temp = np.zeros(xshape, x.dtype)
        temp[:-1, :-1, :-1] = x
        x = temp
    newshape = list(x.shape)
    newshape[0] = num[0]
    newshape[1] = num[1]
    newshape[2] = num[2]
    if num[0] % 2 != 0:
        newshape[0] = num[0] + 1
        newshape[1] = num[1] + 1
        newshape[2] = num[2] + 1
    # no-sampling
    if x.shape[0] == newshape[0]:
        return x
    # Forward transform
    X = np.fft.fftn(x)
    X = np.fft.fftshift(X)
    # Placeholder array for output spectrum
    Y = np.zeros(newshape, X.dtype)
    # upsampling
    if X.shape[0] < newshape[0]:
        dx = abs(newshape[0] - X.shape[0]) // 2
        Y[dx : dx + X.shape[0], dx : dx + X.shape[0], dx : dx + X.shape[0]] = X
    # downsampling
    if newshape[0] < X.shape[0]:
        dx = abs(newshape[0] - X.shape[0]) // 2
        Y[:, :, :] = X[
            dx : dx + newshape[0], dx : dx + newshape[0], dx : dx + newshape[0]
        ]
    if sf:
        return Y
    Y = np.fft.ifftshift(Y)
    return (np.fft.ifftn(Y)).real


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
    # xyz_np = np.zeros((3,len(col_x)), dtype='float')
    # xyz_np = np.zeros((3,len(col_x)), dtype='float')
    # xyz_np = np.zeros((3,len(col_x)), dtype='float')
    # xyz_np[0,:] = np.array(col_x, dtype='float', copy=False)
    # xyz_np[1,:] = np.array(col_y, dtype='float', copy=False)
    # xyz_np[2,:] = np.array(col_z, dtype='float', copy=False)
    x_np = np.array(col_x, dtype="float", copy=False)
    y_np = np.array(col_y, dtype="float", copy=False)
    z_np = np.array(col_z, dtype="float", copy=False)
    Biso_np = np.array(col_Biso, dtype="float", copy=False)
    return cell, x_np, y_np, z_np, Biso_np


def run_refmac_sfcalc(filename, resol, bfac, lig=True, ligfile=None):
    import os
    import os.path
    import subprocess

    fmtz = filename[:-4] + ".mtz"
    cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz]
    if ligfile is not None:
        cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz, "lib_in", ligfile]
        lig = False
    # Creating the sfcalc.inp with custom parameters (resol, Bfac)
    sfcalc_inp = open("sfcalc.inp", "w+")
    sfcalc_inp.write("mode sfcalc\n")
    sfcalc_inp.write("sfcalc cr2f\n")
    if lig:
        sfcalc_inp.write("make newligand continue\n")
    sfcalc_inp.write("resolution %f\n" % resol)
    sfcalc_inp.write("temp set %f\n" % bfac)
    sfcalc_inp.write("source em mb\n")
    sfcalc_inp.write("make hydrogen yes\n")
    sfcalc_inp.write("end")
    sfcalc_inp.close()
    # Read in sfcalc_inp
    PATH = "sfcalc.inp"
    logf = open("sfcalc.log", "w+")
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print("sfcalc.inp exists and is readable")
        inp = open("sfcalc.inp", "r")
        # Run the command with parameters from file f2mtz.inp
        subprocess.call(cmd, stdin=inp, stdout=logf)
    else:
        print("Either the file is missing or not readable")
    logf.close()
    inp.close()


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
            dataframe.to_string(header = True, index = False)
        )
    print("data_emda.txt file was written!")



def apply_transformation_on_model(mmcif_file, rotmat=None, trans=None, outfilename=None):
    import gemmi
    import numpy as np
    
    doc = gemmi.cif.read_file(mmcif_file)
    st = gemmi.read_structure(mmcif_file)
    model = st[0]
    com = model.calculate_center_of_mass()
    #print(com)
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
        vec_rot = rotmat @ vec
        col_x[n] = str(vec_rot[0] + com.x + trans[0])
        col_y[n] = str(vec_rot[1] + com.y + trans[1])
        col_z[n] = str(vec_rot[2] + com.z + trans[2])
    x_np = np.array(col_x, dtype="float", copy=False)
    y_np = np.array(col_y, dtype="float", copy=False)
    z_np = np.array(col_z, dtype="float", copy=False)
    doc.write_file(outfilename)
    return cell, x_np, y_np, z_np


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


#def write_3d2mtz_full(uc, arr, outfile="output.mtz"):
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
