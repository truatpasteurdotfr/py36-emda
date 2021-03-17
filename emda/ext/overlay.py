"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from timeit import default_timer as timer
import numpy as np
import fcodes_fast
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from emda import core
from emda.core import quaternions
from emda.ext.mapfit.utils import get_FRS, create_xyz_grid, get_xyz_sum, set_dim_even
from emda.ext.mapfit import utils, interp_derivatives
import emda.emda_methods as em


np.set_printoptions(suppress=True)  # Suppress insignificant values for clarity

timeit = True


class EmmapOverlay:
    def __init__(self, map_list, modelres=5.0, com=False, mask_list=None):
        self.map_list = map_list
        self.mask_list = mask_list
        self.modelres = modelres
        self.map_unit_cell = None
        self.map_origin = None
        self.map_dim = None
        self.pixsize = None
        self.arr_lst = []
        self.msk_lst = []
        self.carr_lst = []
        self.ceo_lst = None
        self.cfo_lst = None
        self.cbin_idx = None
        self.cdim = None
        self.cbin = None
        self.com = com
        self.com1 = None
        self.comlist = []
        self.box_centr = None
        self.fhf_lst = None
        self.nbin = None
        self.res_arr = None
        self.bin_idx = None
        self.fo_lst = None
        self.eo_lst = None
        self.totalvar_lst = None

    def load_maps(self):
        from scipy import ndimage
        from scipy.ndimage.interpolation import shift

        #com = self.com
        cmask = False
        fhf_lst = []
        full_fhf_lst = []
        if self.mask_list is not None:
            if len(self.map_list) != len(self.mask_list):
                raise SystemExit("map_list and mask_list must have the same size!")
            for i in range(len(self.mask_list)):
                if i == 0:
                    _, mask, _ = em.get_data(self.mask_list[i])
                    mask = utils.set_dim_even(mask)
                    uc, arr, origin = em.get_data(self.map_list[i])
                    arr = utils.set_dim_even(arr)
                    try:
                        assert arr.shape == mask.shape
                    except AssertionError:
                        raise SystemExit("Map and Mask Dimension mismatched!")
                    arr = arr * mask
                    nx, ny, nz = arr.shape
                    map_origin = origin
                    uc_target = uc
                    target_dim = arr.shape
                    target_pix_size = uc_target[0] / target_dim[0]
                    if cmask:
                        corner_mask = utils.remove_unwanted_corners(uc, target_dim)
                    else:
                        corner_mask = 1.0
                    if self.com:
                        com1 = ndimage.measurements.center_of_mass(arr * (arr >= 0.0))
                        print("COM: ", com1)
                        box_centr = (nx // 2, ny // 2, nz // 2)
                        self.com1, self.box_centr = com1, box_centr
                        self.comlist.append(com1)
                        arr_mvd = shift(arr, np.subtract(box_centr, com1))
                        mask_mvd = shift(mask, np.subtract(box_centr, com1))
                        self.arr_lst.append(arr_mvd * corner_mask)
                        self.msk_lst.append(mask_mvd)
                        fhf_lst.append(fftshift(fftn(fftshift(arr_mvd * corner_mask))))
                    else:
                        self.arr_lst.append(arr * corner_mask)
                        self.msk_lst.append(mask)
                        fhf_lst.append(fftshift(fftn(fftshift(arr * corner_mask))))
                else:
                    uc, arr, origin = em.get_data(
                        self.map_list[i],
                        resol=self.modelres,
                        dim=target_dim,
                        uc=uc_target,
                        maporigin=map_origin,
                    )
                    arr = utils.set_dim_even(arr)
                    print("origin: ", origin)
                    _, mask, _ = em.get_data(self.mask_list[i])
                    mask = utils.set_dim_even(mask)
                    try:
                        assert arr.shape == mask.shape
                    except AssertionError:
                        raise SystemExit("Map and Mask Dimension mismatched!")
                    arr = arr * mask
                    curnt_pix_size = uc[0] / arr.shape[0]
                    arr = core.iotools.resample2staticmap(
                        curnt_pix=curnt_pix_size,
                        targt_pix=target_pix_size,
                        targt_dim=target_dim,
                        arr=arr,
                    )
                    mask = core.iotools.resample2staticmap(
                        curnt_pix=curnt_pix_size,
                        targt_pix=target_pix_size,
                        targt_dim=target_dim,
                        arr=mask,
                    )
                    if self.com:
                        com1 = ndimage.measurements.center_of_mass(arr * (arr >= 0.0))
                        print("COM: ", com1)
                        self.comlist.append(com1)
                        arr = shift(arr, np.subtract(box_centr, com1))
                        mask = shift(mask, np.subtract(box_centr, com1))
                    self.arr_lst.append(arr * corner_mask)
                    self.msk_lst.append(mask)
                    fhf_lst.append(fftshift(fftn(fftshift(arr * corner_mask))))
            self.pixsize = target_pix_size
            self.map_origin = map_origin
            self.map_unit_cell = uc_target
            self.map_dim = target_dim
            self.fhf_lst = fhf_lst
        if self.mask_list is None:
            for i in range(len(self.map_list)):
                if i == 0:
                    uc, arr, origin = em.get_data(self.map_list[i])
                    arr = utils.set_dim_even(arr)
                    print("origin: ", origin)
                    nx, ny, nz = arr.shape
                    map_origin = origin
                    uc_target = uc
                    target_dim = arr.shape
                    target_pix_size = uc_target[0] / target_dim[0]
                    if self.com:
                        com1 = ndimage.measurements.center_of_mass(arr * (arr >= 0.0))
                        print("COM before centering: ", com1)
                        self.comlist.append(com1)
                        box_centr = (nx // 2, ny // 2, nz // 2)
                        print("BOX center: ", box_centr)
                        self.com1 = com1
                        self.box_centr = box_centr
                        arr = shift(arr, np.subtract(box_centr, com1))
                        self.arr_lst.append(arr)
                        core.iotools.write_mrc(
                            arr, "static_centered.mrc", uc_target, map_origin
                        )
                        print(
                            "COM after centering: ",
                            ndimage.measurements.center_of_mass(arr * (arr >= 0.0)),
                        )
                    self.arr_lst.append(arr)
                    fhf_lst.append(fftshift(fftn(fftshift(arr))))
                else:
                    uc, arr, origin = em.get_data(
                        self.map_list[i],
                        resol=self.modelres,
                        dim=target_dim,
                        uc=uc_target,
                        maporigin=map_origin,
                    )
                    arr = utils.set_dim_even(arr)
                    em.write_mrc(arr, "modelmap" + str(i) + ".mrc", uc, origin)
                    print("origin: ", origin)
                    curnt_pix_size = uc[0] / arr.shape[0]
                    arr = core.iotools.resample2staticmap(
                        curnt_pix=curnt_pix_size,
                        targt_pix=target_pix_size,
                        targt_dim=target_dim,
                        arr=arr,
                    )
                    if self.com:
                        com1 = ndimage.measurements.center_of_mass(arr * (arr >= 0.0))
                        print("COM: ", com1)
                        self.comlist.append(com1)
                        arr = shift(arr, np.subtract(box_centr, com1))
                        core.iotools.write_mrc(
                            arr, "moving_centered.mrc", uc_target, map_origin
                        )
                        print(
                            "COM after centering: ",
                            ndimage.measurements.center_of_mass(arr * (arr >= 0.0)),
                        )
                    self.arr_lst.append(arr)
                    fhf_lst.append(fftshift(fftn(fftshift(arr))))
            self.pixsize = target_pix_size
            self.map_origin = map_origin
            self.map_unit_cell = uc_target
            self.map_dim = target_dim
            self.fhf_lst = fhf_lst

    def calc_fsc_from_maps(self):
        # function for only two maps fitting
        nmaps = len(self.fhf_lst)
        print('nmaps: ', nmaps)
        fFo_lst = []
        fEo_lst = []
        fBTV_lst = []

        self.nbin, self.res_arr, self.bin_idx = core.restools.get_resolution_array(
            self.map_unit_cell, self.fhf_lst[0]
        )
        #
        for i in range(nmaps):
            _, _, _, totalvar, fo, eo = core.fsc.halfmaps_fsc_variance(
                self.fhf_lst[i], self.fhf_lst[i], self.bin_idx, self.nbin
            )
            fFo_lst.append(fo)
            fEo_lst.append(eo)
            fBTV_lst.append(totalvar)
        #
        self.fo_lst = fFo_lst
        self.eo_lst = fEo_lst
        self.totalvar_lst = fBTV_lst


def cut_resolution_for_linefit(f_list, bin_idx, res_arr, smax):
    # Making data for map fitting
    f_arr = np.asarray(f_list, dtype='complex')
    nx, ny, nz = f_list[0].shape
    cbin = cx = smax
    #print("Cut data at ", res_arr[cbin], " A")
    dx = int((nx - 2 * cx) / 2)
    dy = int((ny - 2 * cx) / 2)
    dz = int((nz - 2 * cx) / 2)
    cBIdx = bin_idx[dx : dx + 2 * cx, dy : dy + 2 * cx, dz : dz + 2 * cx]
    fout = fcodes_fast.cutmap_arr(
        f_arr, bin_idx, cbin, 0, len(res_arr), nx, ny, nz, len(f_list)
    )[:, dx : dx + 2 * cx, dy : dy + 2 * cx, dz : dz + 2 * cx]
    return fout, cBIdx, cbin


class linefit:
    def __init__(self):
        self.e0 = None
        self.e1 = None
        self.bin_idx = None
        self.nbin = None
        self.step = None
        self.q_prev = None
        self.t = None
        self.q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_linefit_static_data(self, e_list, bin_idx, res_arr, smax):
        if len(e_list) == 2:
            eout, self.bin_idx, self.nbin = cut_resolution_for_linefit(e_list, bin_idx, res_arr, smax)
        else:
            print("len(e_list: ", len(e_list))
            raise SystemExit()
        self.e0 = eout[0,:,:,:]
        self.e1 = eout[1,:,:,:]

    def get_fsc_wght(self, e0, ert, bin_idx, nbin):
        cx, cy, cz = e0.shape
        bin_stats = core.fsc.anytwomaps_fsc_covariance(e0, ert, bin_idx, nbin)
        fsc, _ = bin_stats[0], bin_stats[1]
        #fsc = np.array(fsc, dtype=np.float64, copy=False)
        #fsc = fsc / (1 - fsc**2)
        w_grid = fcodes_fast.read_into_grid(bin_idx, fsc, nbin, cx, cy, cz)
        #w_grid = np.array(w_grid, dtype=np.float64, copy=False)
        return w_grid

    def func(self, i):
        tmp = np.insert(self.step * i, 0, 0.0)
        q = tmp + self.q_prev
        q = q / np.sqrt(np.dot(q, q))
        rotmat = quaternions.get_RM(q)
        ers = get_FRS(rotmat, self.e1, interp="linear")
        w_grid = self.get_fsc_wght(self.e0, ers[:, :, :, 0], self.bin_idx, self.nbin)
        fval = np.real(np.sum(w_grid * self.e0 * np.conjugate(ers[:, :, :, 0])))
        #without weights does seem to work for EMD-6952
        #fval = np.real(np.sum(self.e0 * np.conjugate(ers[:, :, :, 0])))
        return -fval

    def scalar_opt(self, t=None):
        from scipy.optimize import minimize_scalar

        nx, ny, nz = self.e1.shape
        # self.alpha_t = self.scalar_opt_trans()
        if t is not None:
            st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
            self.e1 = self.e1 * st
        f = self.func
        res = minimize_scalar(f, method="brent")
        return res.x

    def func_t(self, i):
        nx, ny, nz = self.e0.shape
        #t = self.t + self.step * i
        t = self.step * i
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        e1_t = self.e1 * st
        w_grid = self.get_fsc_wght(self.e0, e1_t, self.bin_idx, self.nbin)
        fval = np.sum(w_grid * self.e0 * np.conjugate(e1_t))
        return -fval.real

    def scalar_opt_trans(self):
        from scipy.optimize import minimize_scalar

        start = timer()
        f = self.func_t
        res = minimize_scalar(f, method="brent")
        end = timer()
        #print("time for trans linefit: ", end-start)
        return res.x


def get_dfs(mapin, xyz, vol):
    nx, ny, nz = mapin.shape
    dfs = np.zeros(shape=(nx, ny, nz, 3), dtype=np.complex64)
    for i in range(3):
        dfs[:, :, :, i] = np.fft.fftshift(
            (1 / vol) * 2j * np.pi * np.fft.fftn(mapin * xyz[i])
        )
    return dfs


def derivatives_rotation(e0, e1, cfo, wgrid, sv, q, xyz, xyz_sum):
    from emda.core.quaternions import derivatives_wrt_q

    # rotation derivatives
    tp2 = (2.0 * np.pi) ** 2
    nx, ny, nz = e0.shape
    vol = nx * ny * nz
    #dFRS = get_dfs(np.real(np.fft.ifftn(np.fft.ifftshift(e1))), xyz, vol)
    dFRS = get_dfs(np.real(np.fft.ifftn(np.fft.ifftshift(cfo))), xyz, vol)
    dRdq = derivatives_wrt_q(q)
    df = np.zeros(3, dtype="float")
    ddf = np.zeros((3, 3), dtype="float")
    for i in range(3):
        a = np.zeros((3, 3), dtype="float")
        for k in range(3):
            for l in range(3):
                if k == 0 or (k > 0 and l >= k):
                    a[k, l] = np.sum(
                        wgrid
                        * np.real(
                            np.conjugate(e0)
                            * (dFRS[:, :, :, k] * sv[l, :, :, :] * dRdq[i, k, l])
                        )
                    )
                else:
                    a[k, l] = a[l, k]
        df[i] = np.sum(a)
    wfsc = wgrid * np.real(np.conjugate(e0) * e1)
    for i in range(3):
        for j in range(3):
            if i == 0 or (i > 0 and j >= i):
                b = np.zeros((3, 3), dtype="float")
                n = -1
                for k in range(3):
                    for l in range(3):
                        if k == 0 or (k > 0 and l >= k):
                            n += 1
                            b[k, l] = (
                                (-tp2 / vol)
                                * xyz_sum[n]
                                * np.sum(
                                    wfsc
                                    * sv[k, :, :, :]
                                    * sv[l, :, :, :]
                                    * dRdq[i, k, l]
                                    * dRdq[j, k, l]
                                )
                            )
                        else:
                            b[k, l] = b[l, k]
                ddf[i, j] = np.sum(b)
            else:
                ddf[i, j] = ddf[j, i]
    ddf_inv = np.linalg.pinv(ddf)
    step = ddf_inv.dot(-df)
    return step


def derivatives_translation(e0, e1, wgrid, w2grid, sv):
    PI = np.pi
    tp2 = (2.0 * PI)**2
    tpi = (2.0 * PI * 1j)
    start = timer()
    # translation derivatives
    df = np.zeros(3, dtype='float')
    ddf = np.zeros((3,3), dtype='float')
    for i in range(3):
        df[i] = np.real(np.sum(wgrid * e0 * np.conjugate(e1 * tpi * sv[i,:,:,:])))
        for j in range(3):
            if(i==0 or (i>0 and j>=i)):
                ddf[i,j] = -tp2 * np.sum(w2grid * sv[i,:,:,:] * sv[j,:,:,:])
            else:
                ddf[i,j] = ddf[j,i]
    ddf_inv = np.linalg.pinv(ddf)
    step = ddf_inv.dot(-df)
    end = timer()
    #print("time for trans deriv. ", end-start)
    return step


def rebox_density(density, newdims):
    nx, ny, nz = density.shape
    cx, cy, cz = list(newdims)
    dx = int((density.shape[0] - cx) / 2)
    dy = int((density.shape[1] - cx) / 2)
    dz = int((density.shape[2] - cx) / 2)
    boxed_density = density[dx : dx + cx, dy : dy + cx, dz : dz + cx]
    return boxed_density


class EmFit:
    def __init__(self, mapobj, interp="linear", dfs=None):
        self.mapobj = mapobj
        self.cut_dim = mapobj.cdim
        self.ful_dim = mapobj.map_dim
        self.cell = mapobj.map_unit_cell
        self.pixsize = mapobj.pixsize
        self.origin = mapobj.map_origin
        self.interp = interp
        self.dfs = dfs
        self.w_grid = None
        self.fsc = None
        self.sv = None
        self.t = None
        self.st = None
        self.step = None
        self.q = None
        self.q_accum = None
        self.q_final_list = []
        self.rotmat = None
        self.t_accum = None
        self.ert = None
        self.frt = None
        self.cfo = None
        self.crt = None
        self.e0 = None
        self.e1 = None
        self.w2_grid = None
        self.fsc_lst = []
        self.le0 = None
        self.le1 = None
        self.lbinindx = None
        self.lnbin = None

    def getst(self):
        cx, cy, cz = self.e0.shape
        st, _, _, _ = fcodes_fast.get_st(cx, cy, cz, self.t)
        return st

    def calc_fsc_t(self):
        cx, cy, cz = self.e0.shape
        self.st, s1, s2, s3 = fcodes_fast.get_st(cx, cy, cz, self.t)
        self.sv = np.array([s1, s2, s3])
        bin_idx = self.mapobj.cbin_idx
        nbin = self.mapobj.cbin
        #self.ert = subroutine2.trilinear2(self.e1 * self.st,bin_idx,self.rotmat,nbin,0,1,cx, cy, cz)[:, :, :, 0]
        self.ert = fcodes_fast.trilinear2(self.e1 * self.st,bin_idx,self.rotmat,nbin,0,1,cx, cy, cz)[:, :, :, 0]
        fsc = core.fsc.anytwomaps_fsc_covariance(
            self.e0, self.ert, self.mapobj.cbin_idx, self.mapobj.cbin
        )[0]
        return fsc

    def calc_fsc_r(self):
        cx, cy, cz = self.e0.shape
        self.st, s1, s2, s3 = fcodes_fast.get_st(cx, cy, cz, self.t)
        self.sv = np.array([s1, s2, s3])
        maps2send = np.stack((self.e1 * self.st, self.cfo * self.st), axis = -1)
        nc = 2
        bin_idx = self.mapobj.cbin_idx
        nbin = self.mapobj.cbin
        #maps = subroutine2.trilinear2(maps2send,bin_idx,self.rotmat,nbin,0,nc,cx, cy, cz)
        maps = fcodes_fast.trilinear2(maps2send,bin_idx,self.rotmat,nbin,0,nc,cx, cy, cz)
        self.ert = maps[:, :, :, 0]
        self.crt = maps[:, :, :, 1]
        fsc = core.fsc.anytwomaps_fsc_covariance(
            self.e0, self.ert, self.mapobj.cbin_idx, self.mapobj.cbin
        )[0]
        return fsc

    def get_wght(self):
        #import subroutine2
        cx, cy, cz = self.e0.shape
        val_arr = np.zeros((self.mapobj.cbin, 2), dtype='float')
        val_arr[:,0] = self.fsc / (1 - self.fsc ** 2)
        fsc_sqd = self.fsc ** 2
        fsc_combi = fsc_sqd / (1 - fsc_sqd)
        val_arr[:,1] = fsc_combi
        #wgrid = subroutine2.read_into_grid(self.mapobj.cbin_idx,val_arr, self.mapobj.cbin, cx, cy, cz)
        wgrid = fcodes_fast.read_into_grid2(self.mapobj.cbin_idx,val_arr, self.mapobj.cbin, cx, cy, cz)
        w_grid = wgrid[:,:,:,0]
        w2_grid = wgrid[:,:,:,1]
        return w_grid, w2_grid

    def functional(self):
        fval = np.sum(self.w_grid * self.e0 * np.conjugate(self.ert))
        return fval.real

    def minimizer(self, ncycles, t_init, rotmat, ifit, smax_lf, fobj=None, q_init=None):
        fsc_lst = []
        fval_list = []
        q_list = []
        t_list = []
        self.e0 = self.mapobj.ceo_lst[0]  # Static map e-data for fit
        xyz = create_xyz_grid(self.cell, self.cut_dim)
        xyz_sum = get_xyz_sum(xyz)
        if q_init is None:
            q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        print("pixel size: ", self.pixsize)
        print("Cycle#   ", "Fval  ", "Rot(deg)  ", "Trans(A)  ", "avg(FSC)")
        self.e1 = self.mapobj.ceo_lst[1]
        self.cfo = self.mapobj.cfo_lst[0]
        mapdim = self.mapobj.map_dim[0]
        self.t = np.asarray(t_init, dtype="float")
        #translation_vec = 0.0
        #self.st = 1.0
        self.rotmat = rotmat
        self.q = quaternions.rot2quart(self.rotmat)
        theta2 = np.arccos((np.trace(self.rotmat) - 1) / 2) * 180.0 / np.pi
        for i in range(ncycles):
            start = timer()
            if i % 2 == 0:
                # translation
                self.fsc = self.calc_fsc_t()
                self.w_grid, self.w2_grid = self.get_wght()
                fval = self.functional()
                fval_list.append(fval)
                self.step = derivatives_translation(self.e0, self.ert, self.w_grid, self.w2_grid, self.sv)
                lft = linefit()
                lft.get_linefit_static_data([self.e0, self.ert], self.mapobj.cbin_idx, self.mapobj.res_arr, smax_lf)
                lft.step = self.step
                #lft.t = self.t
                alpha = lft.scalar_opt_trans()
                # translation
                self.t = self.t + self.step * alpha
                t_accum_angstrom = self.t * self.pixsize * mapdim
                translation_vec = np.sqrt(np.dot(t_accum_angstrom, t_accum_angstrom))
            else:
                # rotation
                self.rotmat = core.quaternions.get_RM(self.q)
                theta2 = np.arccos((np.trace(self.rotmat) - 1) / 2) * 180.0 / np.pi
                self.fsc = self.calc_fsc_r()
                fsc_lst.append(self.fsc)
                self.w_grid, self.w2_grid = self.get_wght()
                fval = self.functional()
                fval_list.append(fval)
                self.step = derivatives_rotation(
                    self.e0, self.ert, self.crt, self.w_grid, self.sv, self.q, xyz, xyz_sum
                )
                lft = linefit()
                """ lft.get_linefit_static_data(
                    [self.e0, self.e1], self.mapobj.cbin_idx, self.mapobj.res_arr, smax_lf
                ) """
                if i == 1:
                    #f_list = [self.e0, self.e1]
                    f_list = [self.e0, self.ert]
                    fout, lbinindx, lnbin = cut_resolution_for_linefit(f_list, self.mapobj.cbin_idx, self.mapobj.res_arr, smax_lf)
                    self.le0 = fout[0,:,:,:]
                    self.le1 = fout[1,:,:,:]
                    self.lbinindx = lbinindx
                    self.lnbin = lnbin
                lft.e0 = self.le0
                lft.e1 = self.le1
                lft.bin_idx = self.lbinindx
                lft.nbin = self.lnbin
                lft.step = self.step
                lft.q_prev = self.q
                alpha_r = lft.scalar_opt()
                tmp = np.insert(self.step * alpha_r, 0, 0.0)
                self.q = self.q + tmp
                self.q = self.q / np.sqrt(np.dot(self.q, self.q))
                self.rotmat = core.quaternions.get_RM(self.q)
                theta2 = np.arccos((np.trace(self.rotmat) - 1) / 2) * 180.0 / np.pi
                
            print(
                "{:5d} {:8.4f} {:6.4f} {:6.4f} {:6.4f}".format(
                    i, fval, theta2, translation_vec, np.average(self.fsc)
                )
            )

            end = timer()
            #if timeit:
                #print("time for one cycle:", end - start)


def fsc_between_static_and_transfomed_map(
    staticmap, movingmap, bin_idx, rm, t, cell, nbin
):
    nx, ny, nz = staticmap.shape
    st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
    #frt_full = subroutine2.trilinear2(movingmap,bin_idx,rm,nbin,0,1,nx, ny, nz)[:, :, :, 0]
    frt_full = fcodes_fast.trilinear2(movingmap,bin_idx,rm,nbin,0,1,nx, ny, nz)[:, :, :, 0]
    f1f2_fsc = core.fsc.anytwomaps_fsc_covariance(staticmap, frt_full, bin_idx, nbin)[0]
    f1f2_fsc = np.nan_to_num(f1f2_fsc, copy=False, nan=0.0) # for aesthetics
    return f1f2_fsc


def get_ibin(bin_fsc):
    # search from rear end
    for i, ifsc in reversed(list(enumerate(bin_fsc))):
        if ifsc > 0.4:
            ibin = i
            if ibin % 2 != 0:
                ibin = ibin - 1
            break
    return ibin


def run_fit(
    emmap1,
    smax,
    rotmat,
    t,
    slf,
    ncycles,
    ifit,
    fobj=None,
    interp=None,
    fitres=None,
):
    from emda.ext.mapfit import frequency_marching

    if fitres is not None:
        if fitres <= emmap1.res_arr[-1]:
            fitbin = len(emmap1.res_arr) - 1
        else:
            dist = np.sqrt((emmap1.res_arr - fitres) ** 2)
            ibin = np.argmin(dist)
            if ibin % 2 != 0:
                ibin = ibin - 1
            fitbin = min([len(dist), ibin])
    if fitres is None:
        fitbin = len(emmap1.res_arr) - 1
    fsc_lst = []
    nmarchingcycles = 10
    for i in range(nmarchingcycles):
        print("Frequency marching cycle # ", i)
        if i == 0:
            f1f2_fsc = fsc_between_static_and_transfomed_map(
                staticmap=emmap1.fo_lst[0],
                movingmap=emmap1.fo_lst[ifit],
                bin_idx=emmap1.bin_idx,
                rm=rotmat,
                t=t,
                cell=emmap1.map_unit_cell,
                nbin=emmap1.nbin,
            )
            ibin = get_ibin(f1f2_fsc)
            if fitbin < ibin:
                ibin = fitbin
            ibin_old = ibin
            print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
            fsc_lst.append(f1f2_fsc)
            if np.average(f1f2_fsc) > 0.999:
                rotmat = rotmat
                t = t
                q_final = quaternions.rot2quart(rotmat)
                fsc_lst.append(f1f2_fsc)
                print("\n***FSC between static and moving maps***\n")
                print("bin#     resolution(A)      start-FSC     end-FSC\n")
                for j in range(len(emmap1.res_arr)):
                    print(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                            j, emmap1.res_arr[j], fsc_lst[1][j], fsc_lst[0][j]
                        )
                    )
                break
        else:
            # Apply initial rotation and translation to calculate fsc
            """ f1f2_fsc = fsc_between_static_and_transfomed_map(
                emmap1.fo_lst[0],
                emmap1.fo_lst[ifit],
                emmap1.bin_idx,
                rotmat,
                t,
                emmap1.map_unit_cell,
                emmap1.nbin,
            ) """
            f1f2_fsc = core.fsc.anytwomaps_fsc_covariance(emmap1.fo_lst[0], frt, emmap1.bin_idx, emmap1.nbin)[0]
            ibin = get_ibin(f1f2_fsc)
            if fitbin < ibin:
                ibin = fitbin
            print("Fitting resolution: ", emmap1.res_arr[ibin], " (A)")
            print("FSC(ibin): ", f1f2_fsc[ibin])
            if ibin_old == ibin or i == nmarchingcycles-1:
                fsc_lst.append(f1f2_fsc)
                q_final = quaternions.rot2quart(rotmat)
                print("\n***FSC between static and moving maps***\n")
                print("bin#     resolution(A)      start-FSC     end-FSC\n")
                for j in range(len(emmap1.res_arr)):
                    print(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                            j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[1][j]
                        )
                    )
                break
            else:
                ibin_old = ibin
        if ibin == 0:
            print("ibin = 0")
            raise SystemExit("Cannot proceed! Stopping now...")
        e_list = [emmap1.eo_lst[0], emmap1.eo_lst[ifit], emmap1.fo_lst[ifit]]
        eout, cBIdx, cbin = cut_resolution_for_linefit(
            e_list, emmap1.bin_idx, emmap1.res_arr, ibin
        )
        static_cutmap = eout[0, :, :, :]
        moving_cutmap = eout[1, :, :, :]
        cfo = eout[2, :, :, :]
        emmap1.ceo_lst = [static_cutmap, moving_cutmap]
        emmap1.cfo_lst = [cfo]
        emmap1.cbin_idx = cBIdx
        emmap1.cdim = moving_cutmap.shape
        emmap1.cbin = cbin
        rfit = EmFit(emmap1)
        if ibin < slf or slf == 0:
            slf = ibin
        #slf = min([ibin, slf])
        slf = ibin
        rfit.minimizer(ncycles, t, rotmat, ifit, smax_lf=slf, fobj=fobj)
        t = rfit.t
        rotmat = quaternions.get_RM(rfit.q)
        # apply transformation on data for next cycle
        nx, ny, nz = emmap1.map_dim
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        frt = utils.get_FRS(rotmat, emmap1.fo_lst[ifit] * st, interp="linear")[:, :, :, 0] 
    return t, q_final


def overlay(
    maplist,
    ncycles=10,
    t_init=[0.0, 0.0, 0.0],
    rotmat_init=np.identity(3),
    smax=6,
    interp="linear",
    modelres=5.0,
    usecom=False,
    masklist=None,
    fobj=None,
    fitres=None,
):
    try:
        emmap1 = EmmapOverlay(map_list=maplist, modelres=modelres, com=usecom, mask_list=masklist)
    except:
        emmap1 = EmmapOverlay(map_list=maplist, modelres=modelres, com=usecom)
    emmap1.load_maps()
    emmap1.calc_fsc_from_maps()
    t = [itm / emmap1.pixsize for itm in t_init]
    rotmat_lst = []
    transl_lst = []
    # resolution estimate for line-fit
    dist = np.sqrt((emmap1.res_arr - smax) ** 2)
    slf = np.argmin(dist) + 1
    if slf % 2 != 0:
        slf = slf - 1
    slf = min([len(dist), slf])
    q_init_list = []
    rotaxis_list = []
    rotmat_list = []
    trans_list = []
    q_final_list = []
    for ifit in range(1, len(emmap1.eo_lst)):
        t, q_final = run_fit(
            emmap1=emmap1,
            smax=smax,
            rotmat=rotmat_init,
            t=t,
            slf=slf,
            ncycles=ncycles,
            ifit=ifit,
            interp=interp,
            fitres=fitres,
        )
        rotmat = quaternions.get_RM(q_final)
        rotmat_list.append(rotmat)
        trans_list.append(t)
    # output maps
    output_rotated_maps(emmap1, rotmat_list, trans_list)
    output_rotated_models(emmap1, maplist, rotmat_list, trans_list)
    return emmap1, rotmat_list, trans_list


def output_rotated_maps(emmap1, r_lst, t_lst, Bf_arr=None):
    from numpy.fft import fftn, fftshift, ifftshift, ifftn, ifftshift
    from emda.ext.mapfit import utils
    from scipy.ndimage.interpolation import shift

    if Bf_arr is None:
        Bf_arr = [0.0]
    fo_lst = emmap1.fo_lst
    cell = emmap1.map_unit_cell
    comlist = emmap1.comlist
    bin_idx = emmap1.bin_idx
    nbin = emmap1.nbin
    imap_f = 0
    f_static = fo_lst[0]
    nx, ny, nz = f_static.shape
    data2write = np.real(ifftshift(ifftn(ifftshift(f_static))))
    print("COM list: ", comlist)
    if len(comlist) > 0:
        data2write = em.shift_density(data2write, shift=np.subtract(comlist[0], emmap1.box_centr))
    core.iotools.write_mrc(data2write, "static_map.mrc", cell)
    i = 0
    for fo, t, rotmat in zip(fo_lst[1:], t_lst, r_lst):
        i += 1
        f1f2_fsc_unaligned = core.fsc.anytwomaps_fsc_covariance(
            f_static, fo, bin_idx, nbin
        )[0]
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        frt = utils.get_FRS(rotmat, fo * st, interp="cubic")[:, :, :, 0]
        data2write = np.real(ifftshift(ifftn(ifftshift(frt))))
        # apply reverse shift to density
        if len(comlist) > 0:
            data2write = em.shift_density(data2write, shift=np.subtract(comlist[0], emmap1.box_centr))
        core.iotools.write_mrc(
            data2write,
            "{0}_{1}.{2}".format("fitted_map", str(i), "mrc"),
            cell,
        )
        # estimating covaraince between current map vs. static map
        f1f2_fsc = core.fsc.anytwomaps_fsc_covariance(f_static, frt, bin_idx, nbin)[0]
        core.plotter.plot_nlines(
            emmap1.res_arr,
            [f1f2_fsc_unaligned[: emmap1.nbin], f1f2_fsc[: emmap1.nbin]],
            "{0}_{1}.{2}".format("fsc", str(i), "eps"),
            ["FSC before", "FSC after"],
        )


def output_rotated_models(emmap1, maplist, r_lst, t_lst):
    from emda.core.iotools import apply_transformation_on_model, pdb2mmcif

    pixsize = emmap1.pixsize
    i = 0
    for model, t, rotmat in zip(maplist[1:], t_lst, r_lst):
        print(rotmat)
        t = np.asarray(t, 'float') *  pixsize * np.asarray(emmap1.map_dim, 'int')
        print(t)
        i += 1
        if model.endswith((".mrc", ".map")):
            continue
        elif model.endswith((".pdb", ".ent")):
            pdb2mmcif(model)
            outcifname = "emda_transformed_model_" + str(i) + ".cif"
            print(outcifname)
            _,_,_,_ = apply_transformation_on_model(mmcif_file="./out.cif",rotmat=rotmat, trans=t, outfilename=outcifname)
        elif model.endswith((".cif")):
            outcifname = "emda_transformed_model_" + str(i) + ".cif"
            _,_,_,_ = apply_transformation_on_model(mmcif_file=model,rotmat=rotmat, trans=t, outfilename=outcifname)


if __name__ == "__main__":
    maplist = [
        #"/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/emda_test/map_transform/emd_3651.map",
        #"/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/emda_test/map_transform/transformed.mrc"
        #"/Users/ranganaw/MRC/REFMAC/EMD-6952/emda_test/map_transform/emd_6952.map",
        #"/Users/ranganaw/MRC/REFMAC/EMD-6952/emda_test/map_transform/transformed.mrc",
        #"/Users/ranganaw/MRC/REFMAC/Vinoth/emda_test/diffmap/postprocess_nat.mrc",
        #"/Users/ranganaw/MRC/REFMAC/Vinoth/emda_test/diffmap/postprocess_lig.mrc",
        "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/EMD-9931/emd_9931.map",
        "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/EMD-9934/emd_9934.map",
    ]

    masklist = [
        "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/DomainMasks/6k7g-9931_A_mask.mrc",
        "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/DomainMasks/6k7i-9934_A_mask.mrc"
    ]
    emmap1, rotmat_lst, transl_lst = overlay(maplist=maplist, masklist=masklist)