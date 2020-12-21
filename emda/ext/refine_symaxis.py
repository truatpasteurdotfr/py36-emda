# lock rotation
from __future__ import absolute_import, division, print_function, unicode_literals
from timeit import default_timer as timer
import numpy as np
import math
import fcodes_fast
import emda.emda_methods as em
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from emda.ext.mapfit import map_class
from emda import core
from emda.core import quaternions
from emda.ext.mapfit.utils import (
    get_FRS,
    get_fsc_wght,
    create_xyz_grid,
    get_xyz_sum,
    set_dim_even,
)


def output_rotated_maps(emdcode, emmap1, r_lst, t_lst, Bf_arr=None):
    if Bf_arr is None:
        Bf_arr = [0.0]
    cell = emmap1.map_unit_cell
    bin_idx = emmap1.bin_idx
    nbin = emmap1.nbin
    imap_f = 0
    f_static = emmap1.fhf_lst[0]
    nx, ny, nz = f_static.shape
    # data2write = np.real(ifftshift(ifftn(ifftshift(f_static))))
    # core.iotools.write_mrc(data2write, "static_map.mrc", cell)
    i = 0
    t = t_lst
    for rotmat in r_lst:
        i += 1
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        f_moving = get_FRS(rotmat, f_static * st, interp="linear")[:, :, :, 0]
        """ data2write = np.real(ifftshift(ifftn(ifftshift(f_moving))))
        core.iotools.write_mrc(
            data2write,
            "{0}_{1}.{2}".format("fitted_map", str(i), "mrc"),
            cell,
        ) """
        # estimating covaraince between current map vs. static map
        f1f2_fsc = core.fsc.anytwomaps_fsc_covariance(
            f_static, f_moving, bin_idx, nbin
        )[0]
        core.plotter.plot_nlines(
            emmap1.res_arr,
            [f1f2_fsc[: emmap1.nbin]],
            "{0}_{1}_{2}.{3}".format(emdcode, "fsc", str(imap_f + 1), "eps"),
            ["FSC after"],
        )


class EmmapOverlay:
    def __init__(self, imap, imask=None):
        self.imap = imap
        self.imask = imask
        self.map_unit_cell = None
        self.map_origin = None
        self.map_dim = None
        self.pixsize = None
        self.arr_lst = []
        self.ceo_lst = None
        self.cfo_lst = None
        self.cbin_idx = None
        self.cdim = None
        self.cbin = None
        self.com = True
        self.com1 = None
        self.box_centr = None
        self.fhf_lst = []
        self.nbin = None
        self.res_arr = None
        self.bin_idx = None
        self.fo_lst = None
        self.eo_lst = None
        self.totalvar_lst = None
        self.q_init_list = []

    def get_maps(self):
        uc, arr, origin = em.get_data(self.imap)
        arr = set_dim_even(arr)
        com = em.center_of_mass_density(arr)
        print("com:", com)
        nx, ny, nz = arr.shape
        box_centr = (nx // 2, ny // 2, nz // 2)
        arr = em.shift_density(arr, np.subtract(box_centr, com))
        print("com after centering:", em.center_of_mass_density(arr))
        self.arr_lst.append(arr)
        self.fhf_lst.append(fftshift(fftn(fftshift(arr))))
        self.pixsize = uc[0] / arr.shape[0]
        self.map_origin = origin
        self.map_unit_cell = uc
        self.map_dim = arr.shape

    def calc_fsc_from_maps(self):
        # function for only two maps fitting
        nmaps = len(self.fhf_lst)
        fFo_lst = []
        fEo_lst = []
        fBTV_lst = []
        self.nbin, self.res_arr, self.bin_idx = core.restools.get_resolution_array(
            self.map_unit_cell, self.fhf_lst[0]
        )
        for i in range(nmaps):
            _, _, _, totalvar, fo, eo = core.fsc.halfmaps_fsc_variance(
                self.fhf_lst[i], self.fhf_lst[i], self.bin_idx, self.nbin
            )
            fFo_lst.append(fo)
            fEo_lst.append(eo)
            fBTV_lst.append(totalvar)
        self.fo_lst = fFo_lst
        self.eo_lst = fEo_lst
        self.totalvar_lst = fBTV_lst


class linefit:
    def __init__(self):
        self.e0 = None
        self.e1 = None
        self.bin_idx = None
        self.nbin = None
        self.axis = None
        self.angle_list = []
        self.step_axis = None
        self.step_t = None
        self.alpha_t = None
        self.axis_ini = None

    def get_linefit_static_data(self, e0, bin_idx, res_arr, smax):
        (
            self.e0,
            self.bin_idx,
            self.nbin,
        ) = core.restools.cut_resolution_for_linefit(e0, bin_idx, res_arr, smax)
        self.e1 = self.e0

    def get_quaternion(self, axis, angle_i):
        rv = axis
        s = math.sin(angle_i / 2.0)
        q1, q2, q3 = rv[0] * s, rv[1] * s, rv[2] * s
        q0 = math.cos(angle_i / 2.0)
        q = np.array([q0, q1, q2, q3], dtype=np.float64)
        return q

    def get_fsc_wght(self, e0, ert, bin_idx, nbin):
        cx, cy, cz = e0.shape
        bin_stats = core.fsc.anytwomaps_fsc_covariance(e0, ert, bin_idx, nbin)
        fsc, _ = bin_stats[0], bin_stats[1]
        fsc = np.array(fsc, dtype=np.float64, copy=False)
        w_grid = fcodes_fast.read_into_grid(bin_idx, fsc, nbin, cx, cy, cz)
        w_grid = np.array(w_grid, dtype=np.float64, copy=False)
        return w_grid

    def func(self, i):
        step_ax = self.step_axis * i
        """ ay, az = step_ax[1], step_ax[2]  # for emfit3
        # ay, az = step_ax[0], step_ax[1] # for emfit4
        ax = np.sqrt(1.0 - ay * ay - az * az)
        self.axis = np.array([ax, ay, az], dtype="float") """
        self.axis = getaxis(self.axis_ini, step_ax)
        fval = 0.0
        for angle_i in self.angle_list:
            q_i = self.get_quaternion(self.axis, angle_i)
            rotmat = core.quaternions.get_RM(q_i)
            ers = get_FRS(rotmat, self.e1, interp="linear")
            w_grid = get_fsc_wght(self.e0, ers[:, :, :, 0], self.bin_idx, self.nbin)
            fval += np.real(np.sum(w_grid * self.e0 * np.conjugate(ers[:, :, :, 0])))
        return -fval

    def func_t(self, i):
        nx, ny, nz = self.e0.shape
        t = self.step_t * i
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        e1_t = self.e1 * st
        w_grid = get_fsc_wght(self.e0, e1_t, self.bin_idx, self.nbin)
        fval = np.sum(w_grid * self.e0 * np.conjugate(e1_t))
        return -fval.real

    def scalar_opt_trans(self):
        from scipy.optimize import minimize_scalar

        f = self.func_t
        res = minimize_scalar(f, method="brent")
        return res.x

    def scalar_opt(self, alpha_t=0.0):
        from scipy.optimize import minimize_scalar

        nx, ny, nz = self.e1.shape
        # self.alpha_t = self.scalar_opt_trans()
        t = self.step_t * alpha_t
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        self.e1 = self.e1 * st
        f = self.func
        res = minimize_scalar(f, method="brent")
        return res.x


def derivatives(e0, e1, w_grid, w2_grid, q, sv, xyz, xyz_sum, vol):
    from emda.core.quaternions import derivatives_wrt_q
    from emda.ext.mapfit.derivatives import get_dqda, new_dFs2

    nx, ny, nz = e0.shape
    sv_np = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    for i in range(3):
        sv_np[:, :, :, i] = sv[i]
    dRdq = derivatives_wrt_q(q)
    dqda = get_dqda(q)
    dFRs = new_dFs2(np.real(np.fft.ifftn(np.fft.ifftshift(e1))), xyz, vol)
    df_val, ddf_val = fcodes_fast.calc_derivatives(
        e0, e1, w_grid, w2_grid, sv_np, dFRs, dRdq, xyz_sum, vol, nx, ny, nz
    )
    ddf_val_inv = np.linalg.pinv(ddf_val)
    step = ddf_val_inv.dot(-df_val)
    # get axis derivatives
    df_ax = np.zeros(3, dtype="float")
    ddf_ax = np.zeros((3, 3), dtype="float")
    dqda = get_dqda(q)
    for i in range(len(dqda)):
        df_ax[i] = df_val[3 + i] * dqda[i]
        for j in range(len(dqda)):
            ddf_ax[i, j] = ddf_val[3 + i, 3 + j] * dqda[i] * dqda[j]
    ddf_ax_inv = np.linalg.pinv(ddf_ax)
    step_ax = ddf_ax_inv.dot(-df_ax)
    return step[:3], step_ax


def getaxis(axis_ini, step_ax):
    axmax = np.argmax(abs(axis_ini))
    if axmax == 0:
        ay, az = step_ax[1], step_ax[2]
        ax = np.sqrt(1.0 - ay * ay - az * az)
    elif axmax == 1:
        ax, az = step_ax[0], step_ax[2]
        ay = np.sqrt(1.0 - ax * ax - az * az)
    elif axmax == 2:
        ax, ay = step_ax[0], step_ax[1]
        az = np.sqrt(1.0 - ax * ax - ay * ay)
    return np.array([ax, ay, az], dtype="float")


class EmFit:
    def __init__(self, mapobj, interp="linear", dfs=None):
        self.mapobj = mapobj
        self.ful_dim = mapobj.map_dim
        self.cell = mapobj.map_unit_cell
        self.pixsize = mapobj.pixsize
        self.origin = mapobj.map_origin
        self.interp = interp
        self.dfs = dfs
        self.cut_dim = mapobj.map_dim
        self.nbin = mapobj.nbin
        self.bin_idx = mapobj.bin_idx
        self.res_arr = mapobj.res_arr
        self.w_grid = None
        self.fsc = None
        self.sv = None
        self.t = None
        self.st = None
        self.step = None
        self.q = None
        self.axis = None
        self.angle = None
        self.rotmat = None
        self.t_accum = None
        self.ert = None
        self.frt = None
        self.e0 = None
        self.e1 = None
        self.w2_grid = None
        self.fsc_lst = []
        self.q_final_list = []
        self.t_final_list = []
        self.axis_final_list = []

    def calc_fsc(self, e1, t, rotmat):
        cx, cy, cz = self.e0.shape
        st, s1, s2, s3 = fcodes_fast.get_st(cx, cy, cz, t)
        sv = np.array([s1, s2, s3])
        ert = get_FRS(rotmat, e1 * st, interp=self.interp)[:, :, :, 0]
        fsc = core.fsc.anytwomaps_fsc_covariance(self.e0, ert, self.bin_idx, self.nbin)[
            0
        ]
        return fsc, st, sv, ert

    def get_wght(self, fsc):
        cx, cy, cz = self.e0.shape
        w_grid = fcodes_fast.read_into_grid(
            self.bin_idx,
            fsc / (1 - fsc ** 2),
            self.nbin,
            cx,
            cy,
            cz,
        )
        fsc_sqd = fsc ** 2
        fsc_combi = fsc_sqd / (1 - fsc_sqd)
        w2_grid = fcodes_fast.read_into_grid(
            self.bin_idx, fsc_combi, self.nbin, cx, cy, cz
        )
        return w_grid, w2_grid

    def functional(self, w_grid, ert):
        fval = np.sum(w_grid * self.e0 * np.conjugate(ert))
        return fval.real

    def get_quaternion(self, angle_i):
        rv = self.axis
        s = math.sin(angle_i / 2.0)
        q1, q2, q3 = rv[0] * s, rv[1] * s, rv[2] * s
        q0 = math.cos(angle_i / 2.0)
        # Quaternion
        q = np.array([q0, q1, q2, q3], dtype=np.float64)
        return q

    def minimizer(self, ncycles, t_init, q_init_list, smax_lf):
        fsc_lst = []
        fval_list = []
        q_list = []
        t_list = []
        nfit = len(q_init_list)
        self.e0 = self.mapobj.eo_lst[0]
        self.e1 = self.e0
        xyz = create_xyz_grid(self.cell, self.cut_dim)
        vol = self.cut_dim[0] * self.cut_dim[0] * self.cut_dim[0]
        xyz_sum = get_xyz_sum(xyz)
        print("Cycle#   ", "Fval  ", "avg(FSC)")
        self.t = np.asarray(t_init, dtype="float")
        angle_list = []
        for i in range(ncycles):
            j = 0
            fval = 0.0
            avg_fsc = 0.0
            step_tr_sum = np.array([0.0, 0.0, 0.0], dtype="float")
            step_ax_sum = np.array([0.0, 0.0, 0.0], dtype="float")
            for ifit in range(nfit):
                if i == 0:
                    q_i = q_init_list[ifit]
                    axis_ang = np.asarray(quaternions.quart2axis(q_i), dtype="float")
                    angle_list.append(axis_ang[-1])
                    axis_ini = np.array(axis_ang[:3], dtype="float")
                else:
                    q_i = self.get_quaternion(angle_list[ifit])
                rotmat_i = quaternions.get_RM(q_i)
                fsc, st, sv, ert = self.calc_fsc(self.e1, self.t, rotmat_i)
                avg_fsc += np.average(fsc)
                w_grid, w2_grid = self.get_wght(fsc)
                fval += self.functional(w_grid, ert)
                step_tr, step_ax = derivatives(
                    self.e0,
                    ert,
                    w_grid,
                    w2_grid,
                    q_i,
                    sv,
                    xyz,
                    xyz_sum,
                    vol,
                )
                step_tr_sum += step_tr
                step_ax_sum += step_ax
            fval_list.append(fval)
            avg_fsc = avg_fsc / nfit
            step_tr = step_tr_sum / nfit
            step_ax = step_ax_sum / nfit
            print("{:5d} {:8.4f} {:6.4f}".format(i, fval, avg_fsc))
            if avg_fsc < 0.4:
                print(
                    "Average FSC is lower than 0.4. The axis being refined may not be correct"
                )
            lft = linefit()
            lft.angle_list = angle_list
            lft.axis_ini = axis_ini
            lft.step_axis = step_ax
            lft.step_t = step_tr
            lft.get_linefit_static_data(self.e0, self.bin_idx, self.res_arr, smax_lf)
            # alpha_t = lft.scalar_opt_trans()
            alpha = lft.scalar_opt()
            # translation
            self.t = self.t  # + step_tr * alpha_t
            # axis
            step_ax = step_ax * alpha
            self.axis = getaxis(axis_ini, step_ax)
            if avg_fsc > 0.9999:
                break
        # plot fval vs ncycle
        from matplotlib import pyplot as plt

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        ax1.plot(fval_list, label="fval", linewidth=2)
        plt.savefig("fval_vs_cycle.eps", format="eps", dpi=300)
        # plt.show()


def overlay(
    emdcode,
    imap,
    rotaxis,
    symorder,
    ncycles=5,
    t_init=[0.0, 0.0, 0.0],
    theta_init=0.0,
    smax=10,
    interp="linear",
    imask=None,
    fobj=None,
    halfmaps=False,
    dfs_interp=False,
    usemodel=False,
    fitres=None,
):
    axis = np.asarray(rotaxis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    initial_axis = axis
    fobj.write("   Initial axis: " + str(axis) + "\n")
    try:
        emmap1 = EmmapOverlay(imap=imap, imask=imask)
    except:
        emmap1 = EmmapOverlay(imap=imap)
    emmap1.get_maps()
    emmap1.calc_fsc_from_maps()
    # resolution estimate for line-fit
    dist = np.sqrt((emmap1.res_arr - smax) ** 2)
    slf = np.argmin(dist) + 1
    if slf % 2 != 0:
        slf = slf - 1
    slf = min([len(dist), slf])
    q_init_list = []
    angle_list = []
    rotmat_list = []
    for n in range(1, symorder):
        theta = float(n * 360.0 / symorder)
        angle_list.append(theta)
        q = quaternions.get_quaternion([[axis[2], axis[1], axis[0]], [theta]])
        print(q)
        q_init_list.append(q)
    fit = EmFit(emmap1)
    fit.minimizer(ncycles=ncycles, t_init=t_init, smax_lf=slf, q_init_list=q_init_list)
    final_axis = fit.axis
    final_tran = fit.t
    print("Final axis: ", final_axis[::-1])
    #print("Final translation (A): ", final_tran * emmap1.pixsize)
    fobj.write("   Final axis: " + str(final_axis[::-1]) + "\n")
    ax = final_axis
    for angle in angle_list:
        q = quaternions.get_quaternion([[ax[0], ax[1], ax[2]], [angle]])
        rotmat_list.append(quaternions.get_RM(q))
    output_rotated_maps(emdcode, emmap1, rotmat_list, final_tran)
    return initial_axis, final_axis[::-1]


def get_intial_axis(imap):
    from emda.ext import run_proshade as proshade

    symorder, x, y, z, theta = proshade.get_symmops_from_proshade(imap)
    return x, y, z, symorder


def main(maplist, emdbidlist=None, fobj=None):
    if fobj is None:
        fobj = open("EMDA_symref.txt", "w")
    refined_axis_ang = []
    initial_ax_list = []
    final_ax_list = []
    fold_list = []
    emdcode_list = []
    if emdbidlist is not None:
        assert len(emdbidlist) == len(maplist)
    else:
        ids = np.arange(len(maplist))
        emdbidlist = [",".join(item) for item in ids.astype(str)]
    for imap, emdbid in zip(maplist, emdbidlist):
        emdcode = "EMD-" + emdbid
        fobj.write("Map: " + imap + "\n")
        fobj.write("emdcode: " + emdcode + "\n")
        x, y, z, symorder = get_intial_axis(imap)
        i = 0
        for ix, iy, iz, fold in zip(x, y, z, symorder):
            rotaxis = [ix, iy, iz]
            emdcode = "{0}_{1}".format(emdcode, str(i))
            i += 1
            initial_axis, final_axis = overlay(
                emdcode=emdcode, imap=imap, rotaxis=rotaxis, symorder=fold, fobj=fobj
            )
            #refined_axis_ang.append(quaternions.quart2axis(q))
            emdcode_list.append(emdcode)
            fold_list.append(fold)
            initial_ax_list.append(initial_axis)
            final_ax_list.append(final_axis)
            #print(emdcode, fold, initial_axis, final_axis)
    #datadict = dict(zip(emdcode_list, fold_list, initial_ax_list, final_ax_list))
    return emdcode_list, fold_list, initial_ax_list, final_ax_list


if __name__ == "__main__":
    # imap = ["/Users/ranganaw/MRC/REFMAC/EMD-0882/map/emd_0882.map"]
    imap = [
        "/Users/ranganaw/MRC/REFMAC/EMD-0882/emda_test/group_rotation/test_pipeline/emd_0959.map"
    ]
    main(imap)