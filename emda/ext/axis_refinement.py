# lock rotation
from __future__ import absolute_import, division, print_function, unicode_literals
from timeit import default_timer as timer
import numpy as np
import sys, math
import fcodes_fast
import emda.emda_methods as em
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from emda import core
from emda.core import quaternions, plotter
from emda.ext.mapfit.utils import (
    get_FRS,
    create_xyz_grid,
    get_xyz_sum,
    set_dim_even,
    sphere_mask
)
from emda.core.quaternions import derivatives_wrt_q
from emda.ext.mapfit.derivatives import new_dFs2#, get_dqda

class EmmapOverlay:
    def __init__(self, imap, imask=None):
        self.imap = imap
        self.imask = imask
        self.map_unit_cell = None
        self.map_origin = None
        self.map_dim = None
        self.pixsize = None
        self.arr_lst = []
        self.arr_original = None
        self.ceo_lst = None
        self.cfo_lst = None
        self.cbin_idx = None
        self.cdim = None
        self.cbin = None
        self.com = True
        self.com1 = None
        self.com2 = None
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
        if self.imask is not None:
            _, mask, _ = em.get_data(self.imask)
            mask = mask * mask > 1e-4
            arr = set_dim_even(arr * mask)
        else:
            arr = arr * sphere_mask(min(arr.shape))
            arr = set_dim_even(arr)
        self.arr_original = arr
        com = em.center_of_mass_density(arr)
        print("com:", com)
        nx, ny, nz = arr.shape
        box_centr = (nx // 2, ny // 2, nz // 2)
        self.box_centr = box_centr
        self.com1 = com
        if self.com:
            arr = em.shift_density(arr, np.subtract(box_centr, com))
            self.com2 = em.center_of_mass_density(arr)
            print("com after centering:", self.com2)
        self.arr_lst.append(arr)
        self.fhf_lst.append(fftshift(fftn(fftshift(arr))))
        self.pixsize = uc[0] / arr.shape[0]
        self.map_origin = origin
        self.map_unit_cell = uc
        self.map_dim = arr.shape

    def calc_fsc_from_maps(self):
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

def cut_resolution_for_linefit(f_list, bin_idx, res_arr, smax):
    # Making data for map fitting
    f_arr = np.asarray(f_list, dtype="complex")
    nx, ny, nz = f_list[0].shape
    cbin = cx = smax
    dx = int((nx - 2 * cx) / 2)
    dy = int((ny - 2 * cx) / 2)
    dz = int((nz - 2 * cx) / 2)
    cBIdx = bin_idx[dx: dx + 2 * cx, dy: dy + 2 * cx, dz: dz + 2 * cx]
    fout = fcodes_fast.cutmap_arr(
        f_arr, bin_idx, cbin, 0, len(res_arr), nx, ny, nz, len(f_list)
    )[:, dx: dx + 2 * cx, dy: dy + 2 * cx, dz: dz + 2 * cx]
    return fout, cBIdx, cbin

def get_dqda(q):
    from emda.core.quaternions import quart2axis
    ax, ay, _, angle = quart2axis(q)
    az = float(math.sqrt(1 - ax**2 - ay**2))
    dqda = np.zeros(3, dtype='float')
    dqda[0,0] = dqda[1,1] = dqda[2,2] = math.sin(angle/2) # diagonals
    dqda[0,1] = dqda[1,0] = 0.
    try:
        dqda[0,2] = dqda[2,0] = (-ax / az) * math.sin(angle/2)
        dqda[1,2] = dqda[2,1] = (-ay / az) * math.sin(angle/2)
    except:
        raise SystemExit("problem in dqda matrix!")
    return dqda

class Bfgs:
    def __init__(self):
        self.method = "BFGS"
        self.e0 = None
        self.e1 = None
        self.ax_init = None
        self.ax_final = None
        self.angle = None
        self.q = np.array([1., 0., 0., 0.], 'float')
        self.t = np.array([0., 0., 0.], 'float')
        self.xyz = None
        self.xyz_sum = None
        self.vol = None
        self.bin_idx = None
        self.binfsc = None
        self.nbin = None
        self.x = np.array([0.1, 0.1, 0.1], 'float')

    def get_quaternion(self, axis, angle):
        angle = np.deg2rad(angle)
        rv = axis
        s = math.sin(angle / 2.0)
        q1, q2, q3 = rv[0] * s, rv[1] * s, rv[2] * s
        q0 = math.cos(angle / 2.0)
        q = np.array([q0, q1, q2, q3], dtype=np.float64)
        return q

    def derivatives(self, x):
        nx, ny, nz = self.e0.shape
        ax = self.ax_init + x
        q = self.get_quaternion(axis=ax, angle=self.angle)
        rotmat = core.quaternions.get_RM(q)
        e1 = get_FRS(rotmat, self.e0, interp='linear')[:, :, :, 0]
        wgrid = self.get_wght(e1)
        dRdq = derivatives_wrt_q(q)
        st, sv1, sv2, sv3 = fcodes_fast.get_st(nx, ny, nz, x[3:])
        sv_np = np.stack((sv1, sv2, sv3), axis = -1)
        dFRs = new_dFs2(np.real(np.fft.ifftn(np.fft.ifftshift(e1))), self.xyz, self.vol)
        df_val, ddf_val = fcodes_fast.calc_derivatives(
            self.e0, e1*st, wgrid, wgrid, sv_np, dFRs, dRdq, self.xyz_sum, self.vol, nx, ny, nz
        )
        # get axis derivatives
        df_ax = np.zeros(3, dtype="float")
        dqda = get_dqda(q)
        df_ax = np.dot(dqda, df_val[3:]) # 1st derivatives
        return df_ax

    def derivatives_trans(self, x):
        # translation derivatives
        nx, ny, nz = self.e0.shape
        vol = nx * ny * nz
        tpi = (2.0 * np.pi * 1j)

        ax = self.ax_init #+ x[:3]
        ax = ax / math.sqrt(np.dot(ax, ax))
        q = self.get_quaternion(axis=ax, angle=self.angle)
        q = q / np.sqrt(np.dot(q, q))
        rotmat = core.quaternions.get_RM(q)
        e1 = get_FRS(rotmat, self.e0, interp='linear')[:, :, :, 0]

        st, sv1, sv2, sv3 = fcodes_fast.get_st(nx, ny, nz, x[3:])
        ert = e1 * st
        sv = [sv1, sv2, sv3]
        binfsc, _, _ = core.fsc.anytwomaps_fsc_covariance(
            self.e0, ert, self.bin_idx, self.nbin)
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = binfsc
        wgrid = fcodes_fast.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

        df = np.zeros(6, dtype="float")
        for i in range(3):
            df[3+i] = np.sum(np.real(wgrid * np.conjugate(self.e0) * (ert * tpi * sv[i]))) / vol
        return -df

    def calc_fsc(self,e1):
        assert self.e0.shape == e1.shape == self.bin_idx.shape
        binfsc, _, bincounts = core.fsc.anytwomaps_fsc_covariance(
            self.e0, e1, self.bin_idx, self.nbin)
        self.binfsc = binfsc

    def get_wght(self, e1): 
        self.calc_fsc(e1)
        nz, ny, nx = self.e0.shape
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = self.binfsc #/ (1 - self.binfsc ** 2)
        self.wgrid = fcodes_fast.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

    #3. functional
    def functional(self, x, info):
        nx, ny, nz = self.e0.shape
        ax = self.ax_init + x[:3]
        ax = ax / math.sqrt(np.dot(ax, ax))
        q = self.get_quaternion(axis=ax, angle=self.angle)
        q = q / np.sqrt(np.dot(q, q))
        rotmat = core.quaternions.get_RM(q)
        e1 = get_FRS(rotmat, self.e0, interp='linear')[:, :, :, 0]
        t = x[3:]
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        e1 = e1 * st
        self.get_wght(e1)
        fval = np.sum(self.wgrid * self.e0 * np.conjugate(e1)) / (nx*ny*nz) # divide by vol is to scale
        #fval = np.sum(self.e0 * np.conjugate(e1)) / (nx*ny*nz)
        if info['Nfeval'] % 20 == 0:
            print('fval, axis, trans', fval.real, ax, t)
        info['Nfeval'] += 1
        return -fval.real

    #3. optimize 
    def optimize(self):
        from scipy.optimize import minimize
        x = np.array([0.0, 0.0, 0.0, 0., 0., 0.], 'float')
        options = {'maxiter': 2000}
        args=({'Nfeval':0},)
        print("Optimization method: ", self.method)
        if self.method.lower() == 'nelder-mead':
            result = minimize(fun=self.functional, x0=x, method='Nelder-Mead', tol=1e-5, options=options, args=args)
        elif self.method.lower() == 'bfgs':
            result = minimize(fun=self.functional, x0=x, method='BFGS', jac=self.derivatives_trans, tol=1e-5, options=options)
        else:
            result = minimize(fun=self.functional, x0=x, method=self.method, jac=self.derivatives, hess=self.hess_r, tol=1e-5, options=options)    
        if result.status:
            print(result)
        self.t = result.x[3:]
        ax = self.ax_init + result.x[:3]
        self.ax_final = ax / math.sqrt(np.dot(ax, ax))
        print('Final axis: ', self.ax_final)   


def getaxis(axis_ini, step_ax):
    axmax = np.argmax(abs(axis_ini))
    if axmax == 0:
        ay, az = axis_ini[1] + step_ax[1], axis_ini[2] + step_ax[2]
        ax = np.sqrt(1.0 - ay * ay - az * az)
    elif axmax == 1:
        ax, az = axis_ini[0] + step_ax[0], axis_ini[2] + step_ax[2]
        ay = np.sqrt(1.0 - ax * ax - az * az)
    elif axmax == 2:
        ax, ay = axis_ini[0] + step_ax[0], axis_ini[1] + step_ax[1]
        az = np.sqrt(1.0 - ax * ax - ay * ay)
    return np.array([ax, ay, az], dtype="float")


def fsc_between_static_and_transfomed_map(
    staticmap, movingmap, bin_idx, rm, t, cell, nbin
):
    nx, ny, nz = staticmap.shape
    st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
    frt_full = get_FRS(rm, movingmap * st, interp="linear")[:, :, :, 0]
    f1f2_fsc = core.fsc.anytwomaps_fsc_covariance(
        staticmap, frt_full, bin_idx, nbin)[0]
    return f1f2_fsc


def get_ibin(bin_fsc, thresh=0.5):
    print('thresh:', thresh)
    # new search from rear end
    for i, ifsc in reversed(list(enumerate(bin_fsc))):
        if ifsc > thresh:
            ibin = i
            if ibin % 2 != 0:
                ibin = ibin - 1
            break
    return ibin


def run_fit(
    emmap1,
    rotmat,
    t,
    ifit=0,
    fitfsc=0.5,
    nmarchingcycles=10,
    fobj=None,
    fitres=None,
):
    from emda.ext.overlay import determine_ibin

    q_init = quaternions.rot2quart(rotmat)
    axis_ang = quaternions.quart2axis(q_init)
    axis_ini = axis_ang[:3]
    angle = axis_ang[-1]
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
    for i in range(nmarchingcycles):
        if i == 0:
            f1f2_fsc = fsc_between_static_and_transfomed_map(
                staticmap=emmap1.fo_lst[0],
                movingmap=emmap1.fo_lst[0],
                bin_idx=emmap1.bin_idx,
                rm=rotmat,
                t=t,
                cell=emmap1.map_unit_cell,
                nbin=emmap1.nbin,
            )
            fsc_lst.append(f1f2_fsc)
            if fitfsc > 0.999:
                rotmat = rotmat
                final_axis = axis_ini
                print("\n***FSC between static and moving maps***\n")
                print("bin#     resolution(A)      start-FSC     end-FSC\n")
                for j in range(len(emmap1.res_arr)):
                    print(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                            j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[0][j]
                        )
                    )
                break
            ibin = determine_ibin(f1f2_fsc)
            if fitbin < ibin:
                ibin = fitbin
            ibin_old = ibin
            q = q_init
            print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
        else:
            # Apply initial rotation and translation to calculate fsc
            f1f2_fsc = fsc_between_static_and_transfomed_map(
                emmap1.fo_lst[0],
                emmap1.fo_lst[ifit],
                emmap1.bin_idx,
                rotmat,
                t,
                emmap1.map_unit_cell,
                emmap1.nbin,
            )
            ibin = get_ibin(f1f2_fsc, thresh=fitfsc)
            if fitbin < ibin:
                ibin = fitbin
            if ibin_old == ibin:
                fsc_lst.append(f1f2_fsc)
                res_arr = emmap1.res_arr[:ibin_old]
                fsc_bef = fsc_lst[0][:ibin_old]
                fsc_aft = fsc_lst[1][:ibin_old]
                print("\n***FSC between static and moving maps***\n")
                print("bin#     resolution(A)      start-FSC     end-FSC\n")
                for j in range(len(res_arr)):
                    print(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                            j, res_arr[j], fsc_bef[j], fsc_aft[j]
                        )
                    )
                print("Plotting FSCs...")
                plotter.plot_nlines(
                    res_arr=res_arr, 
                    list_arr=[fsc_lst[0][:ibin_old], fsc_lst[1][:ibin_old]], 
                    curve_label=["Proshade axis", "EMDA axis"], 
                    plot_title="FSC based on Symmetry axis", 
                    fscline=1.,
                    mapname="fsc_axis.eps")
                break
            else:
                ibin_old = ibin
                print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
        if ibin == 0:
            print("ibin = 0, Cannot proceed! Stopping current axis refinement.")
            fobj.write(
                "ibin = 0, Cannot proceed! Stopping current axis refinement.\n")
            break
        e_list = [emmap1.eo_lst[0], emmap1.fo_lst[0]]
        eout, cBIdx, cbin = cut_resolution_for_linefit(
            e_list, emmap1.bin_idx, emmap1.res_arr, ibin
        )
        static_cutmap = eout[1, :, :, :]  # use Fo instead of Eo for fitting.
        bfgs = Bfgs()
        if i == 0:
            bfgs.ax_init = np.asarray(axis_ini, 'float')
        else:
            bfgs.ax_init = current_axis
        bfgs.angle = float(np.rad2deg(angle))
        bfgs.e0 = static_cutmap
        bfgs.bin_idx = cBIdx
        bfgs.nbin = cbin
        bfgs.method = 'nelder-mead'
        bfgs.optimize()
        current_axis = bfgs.ax_final
        t = -bfgs.t
        q = quaternions.get_quaternion([list(current_axis), bfgs.angle])
        rotmat = quaternions.get_RM(q)
    final_axis = current_axis
    final_t = t
    return final_axis, final_t


def axis_refine(
    imap,
    rotaxis,
    symorder,
    fitfsc=0.5,
    ncycles=10,
    t_init=[0.0, 0.0, 0.0],
    interp="linear",
    imask=None,
    fobj=None,
    fitres=6,
):
    axis = np.asarray(rotaxis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    print("Initial axis and fold: ", axis, symorder)
    if fobj is None:
        fobj = open("EMDA_symref.txt", "w")
    fobj.write("Initial axis and fold: " + str(axis) + str(symorder) + "\n")
    print(" ")
    print("Preparing data for axis refinement...")
    try:
        emmap1 = EmmapOverlay(imap=imap, imask=imask)
    except:
        emmap1 = EmmapOverlay(imap=imap)
    emmap1.get_maps()
    emmap1.calc_fsc_from_maps()
    print(" Number of refinement cycles:", ncycles)
    print("Data resolution for refinement: ", fitres)
    fobj.write(" Number of refinement cycles: " + str(ncycles) + "\n")
    fobj.write("Data resolution for refinement: " + str(fitres) + "\n")
    print("Initial axis and angles:")
    fobj.write("Initial axis and angles: \n")

    angle = float(360.0 / symorder)
    print("   ", axis, angle)
    fobj.write("   " + str(axis) + str(angle) + "\n")
    q = quaternions.get_quaternion([list(axis[::-1]), [angle]])
    rotmat_init = quaternions.get_RM(q)
    frt = get_FRS(rotmat_init, emmap1.fo_lst[0], interp="linear")[:, :, :, 0]
    data2write = np.real(ifftshift(ifftn(ifftshift(frt))))
    em.write_mrc(data2write, "ax_initial_map.mrc", emmap1.map_unit_cell)
    final_axis, final_tran = run_fit(
        emmap1=emmap1,
        rotmat=rotmat_init,
        t=np.asarray(t_init, dtype="float"),
        fitres=fitres,
        fobj=fobj,
        fitfsc=fitfsc,
    )
    # output maps
    q = quaternions.get_quaternion([list(final_axis), angle])
    rotmat = quaternions.get_RM(q)
    nx, ny, nz = emmap1.map_dim
    st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, final_tran)
    frt_full = get_FRS(rotmat, emmap1.fo_lst[0] * st, interp=interp)[:, :, :, 0]
    data2write = np.real(ifftshift(ifftn(ifftshift(frt_full))))
    em.write_mrc(data2write, "ax_refined_map.mrc", emmap1.map_unit_cell)
    data2write = np.real(ifftshift(ifftn(ifftshift(emmap1.fo_lst[0]))))
    em.write_mrc(data2write, "static_map.mrc", emmap1.map_unit_cell)
    #return final_axis[::-1], final_tran[::-1]
    return final_axis, final_tran


def map_output(maplist, imask, axis, angle, translation):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    q = quaternions.get_quaternion([list(axis), angle])
    rotmat = quaternions.get_RM(q)
    arr1 = arr2 = None
    if imask is not None:
        _, mask, _ = em.get_data(imask)
        mask = mask * mask > 1e-4
    else:
        mask = 1.
    if len(maplist) == 2:
        uc, arr1, orig = em.get_data(maplist[0])
        uc, arr2, orig = em.get_data(maplist[1])
        arr1 = set_dim_even(arr1 * mask)
        arr2 = set_dim_even(arr2 * mask)
        arr = (arr1 + arr2)
    elif len(maplist) == 1:
        uc, arr, orig = em.get_data(maplist[0])
        arr = set_dim_even(arr * mask)
    
    com = em.center_of_mass_density(arr)
    print('com: ', com)
    nx, ny, nz = arr.shape
    box_centr = (nx // 2, ny // 2, nz // 2)
    arr = em.shift_density(arr, np.subtract(box_centr, com))
    em.write_mrc(arr, "full_map.mrc", uc)
    fo = fftshift(fftn(fftshift(arr)))
    st = fcodes_fast.get_st(nx, ny, nz, translation)[0]
    frt = get_FRS(rotmat, fo * st, interp="linear")[:, :, :, 0]
    data2write = np.real(ifftshift(ifftn(ifftshift(frt))))
    em.write_mrc(data2write, "transformed_fullmap.mrc", uc)
    if arr1 is not None:
        arr1 = em.shift_density(arr1, np.subtract(box_centr, com))
        arr2 = em.shift_density(arr2, np.subtract(box_centr, com))
        em.write_mrc(arr1, "halfmap1.mrc", uc)
        em.write_mrc(arr2, "halfmap2.mrc", uc)
        fo1 = fftshift(fftn(fftshift(arr1)))
        fo2 = fftshift(fftn(fftshift(arr2)))
        st = fcodes_fast.get_st(nx, ny, nz, translation)[0]
        frt1 = get_FRS(rotmat, fo1 * st, interp="linear")[:, :, :, 0]
        frt2 = get_FRS(rotmat, fo2 * st, interp="linear")[:, :, :, 0]
        data2write1 = np.real(ifftshift(ifftn(ifftshift(frt1))))
        data2write2 = np.real(ifftshift(ifftn(ifftshift(frt2))))
        em.write_mrc(data2write1, "transformed_halfmap1.mrc", uc)      
        em.write_mrc(data2write2, "transformed_halfmap2.mrc", uc)    





if __name__ == "__main__":
    #imap = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/lig_full.mrc"
    #imask = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/halfmap_mask.mrc"
    imap =  "/Users/ranganaw/MRC/REFMAC/EMD-0011/emd_0011_emda_reboxed.mrc"
    imask = "/Users/ranganaw/MRC/REFMAC/EMD-0011/emda_reboxedmask.mrc"
    #rotaxis = [0.99980629, -0.00241615,  0.01953302] # 2-fold axis
    rotaxis = [+0.000,    +0.000,    +1.000] # 3-fold axis
    symorder = 3
    ax_final, t_final = axis_refine(
            imap=imap,
            imask=imask,
            rotaxis=rotaxis, #[0.20726902, 0.97784544, 0.02928904],
            symorder=symorder
        )
    angle = float(360/symorder)
    map_output([imap], imask, ax_final, angle, t_final)
    #hf1 = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/lig_hf1.mrc"
    #hf2 = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/lig_hf2.mrc"
    #map_output([hf1, hf2], imask, ax_final, 180.0, t_final)