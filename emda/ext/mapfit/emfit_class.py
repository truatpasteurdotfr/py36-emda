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
from emda import core
from emda.ext.mapfit.utils import get_FRS, create_xyz_grid, get_xyz_sum

np.set_printoptions(suppress=True)  # Suppress insignificant values for clarity

timeit = False


class EmFit:
    def __init__(self, mapobj, interp="linear", dfs=None):
        self.mapobj = mapobj
        self.cut_dim = mapobj.cdim
        self.ful_dim = mapobj.map_dim
        self.cell = mapobj.map_unit_cell
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
        self.rotmat = None
        self.t_accum = None
        self.ert = None
        self.frt = None
        self.e0 = None
        self.e1 = None
        self.w2_grid = None
        self.fsc_lst = []

    def get_wght(self, e0, ert):
        cx, cy, cz = e0.shape
        start = timer()
        fsc, _ = core.fsc.anytwomaps_fsc_covariance(
            e0, ert, self.mapobj.cbin_idx, self.mapobj.cbin
        )
        w_grid = fcodes_fast.read_into_grid(
            self.mapobj.cbin_idx, fsc / (1 - fsc ** 2), self.mapobj.cbin, cx, cy, cz
        )
        fsc_sqd = fsc ** 2
        fsc_combi = fsc_sqd / (1 - fsc_sqd)
        w2_grid = fcodes_fast.read_into_grid(
            self.mapobj.cbin_idx, fsc_combi, self.mapobj.cbin, cx, cy, cz
        )
        end = timer()
        if timeit:
            print("     weight calc time: ", end - start)
        return w_grid, w2_grid, fsc

    def functional(self, e0, e1, f1=None):
        start = timer()
        cx, cy, cz = e0.shape
        self.st, s1, s2, s3 = fcodes_fast.get_st(cx, cy, cz, self.t)
        self.sv = np.array([s1, s2, s3])
        self.ert = get_FRS(self.rotmat, e1 * self.st, interp=self.interp)[:, :, :, 0]
        if f1 is not None:
            # translate and then rotate
            self.frt = get_FRS(self.rotmat, f1 * self.st, interp=self.interp)[
                :, :, :, 0
            ]
        self.w_grid, self.w2_grid, self.fsc = self.get_wght(e0, self.ert)
        fval = np.sum(self.w_grid * e0 * np.conjugate(self.ert))
        end = timer()
        if timeit:
            print(" functional calc time: ", end - start)
        return fval.real

    def minimizer(self, ncycles, t_init, rotmat, smax_lf, fobj=None):
        import math
        from emda.ext.mapfit import (
            rotmat2quart,
            linefit_class,
            interp_derivatives,
            derivatives,
        )

        fsc_lst = []
        nfit = len(self.mapobj.ceo_lst) - 1
        self.e0 = self.mapobj.ceo_lst[0]  # Static map e-data for fit
        fobj.write("\n")
        fobj.write("Normalized Structure Factors are used for fitting! \n")
        xyz = create_xyz_grid(self.cell, self.cut_dim)
        vol = self.cell[0] * self.cell[1] * self.cell[2]
        xyz_sum = get_xyz_sum(xyz)
        q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        fobj.write("\n")
        fobj.write("Cycle#\n")
        fobj.write("Function value\n")
        fobj.write("Rotation(degrees)\n")
        fobj.write("Translation(A)\n")
        fobj.write("\n")
        print("Cycle#   ", "Func. value  ", "Rotation(degrees)  ", "Translation(A)  ")
        for ifit in range(nfit):
            self.e1 = self.mapobj.ceo_lst[ifit + 1]
            for i in range(ncycles):
                start = timer()
                if i == 0:
                    self.t = np.asarray(t_init)
                    t_accum = self.t
                    t_accum_angstrom = t_accum * self.cell[:3]
                    translation_vec = np.sqrt(
                        np.sum(t_accum_angstrom * t_accum_angstrom)
                    )
                    self.rotmat = rotmat
                    q = rotmat2quart.rot2quart(self.rotmat)
                    self.q = q
                    q_accum = self.q
                    theta2 = np.arccos((np.trace(self.rotmat) - 1) / 2) * 180.0 / np.pi
                    t_accum_previous = t_accum
                    q_accum_previous = q_accum
                else:
                    rm_accum = core.quaternions.get_RM(q_accum)
                    theta2 = np.arccos((np.trace(rm_accum) - 1) / 2) * 180.0 / np.pi
                    if theta2 < 0.01:
                        self.rotmat = np.identity(3)
                    else:
                        self.rotmat = core.quaternions.get_RM(self.q)

                fval = self.functional(self.e0, self.e1)
                print("fval: ", fval)
                if math.isnan(theta2):
                    print("Cannot find a solution! Stopping now...")
                    exit()
                if i == 0:
                    fval_previous = fval
                    fsc = self.fsc
                    fsc_lst.append(fsc)
                if i > 0 and fval_previous < fval or i == ncycles - 1:
                    fsc = self.fsc
                    print(i, fval_previous, fval)
                    print(
                        "average FSC: before after", np.mean(fsc_lst[0]), np.mean(fsc)
                    )
                if i > 0 and fval < fval_previous or i == ncycles - 1:
                    rotmat = core.quaternions.get_RM(q_accum_previous)
                    fsc_lst.append(fsc)
                    self.rotmat = rotmat  # final rotation
                    self.t_accum = t_accum_previous  # final translation
                    self.fsc_lst = fsc_lst
                    t_accum_angstrom = self.t_accum * self.cell[:3]
                    translation_vec = np.sqrt(
                        np.sum(t_accum_angstrom * t_accum_angstrom)
                    )
                    theta2 = np.arccos((np.trace(self.rotmat) - 1) / 2) * 180.0 / np.pi
                    print("thets2, trans: ", theta2, translation_vec)
                    break
                # print('Euler angles: [degrees]: ', Euler_angles)
                print(
                    "{:5d} {:8.4f} {:6.2f} {:6.2f}".format(
                        i, fval, theta2, translation_vec
                    )
                )
                fobj.write(
                    "{:5d} {:8.4f} {:6.2f} {:6.2f}\n".format(
                        i, fval, theta2, translation_vec
                    )
                )
                if self.dfs is not None:
                    start1 = timer()
                    dFRs = interp_derivatives.interp_derivatives(self.dfs, self.rotmat)
                    end1 = timer()
                    if timeit:
                        print(" time for dFRs calculation: ", end1 - start1)
                elif self.dfs is None:
                    dFRs = None
                t_accum_previous = t_accum
                q_accum_previous = q_accum
                self.step, _ = derivatives.new_derivatives(
                    self.e0,
                    self.ert,
                    self.w_grid,
                    self.w2_grid,
                    q,
                    self.sv,
                    xyz,
                    xyz_sum,
                    vol,
                    dFRs,
                )
                self.e1 = self.ert
                start_lf = timer()
                if i == 0:
                    linefit = linefit_class.LineFit()
                    linefit.get_linefit_static_data(
                        self.e0, self.mapobj.cbin_idx, self.mapobj.res_arr, smax_lf
                    )
                linefit.step = self.step
                alpha = linefit.calc_fval_for_different_kvalues_at_this_step(self.e1)
                end_lf = timer()
                if timeit:
                    print(" time for line fit: ", end_lf - start_lf)
                # translation
                self.t = self.step[:3] * alpha[0]
                t_accum = t_accum + self.t
                t_accum_angstrom = t_accum * self.cell[:3]
                translation_vec = np.sqrt(np.sum(t_accum_angstrom * t_accum_angstrom))
                # rotation
                tmp = np.insert(self.step[3:] * alpha[1], 0, 0.0)
                q_accum = q_accum + tmp
                q_accum = q_accum / np.sqrt(np.dot(q_accum, q_accum))
                rm_accum = core.quaternions.get_RM(q_accum)
                theta2 = np.arccos((np.trace(rm_accum) - 1) / 2) * 180.0 / np.pi
                if theta2 < 0.01:
                    tmp = q_init  # revert to no-rotation
                else:
                    tmp = tmp + q_init
                q = tmp / np.sqrt(np.dot(tmp, tmp))
                self.q = q
                fval_previous = fval
                end = timer()
                if timeit:
                    print("time for one cycle:", end - start)

