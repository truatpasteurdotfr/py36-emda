"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from emda.core.quaternions import derivatives_wrt_q
from timeit import default_timer as timer


timeit = False


def new_dFs2(mapin, xyz, vol):
    nx, ny, nz = mapin.shape
    dfs = np.zeros(shape=(nx, ny, nz, 3), dtype=np.complex64)
    for i in range(3):
        dfs[:, :, :, i] = np.fft.fftshift(
            (1 / vol) * 2j * np.pi * np.fft.fftn(mapin * xyz[i])
        )
    return dfs


def new_derivatives(e0, e1, w_grid, w2_grid, q, sv, xyz, xyz_sum, vol, dfrs=None):
    import sys
    import fcodes_fast

    nx, ny, nz = e0.shape
    start = timer()
    sv_np = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    for i in range(3):
        sv_np[:, :, :, i] = sv[i]
    dRdq = derivatives_wrt_q(q)
    if dfrs is None:
        # print('Calculating derivatives...')
        start = timer()
        dFRs = new_dFs2(np.real(np.fft.ifftn(np.fft.ifftshift(e1))), xyz, vol)
        end = timer()
        if timeit:
            print(" time for dFRs calculation: ", end - start)
    if dfrs is not None:
        print("Interpolated DFRs are used!")
        dFRs = dfrs
    start = timer()
    df_val, ddf_val = fcodes_fast.calc_derivatives(
        e0, e1, w_grid, w2_grid, sv_np, dFRs, dRdq, xyz_sum, vol, nx, ny, nz
    )
    end = timer()
    if timeit:
        print(" time for derivative calculation: ", end - start)
    if np.linalg.cond(ddf_val) < 1 / sys.float_info.epsilon:
        ddf_val_inv = np.linalg.pinv(ddf_val)
    else:
        print("Derivative matrix is non-invertible! Stopping now...")
        exit()
    step = ddf_val_inv.dot(-df_val)
    return step, df_val

