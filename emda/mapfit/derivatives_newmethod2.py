"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""


from __future__ import absolute_import, division, print_function, unicode_literals

from timeit import default_timer as timer
import numpy as np
from emda.mapfit.quaternions import derivatives_wrt_q
from emda.mapfit.utils import dFs2
from emda.config import *

#debug_mode = 0

def derivatives(e0,e1,w_grid,w2_grid,q,sv,xyz,xyz_sum,vol):
    import numpy as np
    df_val = np.zeros(6,dtype='float')
    ddf_val = np.zeros(shape=(6,6),dtype='float')
    tp2 = (2.0 * np.pi)**2
    # DERIVATIVES W.R.T. TRANSLATION
    start = timer()
    for i in range(3):
        # 1st derivative
        df_tmp = np.sum(w_grid * e0 * 
                np.conjugate(e1 * (2.0 * np.pi * 1j) * sv[i])) 
        df_val[i] = np.real(df_tmp)
        for j in range(3):
            # estimated 2nd derivative
            if i == 0:
                ddf_tmp = -tp2 * np.sum(w2_grid * sv[i] * sv[j]) 
                #ddf_tmp = -tp2 * np.sum(w_grid * e0 * 
                #           np.conjugate(e1 * sv[i] * sv[j]))
            elif i > 0 and j >= i:
                ddf_tmp = -tp2 * np.sum(w2_grid * sv[i] * sv[j])
            else:
                ddf_val[i,j] = ddf_val[j,i]
            ddf_val[i,j] = np.real(ddf_tmp)
    
    # DERIVATIVES W.R.T. ROTATION
    dRdq = derivatives_wrt_q(q)
    # 1st derivatives of the rotated map w.r.t. s
    dFRs = dFs2(np.real(np.fft.ifftn(np.fft.ifftshift(e1))),xyz,vol)
    a = np.zeros(shape=(3,3),dtype='float')
    b = np.zeros(shape=(3,3),dtype='float')
    for i in range(3):
        a[:,:] = 0.0
        for k in range(3):
            for l in range(3):
                if k == 0:
                    tmp1 = np.sum(w_grid * np.conjugate(e0) * 
                            (dFRs[:,:,:,k] * sv[l] * dRdq[i,k,l]))
                elif k > 0 and l >= k:
                    tmp1 = np.sum(w_grid * np.conjugate(e0) * 
                            (dFRs[:,:,:,k] * sv[l] * dRdq[i,k,l]))
                else:
                    a[k,l] = a[l,k]
                a[k,l] = tmp1.real
        df_val[i+3] = np.sum(a) # df_val[3] to df_val[5]
        
    wfsc = w_grid * np.conjugate(e0) * e1 # THIS WORKS
    for i in range(3):
        for j in range(3):
            if i == 0:
                b[:,:] = 0.0
                n = -1
                for k in range(3):
                    for l in range(3): 
                        if k == 0:
                            n = n + 1
                            tmp2 = -(tp2/vol) * xyz_sum[n] * np.sum(wfsc * 
                                    sv[k] * sv[l] * dRdq[i,k,l] * dRdq[j,k,l])
                        elif k > 0 and l >= k:
                            n = n + 1
                            tmp2 = -(tp2/vol) * xyz_sum[n] * np.sum(wfsc *
                                    sv[k] * sv[l] * dRdq[i,k,l] * dRdq[j,k,l])
                        else:
                            b[k,l] = b[l,k]
                        b[k,l] = tmp2.real
                ddf_val[i+3,j+3] = np.sum(b) # ddf_val[3] to [5]
            elif i > 0 and j >= i:
                b[:,:] = 0.0
                n = -1
                for k in range(3):
                    for l in range(3): 
                        if k == 0:
                            n = n + 1
                            tmp2 = -(tp2/vol) * xyz_sum[n] * np.sum(wfsc * 
                                    sv[k] * sv[l] * dRdq[i,k,l] * dRdq[j,k,l])
                        elif k > 0 and l >= k:
                            n = n + 1
                            tmp2 = -(tp2/vol) * xyz_sum[n] * np.sum(wfsc *
                                    sv[k] * sv[l] * dRdq[i,k,l] * dRdq[j,k,l])
                        else:
                            b[k,l] = b[l,k]
                        b[k,l] = tmp2.real
                ddf_val[i+3,j+3] = np.sum(b) # ddf_val[3] to [5]
            else:
                ddf_val[i+3,j+3] = ddf_val[j+3,i+3]    
    #print(ddf_val)
    # Mixed derivatives
    # NEED A REVIEW
    #for i in range(3):
    #    for j in range(3):
    #        a[:,:] = 0.0
    #        for k in range(3):
    #            for l in range(3):
    #                tmp1 = np.sum(-w_grid * np.conjugate(e0) *
    #                    (dFRs[:,:,:,k] * sv[l] * sv[j] * st * (2.0 * np.pi * 1j) * dRdq[j,k,l]))
    #                a[k,l] = tmp1.real
    #                #print(tmp1)
    #        ddf_val[i,j+3] = np.sum(a) 
    #        ddf_val[i+3,j] = np.sum(a)
    #print(np.linalg.det(ddf_val))
    end = timer()
    #print('time for derivative calculation: ', end-start)
    ddf_val_inv = np.linalg.pinv(ddf_val)
    step = ddf_val_inv.dot(-df_val)
    return step,df_val,e1

def get_FRS(uc,RM,E2):
    if len(E2.shape) == 3:
        E2 = np.expand_dims(E2, axis=3)
    ERS = get_interp(uc,RM,E2)
    return ERS

def get_interp(uc,RM,data):
    import fcodes
    assert len(data.shape) == 4
    ih,ik,il,n = data.shape
    interp3d = fcodes.tricubic(RM,data,debug_mode,n,ih,ik,il)
    return interp3d
