from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import fcodes
from emda.mapfit.quaternions import get_RM, derivatives_wrt_q,rotationMatrixToEulerAngles
from emda.plotter import plot_nlines
from timeit import default_timer as timer
from emda.config import *

#debug_mode = 0 # 0: no debug info, 1: debug

def get_FRS(uc,RM,E2):
    if len(E2.shape) == 3:
        E2 = np.expand_dims(E2, axis=3)
    ERS = get_interp(uc,RM,E2)
    return ERS

def get_interp(uc,RM,data):
    import fcodes
    assert len(data.shape) == 4
    ih,ik,il,n = data.shape
    interp3d = fcodes.tricubic(RM,data,debug_mode,n,[ih,ik,il])
    return interp3d

def make_xyz(nx,ny,nz):
    x = np.zeros(shape=(nx,ny,nz),dtype='float')
    y = np.zeros(shape=(nx,ny,nz),dtype='float')
    z = np.zeros(shape=(nx,ny,nz),dtype='float')
    mnx = -nx//2; mxx = -(mnx + 1)
    mny = -ny//2; mxy = -(mny + 1)
    mnz = -nz//2; mxz = -(mnz + 1)
    print(mnx,mxx)
    for h in range(mnx, mxx):
        for k in range(mny, mxy):
            for l in range(mnz, mxz):
                x[h,k,l] = h/nx
                y[h,k,l] = k/ny
                z[h,k,l] = l/nz
    print(h,k,l)
    return np.array([x,y,z])

def dFs(mapin):
    nx,ny,nz = mapin.shape
    print(mapin.shape)
    xyz = make_xyz(nx,ny,nz)
    tp2 = (2.0 * np.pi)**2 
    n = -1
    dfs_arr = np.zeros(shape=(nx,ny,nz,3),dtype='complex')
    ddfs_arr = np.zeros(shape=(nx,ny,nz,6),dtype='complex')
    print(mapin.shape)
    for i in range(3):
        xyz[i] = np.fft.ifftshift(xyz[i])
    # Calculating dFC/ds using FFT - 1st derivative using eq. 15
    for i in range(3):  
        dfs = -2.0*1j * np.pi * np.fft.fftshift(np.fft.fftn(mapin * xyz[i]))
        dfs_arr[:,:,:,i] = dfs
        # Calculating ddFC/ds using FFT - 2nd derivative using eq. 17
        for j in range(3):
            if i == 0:
                ddfs = -1.0 * tp2 * np.fft.fftshift(np.fft.fftn(mapin * xyz[i] * xyz[j]))
            elif i > 0 and j >= i:
                ddfs = -1.0 * tp2 * np.fft.fftshift(np.fft.fftn(mapin * xyz[i] * xyz[j]))
            else:
                continue
            n = n + 1
            ddfs_arr[:,:,:,n] = ddfs     
    return dfs_arr,ddfs_arr

def make_xyz2(nx,ny,nz):
    x = np.zeros(shape=(nx,ny,nz),dtype='float')
    y = np.zeros(shape=(nx,ny,nz),dtype='float')
    z = np.zeros(shape=(nx,ny,nz),dtype='float')

    for h in range(nx):
        for k in range(ny):
            for l in range(nz):
                x[h,k,l] = h/nx
                y[h,k,l] = k/ny
                z[h,k,l] = l/nz
    return np.array([x,y,z])

def dFs2(mapin,xyz,vol):
    import numpy as np
    nx,ny,nz = mapin.shape
    start = timer()
    dfs_arr = np.zeros(shape=(nx,ny,nz,3),dtype='complex')
    for i in range(3):  
        dfs = (1/vol) * 2j * np.pi * np.fft.fftn(mapin * xyz[i])
        dfs_arr[:,:,:,i] = np.fft.fftshift(dfs)
    end = timer() 
    #print('time for dfs and ddfs calculation: ', end-start)
    return dfs_arr

def create_xyz_grid(uc,nxyz):
    x = np.fft.fftfreq(nxyz[0]) * uc[0]
    y = np.fft.fftfreq(nxyz[1]) * uc[1]
    z = np.fft.fftfreq(nxyz[2]) * uc[2]
    xv, yv, zv = np.meshgrid(x,y,z)
    xyz = [yv,xv,zv]
    for i in range(3):
        xyz[i] = np.fft.ifftshift(xyz[i])
    return xyz

def get_xyz_sum(xyz):
    xyz_sum = np.zeros(shape=(6),dtype='float')
    n = -1
    for i in range(3):  
        for j in range(3):
            if i == 0:
                sumxyz = np.sum(xyz[i] * xyz[j])
            elif i > 0 and j >= i:
                sumxyz = np.sum(xyz[i] * xyz[j])
            else:
                continue
            n = n + 1
            xyz_sum[n] = sumxyz   
    print(xyz_sum)
    return xyz_sum


def avg_and_diffmaps(maps2avg,uc,nbin,sgnl_var,hffsc,bin_idx,res_arr,Bf_arr):
    # average and difference map calculation
    import numpy as np
    import numpy.ma as ma
    import fcodes
    nx,ny,nz = maps2avg[0].shape
    nmaps = len(maps2avg)
    unit_cell = uc
    #all_maps = maps2avg.asarray(copy=False)
    all_maps = np.zeros(shape=(nx,ny,nz,nmaps),dtype='complex')
    for i in range(nmaps):
        all_maps[:,:,:,i] = maps2avg[i]
    print(all_maps.shape)
    #
    S_mat = np.zeros(shape=(nmaps,nmaps,nbin),dtype='float')
    T_mat = np.zeros(shape=(nmaps,nmaps,nbin),dtype='float')
    F_mat = np.zeros(shape=(nmaps,nmaps,nbin),dtype='float')
    # Populate Sigma matrix
    start = timer()
    for i in range(nmaps):
        for j in range(nmaps):
            if i == j:
                print('i,j=',i,j)
                # Diagonals
                S_mat[i,i,:] = 1.0
                T_mat[i,i,:] = 1.0
                F_mat[i,i,:] = np.sqrt(ma.masked_less_equal(hffsc[i],0.0).filled(0.0))
            elif j > i:
                print('i,j=',i,j)
                # Off diagonals
                f1f2_covar,f1f2_fsc = fcodes.calc_covar_and_fsc_betwn_anytwomaps(
                    maps2avg[i],maps2avg[j],bin_idx,nbin,debug_mode,nx,ny,nz)
                # To treat the division-by-zero error with masked arrays
                Sigma_ij_ma = ma.masked_less_equal(f1f2_covar,0.01)
                sgnl_var_ij_ma = ma.masked_less_equal(sgnl_var[i] * sgnl_var[j], 0.0)
                #print(sgnl_var_ij_ma)
                #print(np.sqrt(sgnl_var_ij_ma))
                S_mat[i,j,:] = (Sigma_ij_ma / np.sqrt(sgnl_var_ij_ma)).filled(0.0)
                T_mat[i,j,:] = f1f2_fsc # may need to mask
            else:
                print('i,j=',i,j)
                S_mat[i,j,:] = S_mat[j,i,:]
                T_mat[i,j,:] = T_mat[j,i,:]
    # Plotting
    plot_nlines(res_arr,[S_mat[0,0,:],S_mat[0,1,:],S_mat[1,0,:],S_mat[1,1,:]],'S_mat_fsc_ij.eps',["FSC11","FSC12","FSC21","FSC22"])
    plot_nlines(res_arr,[F_mat[0,0,:],F_mat[1,1,:]],'F_mat_fsc_ij.eps',["sqrt(FSC11)","sqrt(FSC22)"])
    plot_nlines(res_arr,[T_mat[0,0,:],T_mat[0,1,:],T_mat[1,0,:],T_mat[1,1,:]],'T_mat_fsc_ij.eps',["FSC11","FSC12","FSC21","FSC22"])

    end = timer()
    print('Time for sigma matrix population: ', end-start)
    start = timer()
    # Variance weighted matrices calculation
    Wgt = np.zeros(shape=(nmaps,nmaps,nbin))
    for ibin in range(nbin):
        T_mat_inv = np.linalg.pinv(T_mat[:,:,ibin]) # Moore-Penrose psedo-inversion
        tmp = np.dot(F_mat[:,:,ibin],T_mat_inv)
        Wgt[:,:,ibin] = np.dot(S_mat[:,:,ibin],tmp)
    end = timer()
    print('time for pinv  ',end-start)
    # Average map calculation
    #Bf_arr = np.array([-50,50],dtype='float')
    nbf = len(Bf_arr)
    assert all_maps.shape[:3] == bin_idx.shape
    assert all_maps.shape[3] == nmaps
    AVG_Maps = fcodes.calc_avg_maps(all_maps,bin_idx,Wgt,Bf_arr,unit_cell,debug_mode,nbin,nmaps,nbf,nx,ny,nz)
    return AVG_Maps

def remove_unwanted_corners(uc,target_dim):
    from emda.restools import get_resArr,remove_edge
    nx,ny,nz = target_dim
    fResArr = get_resArr(uc,nx)
    cut_mask = remove_edge(fResArr,fResArr[-1])
    cc_mask = np.zeros(shape=(nx,ny,nz),dtype='int')
    cx, cy, cz = cut_mask.shape
    dx = (nx - cx)//2
    dy = (ny - cy)//2
    dz = (nz - cz)//2
    #print(dx,dy,dz)
    cc_mask[dx:dx+cx, dy:dy+cy, dz:dz+cz] = cut_mask
    return cc_mask

def make_data4fit(e0,e1,bin_idx,res_arr,nbin):
    from emda.fsc import anytwomaps_fsc_covariance
    from emda.restools import cut_resolution
    _,fsc = anytwomaps_fsc_covariance(e0,e1,bin_idx,nbin)
    nx, ny, nz = e0.shape
    dist = np.sqrt((fsc - 0.1)**2)
    smax = res_arr[np.argmin(dist)]
    cbin = np.argmin(dist) + 1 # adding 1 because fResArr starts with zero
    if cbin%2 != 0: cx = cbin + 1
    else: cx = cbin
    print('cx = ', cx, 'cnbin=', cbin)
    dx = int((nx - 2*cx)/2)
    dy = int((ny - 2*cx)/2)
    dz = int((nz - 2*cx)/2)
    cBIdx = bin_idx[dx:dx+2*cx,dy:dy+2*cx,dz:dz+2*cx]
    e0 = cut_resolution(e0,bin_idx,res_arr,smax)[dx:dx+2*cx,dy:dy+2*cx,dz:dz+2*cx]
    e1 = cut_resolution(e1,bin_idx,res_arr,smax)[dx:dx+2*cx,dy:dy+2*cx,dz:dz+2*cx]
    return e0, e1, cBIdx


