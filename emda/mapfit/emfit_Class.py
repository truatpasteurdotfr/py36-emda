"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import,division,print_function,unicode_literals
import numpy as np
from timeit import default_timer as timer
from emda.plotter import *
from emda.mapfit.utils import dFs2,get_FRS,get_interp,create_xyz_grid,get_xyz_sum,make_data4fit
import fcodes
from emda.mapfit.quaternions import *
from emda.restools import cut_resolution_for_linefit
from emda.mapfit.derivatives_newmethod2 import derivatives
from emda.fsc import *
from emda.config import *

#debug_mode = 0 # 0: no debug info, 1: debug

class emFit:
    def __init__(self,mapobj):
        self.mapobj         = mapobj
        self.cut_dim        = mapobj.cdim
        self.ful_dim        = mapobj.map_dim
        self.cell           = mapobj.map_unit_cell
        self.origin         = mapobj.map_origin
        self.w_grid         = None
        self.fsc            = None 
        self.sv             = None
        self.t              = None
        self.st             = None
        self.step           = None
        self.rotmat         = None
        self.frt_full_lst   = None
        self.ert_full_lst   = None
        self.rt_hf_lst      = None

    def get_wght(self,e0,ert):
        cx,cy,cz = e0.shape
        start = timer()
        _,fsc = anytwomaps_fsc_covariance(e0, ert,
                                            self.mapobj.cbin_idx,
                                            self.mapobj.cbin)
        w_grid = fcodes.read_into_grid(self.mapobj.cbin_idx,
                                fsc/(1-fsc**2),
                                self.mapobj.cbin,
                                cx,cy,cz)
        fsc_sqd = fsc**2
        fsc_combi = fsc_sqd/(1 - fsc_sqd)
        w2_grid = fcodes.read_into_grid(self.mapobj.cbin_idx,
                                fsc_combi,
                                self.mapobj.cbin,cx,cy,cz)
        end = timer()
        #print('weight calc time: ', end-start)
        return w_grid,w2_grid,fsc

    def functional(self,e0,e1):
        start = timer()
        cx,cy,cz = e0.shape
        self.st,s1,s2,s3 = fcodes.get_st(cx,cy,cz,self.t)
        #self.st = np.transpose(st_transpose)
        self.sv = np.array([s1,s2,s3])
        self.ert = get_FRS(self.cell,self.rotmat,e1 * self.st)[:,:,:,0]
        # translate and then rotate
        self.w_grid,self.w2_grid,self.fsc = self.get_wght(e0, self.ert) 
        fval = np.sum(self.w_grid * e0 * np.conjugate(self.ert))
        end = timer()
        #print('functional calc time: ', end-start)
        return fval.real

    def f2(self,k):
        # w = 1.0 for line search
        nx,ny,nz = self.e0_lf.shape
        t = self.step[:3]*k[0]
        st,_,_,_ = fcodes.get_st(nx,ny,nz,t)
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        tmp = np.insert(self.step[3:]*k[1], 0, 0.0)
        tmp = tmp + q_init
        q = tmp/np.sqrt(np.dot(tmp, tmp))
        rotmat = get_RM(q)
        ers = get_FRS(self.cell,rotmat,self.e1_lf * st)
        fval = np.sum(self.e0_lf * np.conjugate(ers[:,:,:,0]))
        return -fval.real

    def calc_fval_for_different_kvalues_at_this_step(self,step,e0,e1):
        start = timer()
        smax = 15 # cut resolution in Angstrom
        self.e0_lf = cut_resolution_for_linefit(self.e0,self.mapobj.cbin_idx,
                                                self.mapobj.res_arr,smax)
        self.e1_lf = cut_resolution_for_linefit(self.e1,self.mapobj.cbin_idx,
                                                self.mapobj.res_arr,smax)
        from scipy import optimize
        init_guess = [1.0,1.0]
        minimum = optimize.minimize(self.f2, init_guess, method='Powell')
        end = timer()
        #print('time for line search: ', end-start)
        return minimum.x

    def minimizer(self,ncycles,t_init,theta_init):
        #fval_old         = 0.0
        fsc_lst         = []
        fval_lst        = []
        theta2_lst      = []
        trans_lst       = []
        frt_full_lst    = [] # rotated and translated F2 full maps
        ert_full_lst    = []
        rt_hf_lst       = [] # rotated and translated F2 half maps

        # rotated and translated half maps (static map)
        rt_hf_lst.append(self.mapobj.fhf_lst[0])
        rt_hf_lst.append(self.mapobj.fhf_lst[1])
        # rotated and translated full maps (static map)
        frt_full_lst.append(self.mapobj.fo_lst[0])
        ert_full_lst.append(self.mapobj.eo_lst[0])
        #
        nfit = len(self.mapobj.ceo_lst) - 1
        self.e0 = self.mapobj.ceo_lst[0] # Static map e-data for fit
        xyz = create_xyz_grid(self.cell, self.cut_dim)
        xyz_sum = get_xyz_sum(xyz)
        vol = self.cell[0] * self.cell[1] * self.cell[2]
        start = timer()
        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        for ifit in range(nfit):
            self.e1 = self.mapobj.ceo_lst[ifit + 1]
            for i in range(ncycles):
                if i == 0:
                    '''self.e0,self.e1,self.cbin_idx = make_data4fit(
                                                    self.mapobj.eo_lst[0],
                                                    self.mapobj.eo_lst[ifit + 1],
                                                    self.mapobj.bin_idx,
                                                    self.mapobj.res_arr,
                                                    len(self.mapobj.res_arr))'''
                    self.t = np.asarray(t_init)
                    t_accum = self.t
                    translation_vec = np.sqrt(np.sum(self.t * self.t))
                    q = get_quaternion(np.pi*theta_init/180.0)
                    q = q/np.sqrt(np.dot(q,q))
                    self.q = q
                    q_accum = q
                    self.rotmat = get_RM(q)
                    theta2 = np.arccos((np.trace(self.rotmat) - 1)/2) * 180./np.pi
                else:
                    self.rotmat = get_RM(q)
                    rm_accum = get_RM(q_accum)
                    theta2 = np.arccos((np.trace(rm_accum) - 1)/2) * 180./np.pi

                fval = self.functional(self.e0, self.e1)
                fval_lst.append(fval) 
                theta2_lst.append(theta2)
                trans_lst.append(translation_vec)
                print(i,fval,theta2,translation_vec)#,q,self.t)

                if i == 0 or i == ncycles-1: 
                    fsc_lst.append(self.fsc)

                if i == ncycles-1:
                    # output maps
                    self.output_rotated_maps(ifit,t_accum,q_accum)

                self.step,self.grad,self.e1 = derivatives(self.e0,self.ert,
                                                    self.w_grid,self.w2_grid,
                                                    q,self.sv,xyz,xyz_sum,vol)
                alpha = self.calc_fval_for_different_kvalues_at_this_step(
                                                    self.step,self.e0,self.e1)
                self.t = self.step[:3]*alpha[0]
                t_accum = t_accum + self.t
                translation_vec = np.sqrt(np.sum(t_accum * t_accum))
                tmp = np.insert(self.step[3:]*alpha[1],0,0.0)
                q_accum = q_accum + tmp
                q_accum = q_accum/np.sqrt(np.dot(q_accum, q_accum))
                tmp = tmp + q_init
                q = tmp/np.sqrt(np.dot(tmp, tmp))
                self.q = q
            end = timer()
            print('time for one cycle:', end-start)
            self.frt_full_lst = frt_full_lst
            self.ert_full_lst = ert_full_lst
            self.rt_hf    = rt_hf_lst
            plot_nlines(self.mapobj.cres_arr,fsc_lst,
                        'before_and_after_fit.eps',["Start","End"])

    def output_rotated_maps(self,ifit,t,q):
        # Final fit parameters
        print('Final fit parameters:')
        translation = np.sqrt(np.sum(t * t))
        rotmat = get_RM(q)
        euler_angs = rotationMatrixToEulerAngles(rotmat)
        print('euler_angs [degrees]: ', euler_angs * 180./np.pi)
        # fitted map output
        imap_f = ifit + 1
        imap_hf = 2 * imap_f
        nx,ny,nz = self.mapobj.map_dim
        st,_,_,_ = fcodes.get_st(nx,ny,nz,t)
        newhf1 = get_FRS(self.cell,rotmat,self.mapobj.fhf_lst[imap_hf] * st)[:,:,:,0]
        newhf2 = get_FRS(self.cell,rotmat,self.mapobj.fhf_lst[imap_hf + 1] * st)[:,:,:,0]
        write_mrc(newhf1,'rot_trans_hf1_f2.mrc',self.cell,self.origin)
        write_mrc(newhf2,'rot_trans_hf2_f2.mrc',self.cell,self.origin)

def write_mrc(mapdata,filename,unit_cell,map_origin):
    import mrcfile as mrc
    import numpy as np
    data2write = np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mapdata))))
    file = mrc.new(name=filename, data=np.float32(data2write), compression=None, overwrite=True)
    file.header.cella.x = unit_cell[0]
    file.header.cella.y = unit_cell[1]
    file.header.cella.z = unit_cell[2]
    file.header.nxstart = map_origin[0]
    file.header.nystart = map_origin[1]
    file.header.nzstart = map_origin[2]
    file.close()


