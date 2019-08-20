import numpy as np
from timeit import default_timer as timer
import numpy as np
from emda.iotools import read_map,resample2staticmap
from emda.mapfit.utils import remove_unwanted_corners
from emda.restools import get_resolution_array,cut_resolution
from emda.fsc import *
from emda.config import *

#debug_mode = 0 # 0: no debug info, 1: debug

class EmMap:
    
    def __init__(self,hfmap_list):
        self.hfmap_list         = hfmap_list
        self.map_unit_cell      = None
        self.map_origin         = None
        self.map_dim            = None
        self.high_res           = 6.0   

    def load_maps(self):
        fhf_lst = []
        for i in range(len(self.hfmap_list)):
            uc,arr,origin = read_map(self.hfmap_list[i])
            if i == 0: 
                map_origin = origin
                uc_target = uc
                target_dim = arr.shape
                target_pix_size = uc_target[0]/target_dim[0]
                corner_mask = remove_unwanted_corners(uc,target_dim)
                fhf_lst.append(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr * corner_mask))))
            else:
                newarr = resample2staticmap(target_pix_size,target_dim,uc,arr)
                fhf_lst.append(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(newarr * corner_mask))))
        self.map_origin     = map_origin
        self.map_unit_cell  = uc_target
        self.map_dim        = target_dim 
        self.fhf_lst       = fhf_lst           

    def calc_fsc_variance_from_halfdata(self):  
        nmaps = len(self.fhf_lst) 
        fFo_lst = []
        fEo_lst = []
        fBNV_lst = []
        fBSV_lst = []
        fBTV_lst = []
        fBFsc_lst = []
        # create resolution array and bin_idx
        self.nbin,self.res_arr,self.bin_idx = get_resolution_array(
                                                        self.map_unit_cell,
                                                        self.fhf_lst[0])
        for i in range(0,nmaps,2):
            # calculate fsc and other parameters using half maps
            bin_fsc,noisevar,signalvar,totalvar,fo,eo = halfmaps_fsc_variance(
                self.fhf_lst[i],self.fhf_lst[i+1],self.bin_idx,self.nbin) 

            fFo_lst.append(fo)
            fEo_lst.append(eo)
            fBNV_lst.append(noisevar)
            fBSV_lst.append(signalvar)
            fBTV_lst.append(totalvar)
            fBFsc_lst.append(bin_fsc)

        self.fo_lst            = fFo_lst
        self.eo_lst            = fEo_lst
        self.signalvar_lst     = fBSV_lst
        self.totalvar_lst      = fBTV_lst
        self.hffsc_lst         = fBFsc_lst

    def make_data_for_fit(self,smax=6):
        # Making data for map fitting
        print('data for fit will be truncated at ',smax,' A')
        nfit = len(self.fo_lst)
        #smax = self.high_res
        dist = np.sqrt((self.res_arr - smax)**2)
        cbin = np.argmin(dist) + 1 # adding 1 because fResArr starts with zero
        cResArr = self.res_arr[:cbin]
        if cbin%2 != 0: cx = cbin + 1
        else: cx = cbin
        print('cx = ', cx, 'cbin=', cbin)
        dx = int((self.map_dim[0] - 2*cx)/2)
        dy = int((self.map_dim[1] - 2*cx)/2)
        dz = int((self.map_dim[2] - 2*cx)/2)
        cBIdx = self.bin_idx[dx:dx+2*cx,dy:dy+2*cx,dz:dz+2*cx]
        #
        cEo_lst = []
        cFo_lst = []
        for i in range(nfit):
            cEo_lst.append(cut_resolution(self.eo_lst[i],self.bin_idx,self.res_arr,smax)
                            [dx:dx+2*cx,dy:dy+2*cx,dz:dz+2*cx])
            cFo_lst.append(cut_resolution(self.fo_lst[i],self.bin_idx,self.res_arr,smax)
                            [dx:dx+2*cx,dy:dy+2*cx,dz:dz+2*cx])
        #
        cx,cy,cz = cBIdx.shape
        self.cfo_lst      = cFo_lst
        self.ceo_lst      = cEo_lst
        self.cbin     = cbin
        self.cdim     = [cx,cy,cz]
        self.cres_arr = cResArr
        self.cbin_idx  = cBIdx
