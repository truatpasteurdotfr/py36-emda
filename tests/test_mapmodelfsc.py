# testcode for map-model calculation
import sys
import emda.emda_methods as em
import numpy as np
import emda.core as core


def calculate_fsc(uc, arr1, modelmap):
    f_map = np.fft.fftshift(np.fft.fftn(arr1))
    f_model = np.fft.fftshift(np.fft.fftn(modelmap))
    nbin, res_arr, bin_idx = core.restools.get_resolution_array(uc, arr1)
    bin_stats = core.fsc.anytwomaps_fsc_covariance(
        f1=f_map, f2=f_model, bin_idx=bin_idx, nbin=nbin
    )
    print("FSC calculation Done.")
    bin_fsc, bin_count = bin_stats[0], bin_stats[2]
    return res_arr, bin_fsc



if __name__ == "__main__":
    emdbid = 'xxx'
    imap, imodel, resol = sys.argv[1:]

    uc, arr, orig = em.get_data(imap)

    model_map = em.model2map_gm(modelxyz=imodel, 
                    resol=float(resol), 
                    dim=arr.shape, 
                    cell=uc, 
                    maporigin=orig)
    em.write_mrc(model_map, 'model_map.mrc', uc, orig)
        
    modelmask = em.mask_from_atomic_model(mapname=imap, 
                    modelname=imodel, 
                    atmrad=5)
    em.write_mrc(modelmask, 'modelmask.mrc', uc, orig)

    res_arr, bin_fsc = calculate_fsc(uc, arr*modelmask, model_map*modelmask)

    core.plotter.plot_nlines(
        res_arr=res_arr,
        list_arr=[bin_fsc],
        mapname="{}_mapmodel_fsc.eps".format(emdbid),
        curve_label=["FSC"],
        fscline=0.5,
        plot_title="Map-model FSC"
    )


