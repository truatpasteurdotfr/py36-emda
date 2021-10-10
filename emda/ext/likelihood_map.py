from emda.core.iotools import read_map
import numpy as np
import emda.emda_methods as em
from numpy.fft import fftn, fftshift, ifftshift, ifftn
import fcodes_fast

def read_maps(maplist):
    half_pair1 = maplist[0:2]
    half_pair2 = maplist[2:4]
    #1. read in half maps
    uc, arr1, origin = em.get_data(half_pair1[0])
    uc, arr2, origin = em.get_data(half_pair1[1])
    tpixsize = [uc[i]/shape for i, shape in enumerate(arr1.shape)]

    p1_hf1 = fftshift(fftn(fftshift(arr1)))
    p1_hf2 = fftshift(fftn(fftshift(arr2)))
    p1_f = (p1_hf1 + p1_hf2) / 2

    uc2, arr3, _ = em.get_data(half_pair2[0])
    _, arr4, _ = em.get_data(half_pair2[1])
    cpixsize = [uc2[i]/shape for i, shape in enumerate(arr3.shape)]
    #2. resample p2 on p1
    newarr3 = em.resample_data(curnt_pix=cpixsize, 
                            targt_pix=tpixsize,
                            targt_dim=arr1.shape,
                            arr = arr3)
    newarr4 = em.resample_data(curnt_pix=cpixsize, 
                            targt_pix=tpixsize,
                            targt_dim=arr1.shape,
                            arr = arr4)
    p2_hf1 = fftshift(fftn(fftshift(newarr3)))
    p2_hf2 = fftshift(fftn(fftshift(newarr4)))
    p2_f = (p2_hf1 + p2_hf2) / 2
    uc1 = uc
    origin1 = origin
    return [uc1, origin1, p1_hf1, p1_hf2, p1_f, p2_hf1, p2_hf2, p2_f]


def fit_halfmaps(maplist, rotmat=None, t_init=None, fitres=None):
    if rotmat is None:
        rotmat = np.identity(3, 'float')
    if t_init is None:
        t_init = np.array([0., 0., 0.], 'float')
    mapinfo = read_maps(maplist)
    uc = mapinfo[0]
    origin = mapinfo[1]
    p1_hf1 = mapinfo[2]
    p1_hf2 = mapinfo[3]
    p1_f   = mapinfo[4] 
    p2_hf1 = mapinfo[5] 
    p2_hf2 = mapinfo[6]
    p2_f   = mapinfo[7]

    #3. estimate fitting parameters for p2
    from emda.ext.overlay import EmmapOverlay, run_fit
    from emda import core
    from emda.ext.mapfit import utils as maputils
    emmap1 = EmmapOverlay(map_list=[])
    emmap1.pixsize = [uc[i]/shape for i, shape in enumerate(p1_f.shape)]
    emmap1.map_dim = [shape for shape in p1_f.shape]
    emmap1.map_origin = origin
    emmap1.map_unit_cell = uc
    emmap1.com = True
    emmap1.fhf_lst = [p1_f, p2_f]
    emmap1.calc_fsc_from_maps()
    q_init = core.quaternions.rot2quart(rm=rotmat)
    t, q_final = run_fit(
            emmap1=emmap1,
            rotmat=core.quaternions.get_RM(q_init),
            t=[itm / emmap1.pixsize[i] for i, itm in enumerate(t_init)],
            ncycles=100,
            ifit=1,
            fitres=fitres,
        )
    rotmat = core.quaternions.get_RM(q_final)
    #4. output rotated and translated maps
    fmaps = np.stack((emmap1.fo_lst[1], p2_hf1, p2_hf2), axis=-1)
    f_static = emmap1.fo_lst[0]
    nz, ny, nx = f_static.shape

    """ f1f2_fsc_unaligned = core.fsc.anytwomaps_fsc_covariance(
        f_static, emmap1.fo_lst[1], emmap1.bin_idx, emmap1.nbin
    )[0] """
    #output static_fullmap
    data2write = np.real(ifftshift(ifftn(ifftshift(f_static))))
    core.iotools.write_mrc(
        data2write,
        'static_fullmap.mrc',
        emmap1.map_unit_cell,
    )
    frs = maputils.get_FRS(rotmat, fmaps, interp="cubic")
    st, _, _, _ = fcodes_fast.get_st(nz, ny, nx, t)
    frt_f = frs[:,:,:,0] * st
    data2write = np.real(ifftshift(ifftn(ifftshift(frt_f))))
    core.iotools.write_mrc(
        data2write,
        'moving_fullmap.mrc',
        emmap1.map_unit_cell,
    )
    frt_hf1 = frs[:,:,:,1] * st
    frt_hf2 = frs[:,:,:,2] * st
    return [emmap1, p1_hf1, p1_hf2, frt_hf1, frt_hf2]


def get_stats(hf1, hf2, bin_idx, nbin, mode):
    (
    fo,
    eo,
    nv,
    sv,
    tv,
    bin_fsc,
    bincount,
    ) = fcodes_fast.calc_fsc_using_halfmaps(
        hf1, hf2, bin_idx, nbin, 0, hf1.shape[0], hf1.shape[1], hf1.shape[2]
    )
    return [eo, fo, nv, sv, tv, bin_fsc]


def avg_and_diffmaps(
    maps2avg,
    uc,
    nbin,
    sgnl_var,
    totl_var,
    covar,
    hffsc,
    bin_idx,
    s_grid,
    res_arr,
    Bf_arr,
):
    import fcodes_fast
    import emda.core as core
    debug_mode = 0

    ######## smoothening signals
    """ from emda.ext.mapfit.newsignal import get_extended_signal
    #res_arr, signal, bin_fsc, fobj=None
    fobj = open('test.txt', '+w')
    for i, signal in enumerate(sgnl_var):
        sgnl_var[i] = get_extended_signal(res_arr=res_arr,
            signal=sgnl_var[i], bin_fsc=hffsc[i], fobj=fobj, fsc_cutoff=0.3)
    covar[0] = get_extended_signal(res_arr=res_arr,
            signal=covar[0], bin_fsc=covar[1], fobj=fobj, fsc_cutoff=0.3) """

    ######## test input data
    """ print(len(sgnl_var))
    print(len(totl_var))
    print(len(covar))
    cc = covar[0]/np.sqrt(sgnl_var[0] * sgnl_var[1])
    cc2 = covar[0]/np.sqrt(totl_var[0] * totl_var[1])
    for i, _ in enumerate(sgnl_var[0]):
        print(res_arr[i], sgnl_var[0][i], sgnl_var[1][i], covar[0][i], cc[i])
    print('T')
    for i, _ in enumerate(totl_var[0]):
        print(res_arr[i], totl_var[0][i], totl_var[1][i], covar[0][i], cc2[i]) """


    ######### average and difference map calculation

    nx, ny, nz = maps2avg[0].shape
    nmaps = len(maps2avg)
    unit_cell = uc
    all_maps = np.zeros(shape=(nx, ny, nz, nmaps), dtype="complex")
    for i in range(nmaps):
        all_maps[:, :, :, i] = maps2avg[i]
    print(all_maps.shape)
    #
    S_mat = np.zeros(shape=(nmaps, nmaps, nbin), dtype="float")
    T_mat = np.zeros(shape=(nmaps, nmaps, nbin), dtype="float")
    F_mat = np.zeros(shape=(nmaps, nmaps, nbin), dtype="float")

    #### if signal-fsc is below zero, make all zero thereafter
    from emda.ext.mapfit.mapaverage import set_array

    covariance = set_array(covar[0])
    signal1 = set_array(sgnl_var[0])
    signal2 = set_array(sgnl_var[1])
    if not (np.sqrt(signal1 * signal2) >= covariance).all():
        print("There is a problem in signal variances and covariance.")
        print("Trucated data will be used in the calculation")
        prod = np.sqrt(signal1 * signal2)
        for i, cv in enumerate(covariance):
            if prod[i] < cv:
                break
        covariance = set_array(covariance, cv)
        for i, cv in enumerate(covariance):
            print(res_arr[i], signal1[i], signal2[i], prod[i], cv)        
        #raise SystemExit("Problem in signal variances and covariance.")
    total1, total2 = totl_var[0], totl_var[1]
    if not (total1 > 0.).all():
        raise SystemExit("Problems in total variance in map 1")
    if not (total2 > 0.).all():
        raise SystemExit("Problems in total variance in map 2")
    if not (np.sqrt(total1 * total2) >= covariance).all():
        raise SystemExit("Problem in total variances and covariance.")    
    #fsc1 = set_array(signal1 / total1)
    #fsc2 = set_array(signal2 / total2)
    fsc1 = set_array(hffsc[0])
    fsc2 = set_array(hffsc[1])
    fsc1 = 2 * fsc1 / (1.0 + fsc1)
    fsc2 = 2 * fsc2 / (1.0 + fsc2)


    # just for two maps
    reg = 1e-5
    S_mat[0,0,:] = S_mat[1,1,:] = 1.0
    S_mat[0,1,:] = S_mat[1,0,:] = covariance / (np.sqrt(signal1 * signal2) + reg)
    F_mat[0,0,:] = np.sqrt(fsc1)
    F_mat[1,1,:] = np.sqrt(fsc2)
    T_mat[0,0,:] = T_mat[1,1,:] = 1.0
    T_mat[0,1,:] = T_mat[1,0,:] = covariance / np.sqrt(total1 * total2)
    # Plotting
    core.plotter.plot_nlines_log(
        res_arr,
        [covariance, signal1, signal2],
        ["S12", "S11", "S22"],
        "log_variance_signal.eps",
    )
    core.plotter.plot_nlines_log(
        res_arr,
        [covariance, total1, total2],
        ["S12", "T11", "T22"],
        "log_variance_totalv.eps",
    )
    core.plotter.plot_nlines(
        res_arr,
        [S_mat[0, 0, :], S_mat[0, 1, :], S_mat[1, 0, :], S_mat[1, 1, :]],
        "S_mat_fsc_ij.eps",
        ["FSC11", "FSC12", "FSC21", "FSC22"],
    )
    core.plotter.plot_nlines(
        res_arr,
        [F_mat[0, 0, :], F_mat[1, 1, :]],
        "F_mat_fsc_ij.eps",
        ["sqrt(FSC11)", "sqrt(FSC22)"],
    )
    core.plotter.plot_nlines(
        res_arr,
        [T_mat[0, 0, :], T_mat[0, 1, :], T_mat[1, 0, :], T_mat[1, 1, :]],
        "T_mat_fsc_ij.eps",
        ["FSC11", "FSC12", "FSC21", "FSC22"],
    )
    # Variance weighted matrices calculation
    Wgt = np.zeros(shape=(nmaps, nmaps, nbin))
    for ibin in range(nbin):
        T_mat_inv = np.linalg.pinv(T_mat[:, :, ibin])  # Moore-Penrose psedo-inversion
        tmp = np.dot(F_mat[:, :, ibin], T_mat_inv)
        Wgt[:, :, ibin] = np.dot(S_mat[:, :, ibin], tmp)
    core.plotter.plot_nlines(
        res_arr,
        [Wgt[0, 0, :], Wgt[0, 1, :], Wgt[1, 0, :], Wgt[1, 1, :]],
        "Wgt_map_ij.eps",
        ["W11", "W12", "W21", "W22"],
    )
    # output data
    fsmat = open("smat.txt", "w")
    ftmat = open("tmat.txt", "w")
    # ftmatinv = open("tmatinv.txt", "w")
    fwmat = open("wmat.txt", "w")
    for i in range(nbin):
        fsmat.write(
            "{:.2f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                res_arr[i],
                S_mat[0, 0, i],
                S_mat[0, 1, i],
                S_mat[1, 0, i],
                S_mat[1, 1, i],
            )
        )
        ftmat.write(
            "{:.2f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                res_arr[i],
                T_mat[0, 0, i],
                T_mat[0, 1, i],
                T_mat[1, 0, i],
                T_mat[1, 1, i],
            )
        )
        fwmat.write(
            "{:.2f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                res_arr[i], Wgt[0, 0, i], Wgt[0, 1, i], Wgt[1, 0, i], Wgt[1, 1, i]
            )
        )

    # Average map calculation
    nbf = len(Bf_arr)
    assert all_maps.shape[:3] == bin_idx.shape
    assert all_maps.shape[3] == nmaps
    AVG_Maps = fcodes_fast.calc_avg_maps(
        all_maps,
        bin_idx,
        s_grid,
        Wgt,
        Bf_arr,
        unit_cell,
        debug_mode,
        nbin,
        nmaps,
        nbf,
        nx,
        ny,
        nz,
    )
    return AVG_Maps


def calc_likelihoodmap(fitresults):
    import math

    p1_hf1, p1_hf2 = fitresults[1], fitresults[2]
    frt_hf1, frt_hf2 = fitresults[3], fitresults[4]
    emmap1 = fitresults[0]
    uc = emmap1.map_unit_cell
    bin_idx = emmap1.bin_idx
    nbin = emmap1.nbin
    res_arr = emmap1.res_arr
    origin = emmap1.map_origin

    nz, ny, nx = p1_hf1.shape
    maxbin = np.amax(np.array([nx // 2, ny // 2, nz // 2]))
    _, s_grid, _ = fcodes_fast.resolution_grid_full(
                            uc, 0.0, 1, maxbin, nx, ny, nz
                        )

    stats1 = get_stats(hf1=p1_hf1, 
                       hf2=p1_hf2, 
                       bin_idx=bin_idx, 
                       nbin=nbin, 
                       mode=0)

    stats2 = get_stats(hf1=frt_hf1, 
                       hf2=frt_hf2, 
                       bin_idx=bin_idx, 
                       nbin=nbin, 
                       mode=0)

    # 2. now calculate covariance between full maps of each
    stats3 = get_stats(hf1=stats1[1], 
                       hf2=stats2[1], 
                       bin_idx=bin_idx, 
                       nbin=nbin, 
                       mode=0)

    # 3. print results
    """ for i in range(nbin):
        print(stats1[4][i], stats2[4][i], stats3[3][i], 
            stats3[3][i]/math.sqrt(stats1[4][i] * stats2[4][i]))
        print(stats2[3][i], stats2[4][i]) """

    # 4. calculate average map
    from emda.ext.mapfit.utils import output_maps #, avg_and_diffmaps
    averagemaps = avg_and_diffmaps(maps2avg=[stats1[0], stats2[0]],
                                uc=uc,
                                nbin=nbin,
                                sgnl_var=[stats1[3],stats2[3]],
                                totl_var=[stats1[4],stats2[4]],
                                covar=[stats3[3]],
                                hffsc=[stats1[5], stats2[5]],
                                bin_idx=bin_idx,
                                s_grid=s_grid,
                                res_arr=res_arr,
                                Bf_arr=[0.0])
    # 5. output maps
    output_maps(averagemaps=averagemaps,
                com_lst=[],
                t_lst=[],
                r_lst=[],
                unit_cell=uc,
                map_origin=origin,
                bf_arr=[0.0],
                center=None,
                )


def output_maps(
    averagemaps, com_lst, t_lst, r_lst, center, unit_cell, map_origin, bf_arr
):
    import emda.core as core

    nx, ny, nz = averagemaps.shape[:3]
    for ibf in range(averagemaps.shape[4]):
        if bf_arr[ibf] < 0.0:
            Bcode = "_blur" + str(abs(bf_arr[ibf]))
        elif bf_arr[ibf] > 0.0:
            Bcode = "_sharp" + str(abs(bf_arr[ibf]))
        else:
            Bcode = "_unsharpened"
        for imap in range(averagemaps.shape[3]):
            filename_mrc = "avgmap_" + str(imap) + Bcode + ".mrc"
            data2write = np.real(ifftshift(ifftn(ifftshift(averagemaps[:, :, :, imap, ibf]))))
            core.iotools.write_mrc(data2write, filename_mrc, unit_cell, map_origin)

        # Difference and Biase removed map calculation
        nmaps = averagemaps.shape[3]
        for m in range(nmaps-1):
           for n in range(m+1,nmaps):
               fname_diff1 = 'diffmap_m'+str(n)+'-m'+str(m)+Bcode
               dm1 = np.real(ifftshift(ifftn(ifftshift(averagemaps[:,:,:,n,ibf] - averagemaps[:,:,:,m,ibf]))))
               core.iotools.write_mrc(dm1,
                                 fname_diff1+'.mrc',
                                 unit_cell,map_origin)
               dm2 = np.real(ifftshift(ifftn(ifftshift(averagemaps[:,:,:,m,ibf] - averagemaps[:,:,:,n,ibf]))))
               fname_diff2 = 'diffmap_m'+str(m)+'-m'+str(n)+Bcode
               core.iotools.write_mrc(dm2,
                                 fname_diff2+'.mrc',
                                 unit_cell,map_origin)


def main(maplist, fit=True, fitres=None, masklist=None):
    nmaps = len(maplist)
    if nmaps != 4:
        print("Current implementation accept only two pairs.")
        raise SystemExit("Likelihood based difference map needs half maps for each map")
    else:
        if fit:
            # likelihood diffmap with fit
            fitresults = fit_halfmaps(maplist=maplist,fitres=fitres)
            calc_likelihoodmap(fitresults)
        else:
            # difference map without fit
            mapinfo = read_maps(maplist)
            uc = mapinfo[0]
            origin = mapinfo[1]
            p1_hf1 = mapinfo[2]
            p1_hf2 = mapinfo[3]
            p1_f   = mapinfo[4] 
            p2_hf1 = mapinfo[5] 
            p2_hf2 = mapinfo[6]
            p2_f   = mapinfo[7]
            from emda.ext.overlay import EmmapOverlay
            from emda.core import restools
            emmap1 = EmmapOverlay(map_list=[])
            emmap1.pixsize = [uc[i]/shape for i, shape in enumerate(p1_f.shape)]
            emmap1.map_dim = [shape for shape in p1_f.shape]
            emmap1.map_origin = origin
            emmap1.map_unit_cell = uc   
            nbin, res_arr, bin_idx = restools.get_resolution_array(uc, p1_hf1)
            emmap1.bin_idx = bin_idx
            emmap1.nbin = nbin
            emmap1.res_arr = res_arr
            calc_likelihoodmap([emmap1, p1_hf1, p1_hf2, p2_hf1, p2_hf2])


if __name__=="__main__":
    maplist = [
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/lig_hf1.mrc",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/lig_hf2.mrc",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/pair_alignment/nat_fitted_halfmap1.mrc",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/pair_alignment/nat_fitted_halfmap2.mrc",            
                ]
    main(maplist, fit=False)