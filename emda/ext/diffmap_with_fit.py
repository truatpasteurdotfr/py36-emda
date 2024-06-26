# difference map combined with map-to-map fit
import numpy as np
import emda.emda_methods as em
import fcodes_fast
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from emda.core import iotools, quaternions, fsc, plotter
from emda.ext.mapfit import utils
from emda.ext.difference import diffmap_normalisedsf
from emda.ext.overlay import EmmapOverlay, run_fit, output_rotated_models
from emda.ext import diffmap_snr

""" path = "/Users/ranganaw/MRC/REFMAC/Jude/rcc/JUDE/DT72/chains/diffmap_DT72_vs_BRC/"
#maplist = [path + "DT72_bestmap.mrc", path + "BRC_bestmap.mrc"]
#masklist = [path + "DT72_mask.mrc", path + "BRC_mask.mrc"]

maplist = [
        "/Users/ranganaw/MRC/REFMAC/Vinoth/nat_half1_class001_unfil.map",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/nat_half2_class001_unfil.map",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/lig_half1_class001_unfil.map",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/lig_half2_class001_unfil.map",
] """


def fitmaps(maplist, ncy=100, usecom=False, masklist=None, fitres=None, 
        fobs=None, rot=None, trans=None, axr=None):
    import emda.emda_methods as em
    print(maplist)
    emmap1, rotmat_list, trans_list = em.overlay_maps(
        maplist=maplist,
        masklist=masklist,
        ncy=ncy,
        rot=rot,
        fitres=fitres,
        nocom=usecom,
        axr=axr,
        tra=trans,
    )
    return emmap1, rotmat_list, trans_list 


def apply_transformation_on_f(emmap1, r_lst, t_lst):
    f_list = []
    fo_lst = emmap1.fo_lst
    bin_idx = emmap1.bin_idx
    nbin = emmap1.nbin
    f_static = fo_lst[0]
    f_list.append(f_static)
    nx, ny, nz = f_static.shape
    i = 0
    for fo, t, rotmat in zip(fo_lst[1:], t_lst, r_lst):
        i += 1
        fsc_unaligned = fsc.anytwomaps_fsc_covariance(
            f_static, fo, bin_idx, nbin
        )[0]
        st, _, _, _ = fcodes_fast.get_st(nx, ny, nz, t)
        frt = utils.get_FRS(rotmat, fo * st, interp="cubic")[:, :, :, 0]  
        fsc_aligned = fsc.anytwomaps_fsc_covariance(f_static, frt, bin_idx, nbin)[0]  
        f_list.append(frt)
        plotter.plot_nlines(
            emmap1.res_arr,
            [fsc_unaligned[:nbin], fsc_aligned[:nbin]],
            "{0}_{1}.{2}".format("fsc", str(i), "eps"),
            ["FSC before", "FSC after"],
        )
    return f_list


def calculate_diffmap(emmap1, f_list, resol):
    if len(f_list) > 2: 
        print("At the moment I can handle only two maps max.")
    nx, ny, nz = f_list[0].shape
    diffmap = fcodes_fast.diffmap_norm(
        fo=f_list[0],
        fc=f_list[1],
        bin_idx=emmap1.bin_idx,
        res_arr=emmap1.res_arr,
        smax=resol,
        mode=1,
        nbin=emmap1.nbin,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    return diffmap


def mapoutput(list_maps, uc, origin, masklist=None):
    if all(mask is not None for mask in masklist):
        from emda.ext.utils import binarize_mask

        # combined mask
        msk = binarize_mask(mask=masklist[0]) * binarize_mask(mask=masklist[1])

        # calculate rmsd
        sum_mask = np.sum(msk)
        mean_for_masked_diffmap_m1m2 = np.sum(list_maps[0] * msk) / sum_mask
        mean_for_masked_diffmap_m2m1 = np.sum(list_maps[1] * msk) / sum_mask
        mean_for_masked_nem1 = np.sum(list_maps[2] * msk) / sum_mask
        mean_for_masked_nem2 = np.sum(list_maps[3] * msk) / sum_mask

        dev_from_masked_mean_diffmap_m1m2 = (list_maps[0] - mean_for_masked_diffmap_m1m2) * msk
        dev_from_masked_mean_diffmap_m2m1 = (list_maps[1] - mean_for_masked_diffmap_m2m1) * msk
        dev_from_masked_mean_nem1 = (list_maps[2] - mean_for_masked_nem1) * msk
        dev_from_masked_mean_nem2 = (list_maps[3] - mean_for_masked_nem2) * msk

        rmsd_diffmap_m1m2 = np.sqrt(np.sum(dev_from_masked_mean_diffmap_m1m2 * dev_from_masked_mean_diffmap_m1m2) / sum_mask)
        rmsd_diffmap_m2m1 = np.sqrt(np.sum(dev_from_masked_mean_diffmap_m2m1 * dev_from_masked_mean_diffmap_m2m1) / sum_mask)
        rmsd_nem1 = np.sqrt(np.sum(dev_from_masked_mean_nem1 * dev_from_masked_mean_nem1) / sum_mask)
        rmsd_nem2 = np.sqrt(np.sum(dev_from_masked_mean_nem2 * dev_from_masked_mean_nem2) / sum_mask)

        # output maps
        iotools.write_mrc((list_maps[0] / rmsd_diffmap_m1m2) * msk, 'emda_diffmap_m1-m2.mrc', uc, origin)
        iotools.write_mrc((list_maps[1] / rmsd_diffmap_m2m1) * msk, 'emda_diffmap_m2-m1.mrc', uc, origin)
        iotools.write_mrc((list_maps[2] / rmsd_nem1) * msk, 'emda_map1.mrc', uc, origin)
        iotools.write_mrc((list_maps[3] / rmsd_nem2) * msk, 'emda_map2.mrc', uc, origin)


        """ for i, _ in enumerate(masklist):
            masked_mean = np.sum(list_maps[i] * msk) / np.sum(msk)
            diff = (list_maps[i] - masked_mean) * msk
            rmsd = np.sqrt(np.sum(diff * diff) / np.sum(msk))
            print("rmsd: ", rmsd)   
            fname_dif = "emda_diffmap_m%s.mrc" % (str(i+1))
            #iotools.write_mrc(((list_maps[i] - masked_mean) / rmsd) * msk, fname_dif, uc, origin)
            iotools.write_mrc((list_maps[i] / rmsd) * msk, fname_dif, uc, origin)
            fname_map = "emda_map%s.mrc" % (str(i+1))
            masked_mean_map = np.sum(list_maps[i+2] * msk) / np.sum(msk)
            diff2 = (list_maps[i+2] - masked_mean_map) * msk
            rmsd2 = np.sqrt(np.sum(diff2 * diff2) / np.sum(msk))
            #iotools.write_mrc(((list_maps[i+2]- masked_mean_map) / rmsd2) * msk, fname_map, uc, origin)
            iotools.write_mrc((list_maps[i+2] / rmsd2) * msk, fname_map, uc, origin) """
    else:
        iotools.write_mrc(list_maps[0], 'emda_diffmap_m1-m2.mrc', uc, origin)
        iotools.write_mrc(list_maps[1], 'emda_diffmap_m2-m1.mrc', uc, origin)
        iotools.write_mrc(list_maps[2], 'emda_map1.mrc', uc, origin)
        iotools.write_mrc(list_maps[3], 'emda_map2.mrc', uc, origin)

        """ for i , _ in enumerate(list_maps):
            if i < 2:
                fname_dif = "emda_diffmap_m%s.mrc" % (str(i+1))
                iotools.write_mrc(list_maps[i], fname_dif, uc, origin)
            else:
                fname_map = "emda_map%s.mrc" % (str(i-1))
                iotools.write_mrc(list_maps[i], fname_map, uc, origin) """


class MapOut:
    def __init__(self):
        self.diffmap = None
        self.masklist = None
        self.cell = None
        self.origin = None


def main(maplist, diffmapres=3, ncy=5, fit=False, usecom=False, 
        usehalfmaps=False, fitres=None, masklist=None, rot=None, axr=None, trans=None):
    results = MapOut()
    modelres = diffmapres
    if usehalfmaps:
        print("Likelihood difference map using half maps")
        # call other module
        import emda.ext.likelihood_map as lhmap
        lhmap.main(maplist=maplist, fit=fit, fitres=fitres)
        raise SystemExit() #

        """ #previous differnce map
        assert len(maplist) == 4
        if masklist is not None:
            assert len(masklist) == 2
        try:
            results = diffmap_snr.main(maplist=maplist, 
                                       fit=fit, 
                                       ncy=ncy, 
                                       usecom=usecom, 
                                       fitres=fitres, 
                                       resol=diffmapres, 
                                       masklist=masklist, 
                                       results=results)
        except ValueError:
            results = diffmap_snr.main(maplist=maplist, 
                                       fit=fit, 
                                       ncy=ncy, 
                                       usecom=usecom, 
                                       fitres=fitres, 
                                       resol=diffmapres, 
                                       results=results) """
    else:
        if maplist[0].endswith(((".mrc", ".map"))) and maplist[1].endswith(((".pdb", ".cif", ".ent"))):
            uc, arr1, origin = em.get_data(maplist[0])
            if modelres < 0.1:
                raise SystemExit(
                    "Please specify resolution to calculate map from atomic model"
                )
            # calculate map from model
            modelmap = em.model2map(
                modelxyz=maplist[1],
                dim=arr1.shape,
                resol=modelres,
                cell=uc,
                maporigin=origin,
            )
            em.write_mrc(modelmap, "emda_modelmap.mrc", uc, origin)
            maplist[1] = "emda_modelmap.mrc"
        if fit:
            emmap1, rotmat_list, trans_list = fitmaps(maplist=maplist, 
                                                masklist=masklist, 
                                                usecom=usecom, 
                                                ncy=ncy, 
                                                fitres=fitres,
                                                rot=rot,
                                                trans=trans,
                                                axr=axr)
            flist = apply_transformation_on_f(emmap1, rotmat_list, trans_list)
            output_rotated_models(emmap1=emmap1, maplist=maplist, r_lst=rotmat_list, t_lst=trans_list)
            print("resol: ", diffmapres)
            diffmap = calculate_diffmap(emmap1=emmap1, f_list=flist, resol=diffmapres)
            msk_list = emmap1.msk_lst
            results.diffmap = diffmap
            results.masklist = msk_list
            results.cell = emmap1.map_unit_cell
            results.origin = emmap1.map_origin
        else:
            uc, arr1, origin = em.get_data(maplist[0])
            _, arr2, _ = em.get_data(maplist[1])
            if all(mask is not None for mask in masklist):
                _, msk1, _ = iotools.read_map(masklist[0])
                _, msk2, _ = iotools.read_map(masklist[1])
                results.masklist = [msk1, msk2]
                print("applying masks on maps..")
                arr1 = arr1 * msk1
                arr2 = arr2 * msk2
            else:
                results.masklist = masklist             
            f1 = fftshift(fftn(arr1))
            f2 = fftshift(fftn(arr2))
            diffmap = diffmap_normalisedsf(f1, f2, uc, smax=diffmapres, origin=origin)
            results.diffmap = diffmap
            results.cell = uc
            results.origin = origin
    return results



def difference_map(maplist, diffmapres=3.0, mode="norm", fit=False, 
        ncy=100, usecom=False, usehalfmaps=False, fitres=None, masklist=None,
        rot=None, axr=None, trans=None):
    """Calculates difference map.

    Arguments:
        Inputs:
            maplist: string
                List of map names to calculate difference maps.
                If combined with fit parameter, firstmap in the list
                will be taken as static/reference map. If this list
                contains coordinate file (PDB/CIF), give it in the second place.
                Always give MRC/MAP file at the beginning of the list.
                e.g:
                    [test1.mrc, test2.mrc] or
                    [test1.mrc, model1.pdb/cif]
                If combined with usehalfmaps argument, then halfmaps of the
                firstmap should be given first and those for second next.
                e.g:
                    [map1-halfmap1.mrc, map1-halfmap2.mrc, 
                     map2-halfmap1.mrc, map2-halfmap2.mrc]

            masklist: string, optional
                List of masks to apply on maps.
                All masks should be in MRC/MAP format.
                e.g:
                    [mask1.mrc, mask2.mrc]

            diffmapres: float
                Resolution to which difference map be calculated.
                If an atomic model involved, this resolution will be used
                for map calculation from coordinates

            mode: string, optional
                Different modes to scale maps. Two difference modes are supported.
                'ampli' - scale between maps is based on amplitudes .
                'norm' [Default] - normalized maps are used to calculate difference map.
                If fit is enabled, only norm mode used.

            usehalfmaps: boolean
                If employed, halfmaps are used for fitting and 
                difference map calculation.
                Default is False.

            fit: boolean
                If employed, maps and superimposed before calculating
                difference map.
                Default is False.

        Outputs:
            Outputs diffence maps and initial maps after scaling in MRC format.
            Differece maps are labelled as
                emda_diffmap_m1.mrc, emda_diffmap_m2.mrc
            Scaled maps are labelled as
                emda_map1.mrc, emda_map2.mrc
    """
    from emda.ext import difference

    if mode == "norm":
        results = main(maplist=maplist, diffmapres=diffmapres, 
                       fit=fit, ncy=ncy, fitres=fitres, usecom=usecom, 
                       usehalfmaps=usehalfmaps, masklist=masklist,
                       rot=rot, axr=axr, trans=trans
                       )
        diffmap = results.diffmap
        list_maps = []
        list_masks = results.masklist
        uc = results.cell
        origin = results.origin
        if fit:
            for i in range(diffmap.shape[3]):
                imap = np.real(ifftshift(ifftn(ifftshift(diffmap[:, :, :, i]))))
                list_maps.append(imap)
        else:
            for i in range(diffmap.shape[3]):
                imap = np.real(ifftn(ifftshift(diffmap[:, :, :, i])))
                list_maps.append(imap)
        mapoutput(list_maps=list_maps, uc=uc, origin=origin, masklist=list_masks)

    if mode == "ampli":
        modelres = diffmapres
        assert len(maplist) == 2
        if maplist[0].endswith(((".mrc", ".map"))) and maplist[1].endswith(
            ((".mrc", ".map"))
        ):
            uc, arr1, origin = iotools.read_map(maplist[0])
            _, arr2, _ = iotools.read_map(maplist[1])
        elif maplist[0].endswith(((".mrc", ".map"))) and maplist[1].endswith(
            ((".pdb", ".cif", ".ent"))
        ):
            uc, arr1, origin = iotools.read_map(maplist[0])
            if modelres < 0.1:
                raise SystemExit(
                    "Please specify resolution to calculate map from atomic model"
                )
            # calculate map from model
            arr2 = em.model2map(
                modelxyz=maplist[1],
                dim=arr1.shape,
                resol=modelres,
                cell=uc,
                maporigin=origin,
            )

        f1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr1)))
        f2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(arr2)))
        list_masks = None
        if masklist is not None:
            assert len(maplist) == len(masklist) == 2
            _, msk1, _ = iotools.read_map(masklist[0])
            _, msk2, _ = iotools.read_map(masklist[1])
            list_masks = [msk1, msk2]

        diffmap = difference.diffmap_scalebyampli(
            f1=f1, f2=f2, cell=uc, origin=origin, smax=diffmapres
        )
        list_maps = []
        for i in range(diffmap.shape[3]):
            imap = np.real(ifftshift(ifftn(ifftshift(diffmap[:, :, :, i]))))
            list_maps.append(imap)
        mapoutput(list_maps=list_maps, uc=uc, origin=origin, masklist=list_masks)

""" if __name__ == "__main__":
    difference_map(maplist=maplist, fit=True, usehalfmaps=True) """
    
