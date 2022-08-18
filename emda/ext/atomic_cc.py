"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# Get CC at atoms
from __future__ import absolute_import, division, print_function, unicode_literals
import sys, math
import numpy as np
import gemmi
import emda.emda_methods as em
from emda.core import iotools
from scipy.interpolate import RegularGridInterpolator


def get_sequence(modelname):
    st = gemmi.read_structure(modelname)
    block = st.sole_block()
    seq = block.find_values("_atom_site.label_comp_id")
    return seq


def get_chains(modelname, xyz_cc_map, xyz_cc_mapmodel, fid, fid2):
    mapavgcc_list = []
    mapmodelavgcc_list = []
    st = gemmi.read_structure(modelname)
    atm_counter = -1
    res_counter = -1
    chain_list = []
    residue_name_list = []
    residue_num_list = []
    chain_ids = []
    fid.write("ChainID  Residue  Atom  mapCC  CC*  mapmodelCC\n")
    fid2.write("ChainID  Residue  mapCC  CC*  mapmodelCC\n")
    for chain in st[0]:
        polymer = chain.get_polymer()
        for ires, residue in enumerate(polymer):
            if ires == 0:
                chain_ids.append(chain.name)
            mapavgcc = 0.0
            mapmodelavgcc = 0.0
            j = 0
            for _, atom in enumerate(residue):
                atm_counter += 1
                j += 1
                mapCC = xyz_cc_map[atm_counter]
                if mapCC < 0.:
                    print('mapCC: ', mapCC)
                    CC_str = 0.
                else:
                    CC_str = math.sqrt(mapCC)
                mapmodelCC = xyz_cc_mapmodel[atm_counter]
                mapavgcc += mapCC #xyz_cc_map[atm_counter]
                mapmodelavgcc += mapmodelCC #xyz_cc_mapmodel[atm_counter]
                fid.write("{} {} {} {:-6.2f} {:-6.2f} {:-6.2f}\n".format(
                    chain.name, residue.name, atom.name, mapCC, CC_str, mapmodelCC))
            mapavgcc_list.append(mapavgcc/j)
            mapmodelavgcc_list.append(mapmodelavgcc/j)
            print("{} {} {:-6.2f} {:-6.2f}".format(
                chain.name, residue.name+str(residue.seqid.num), mapavgcc/j, mapmodelavgcc/j))
            mapcc_resi = mapavgcc/j
            ccstr_resi = math.sqrt(mapcc_resi)
            mapmodelcc_resi = mapmodelavgcc/j
            fid.write("{} {} {} {:-6.2f} {:-6.2f} {:-6.2f}\n".format(
                "    ", chain.name, residue.name+str(residue.seqid.num), mapcc_resi, ccstr_resi, mapmodelcc_resi))
            fid2.write("{} {} {:-6.2f} {:-6.2f} {:-6.2f}\n".format(
                chain.name, residue.name+str(residue.seqid.num), mapcc_resi, ccstr_resi, mapmodelcc_resi))
            chain_list.append(chain.name)
            residue_name_list.append(residue.name)
            residue_num_list.append(residue.seqid.num)
    return chain_ids, chain_list, residue_name_list, residue_num_list, mapavgcc_list, mapmodelavgcc_list


def get_rcc(map1, map2, modelname, model_resol):
    from emda.ext.realsp_local import RealspaceLocalCC
    rccobj = RealspaceLocalCC()
    rccobj.hfmap1name = map1
    rccobj.hfmap2name = map2
    rccobj.model = modelname # "modelmap.mrc"
    rccobj.model_resol = model_resol
    #rccobj.model = "modelmap.mrc"
    rccobj.maskname = "emda_atomic_mask.mrc"
    rccobj.norm = True
    rccobj.rcc()
    return rccobj.fullmapcc, rccobj.mapmodelcc


def interpolate_cc(data, x, y, z):
    # interpolate on the grid
    linx = np.linspace(0, data.shape[0], data.shape[0])
    liny = np.linspace(0, data.shape[1], data.shape[1])
    linz = np.linspace(0, data.shape[2], data.shape[2])
    my_interpolating_function = RegularGridInterpolator(
        (linx, liny, linz), data, method="nearest")
    xyz_cc = my_interpolating_function(np.column_stack((z, y, x)))
    return xyz_cc


def get_atomic_cc(map1, map2, modelname, resol):
    # read maps and calculate correlation
    uc, arr1, orig = em.get_data(map1)
    uc, arr2, orig = em.get_data(map2)
    _ = em.mask_from_atomic_model(map1, modelname)
    #uc, r_full_cc, orig = em.get_data("rcc_fullmap_smax9.mrc")
    #uc, r_mapmodelcc, orig = em.get_data("rcc_mapmodel_smax9.mrc")
    r_full_cc, r_mapmodelcc = get_rcc(map1, map2, modelname, resol)
    em.write_mrc(r_full_cc, "fullmapcc.mrc", uc, orig)
    em.write_mrc(r_mapmodelcc, "modelmapcc.mrc", uc, orig)
    # read the model and map coordinates into the grid
    if modelname.endswith((".pdb", ".ent")):
        iotools.pdb2mmcif(modelname)
        _, x_np, y_np, z_np, _ = iotools.read_mmcif('out.cif')
    elif modelname.endswith((".cif")):
        _, x_np, y_np, z_np, _ = iotools.read_mmcif(modelname)
    # now map model coords into the 3d grid.
    x = (x_np * arr1.shape[0] / uc[0]).astype('float')
    y = (y_np * arr1.shape[1] / uc[1]).astype('float')
    z = (z_np * arr1.shape[2] / uc[2]).astype('float')
    mapcc = interpolate_cc(r_full_cc, x, y, z)
    mapmodelcc = interpolate_cc(r_mapmodelcc, x, y, z)
    return mapcc, mapmodelcc

def get_atomic_cc_helper_for_rcc(uc, rcc_map, rcc_mapmodel, modelname):
    # read the model and map coordinates into the grid
    arr1 = rcc_map
    if modelname.endswith((".pdb", ".ent")):
        iotools.pdb2mmcif(modelname)
        _, x_np, y_np, z_np, _ = iotools.read_mmcif('out.cif')
    elif modelname.endswith((".cif")):
        _, x_np, y_np, z_np, _ = iotools.read_mmcif(modelname)
    # now map model coords into the 3d grid.
    x = (x_np * arr1.shape[0] / uc[0]).astype('float')
    y = (y_np * arr1.shape[1] / uc[1]).astype('float')
    z = (z_np * arr1.shape[2] / uc[2]).astype('float')
    mapcc = interpolate_cc(rcc_map, x, y, z)
    mapmodelcc = interpolate_cc(rcc_mapmodel, x, y, z)
    return mapcc, mapmodelcc


def plot_cc(residue_cc, xlabel="Residue Number", ylabel="Correlation Coefficient", resnum_chain=None):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    if resnum_chain is not None:
        fig = plt.figure(figsize=(10, 2))
        maxnum = max(resnum_chain)
        minnum = min(resnum_chain)
        print(maxnum)
        cc_np = np.zeros(maxnum+1, dtype='float')
        for i, resnum in enumerate(resnum_chain):
            cc_np[resnum] = residue_cc[i]
        ax1 = fig.add_subplot(111)
        ax1.plot(cc_np)
        xticks = np.arange(minnum, maxnum+1, 100)
        ax1.set_xticks(xticks)
        ax1.set_xlabel(xlabel)
        plt.ylabel(ylabel)
        ax1.set_xlim([minnum, maxnum])
        ax1.set_ylim([0, 1.0])
    else:
        plt.plot(residue_cc)
    plt.grid(True, color='grey', linestyle='dotted', linewidth=1)
    plt.show()


def get_space(beg_res_list, end_res_list, facecolor='green'):
    import matplotlib.patches as patches
    space = []
    for ib, ie in zip(beg_res_list, end_res_list):
        print(ib, ie)
        space.append(patches.Rectangle(
                    (int(ib), 1.1),
                    int(ie),
                    0.08,
                    #edgecolor = 'blue',
                    facecolor=facecolor,
                    fill=True
                ))
    return space


def plot_ncc(
    chainids, 
    mapcclist, 
    mapmodelcclist, 
    xlabel="Residue Number", 
    ylabel="CC", 
    resnumlist=None, 
    secondary_struc=None,
    #beg_res_list_all=None, 
    #end_res_list_all=None,
    ):
    import matplotlib
    matplotlib.use(matplotlib.get_backend())
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

    nplot = len(chainids)
    fig = plt.subplots(nrows=nplot, figsize=(15, 15))
    i = 0
    for i, _ in enumerate(chainids):
        ax1 = plt.subplot(nplot, 1, i+1)
        residue_mapcc = mapcclist[i]
        residue_mapmodelcc = mapmodelcclist[i]
        resnum_chain = resnumlist[i]
        hlxbeg_res_list = secondary_struc[0][i]
        hlxend_res_list = secondary_struc[1][i]
        #hlxbeg_res_list = beg_res_list_all[i]
        #hlxend_res_list = end_res_list_all[i]
        shtbeg_res_list = secondary_struc[2][i]
        shtend_res_list = secondary_struc[3][i]

        maxnum = max(resnum_chain)
        minnum = min(resnum_chain)
        #print(maxnum)
        mapcc_np = np.empty(maxnum+1)
        mapcc_np[:] = np.NaN
        mapmodelcc_np = np.empty(maxnum+1)
        mapmodelcc_np[:] = np.NaN
        for j, resnum in enumerate(resnum_chain):
            mapcc_np[resnum] = residue_mapcc[j]
            mapmodelcc_np[resnum] = residue_mapmodelcc[j]
        # chain average correlation values
        avg_mapcc = np.nanmean(mapcc_np)
        avg_mapmodelcc = np.nanmean(mapmodelcc_np)
        #
        line1, = ax1.plot(mapcc_np)
        line2, = ax1.plot(mapmodelcc_np)
        #ax1.plot((minnum, maxnum), (avg_mapcc, avg_mapcc), color="black", linestyle=":")
        #ax1.plot((minnum, maxnum), (avg_mapmodelcc, avg_mapmodelcc), color="black", linestyle=":")
        # Patch
        for ib, ie in zip(hlxbeg_res_list, hlxend_res_list):
            # Helices
            ax1.add_patch(
                patches.Rectangle(
                    (float(ib), 1.05),
                    float(ie) - float(ib),
                    0.05,
                    #edgecolor = 'blue',
                    facecolor='red',
                    alpha=0.5,
                    fill=True
                ))
        for ib, ie in zip(shtbeg_res_list, shtend_res_list):
            # Sheets
            ax1.add_patch(
                patches.Rectangle(
                    (float(ib), 1.05),
                    float(ie) - float(ib),
                    0.05,
                    #edgecolor = 'blue',
                    facecolor='blue',
                    alpha=0.5,
                    fill=True
                ))
        #
        xticks = np.arange(minnum, maxnum+1, (maxnum-minnum+1)//10)
        yticks = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float')
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        # ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_xlim([minnum, maxnum])
        ax1.set_ylim([-0.1, 1.2])
        ax1.title.set_text("Chain " + chainids[i])
        ax1.grid(True, color='grey', linestyle='dotted', linewidth=1)
        ax1.legend((line1, line2, ), ('Fullmap', 'Map-model'), loc='center left', bbox_to_anchor=(1, 0.5))
        textstr = '\n'.join(('Average CC:',
                            'Fullmap = %.2f' % (avg_mapcc, ),
                            'Map-model = %.2f' % (avg_mapmodelcc, )))
        ax1.text(minnum, 0.0, textstr, bbox=dict(facecolor='green', alpha=0.2))
        custom_lines = [Line2D([0], [0], color='red', alpha=0.5, lw=4),
                        Line2D([0], [0], color='blue', alpha=0.5, lw=4)]
        ax2 = ax1.twinx()
        ax2.get_yaxis().set_visible(False)
        ax2.legend(custom_lines, ['Helix', 'Sheet'], loc=(0.41, 0.92), ncol=2, frameon=False)
    plt.xlabel(xlabel)
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.55)
    plt.savefig('Residue_correlation_plots.pdf')
    #plt.show()


def find_gap(resnumlist):
    from itertools import chain
    from operator import sub

    a = resnumlist
    print(list(chain.from_iterable((a[i] + d for d in range(1, diff))
                                   for i, diff in enumerate(map(sub, a[1:], a))
                                   if diff > 1)))

def choose_chains_to_plot(
    chainids, 
    chainlist, 
    resnamelist, 
    resnumlist, 
    mapcclist, 
    mapmodelcclist,
    secondary_struc_elements,
    ):
    mapcc_chain_list = []
    mapmodelcc_chain_list = []
    cc_resnum_list = []
    for ichain in chainids:
        mapcc_ichain = [cc for cc, x in zip(
            mapcclist, chainlist) if x == ichain]
        mapmodelcc_ichain = [cc for cc, x in zip(
            mapmodelcclist, chainlist) if x == ichain]
        resnum_ichain = [resnum for resnum, x in zip(
            resnumlist, chainlist) if x == ichain]
        mapcc_chain_list.append(mapcc_ichain)
        mapmodelcc_chain_list.append(mapmodelcc_ichain)
        #print(max(resnum_ichain))  # ERROR
        cc_resnum_list.append(resnum_ichain)
        #plot_cc(residue_cc=cc_ichain, resnum_chain=resnum_ichain)
    plot_ncc(chainids, mapcclist=mapcc_chain_list,
             mapmodelcclist=mapmodelcc_chain_list, 
             resnumlist=cc_resnum_list, 
             secondary_struc=secondary_struc_elements
             )


def secondary_structure(chainids, modelname):
    doc = gemmi.cif.read_file(modelname)
    block = doc.sole_block()
    # helices
    hlxbeg_seq_id = block.find_values("_struct_conf.beg_auth_seq_id")
    hlxend_seq_id = block.find_values("_struct_conf.end_auth_seq_id")
    hlxchain_id = block.find_values("_struct_conf.end_auth_asym_id")
    # sheets
    shtbeg_seq_id = block.find_values("_struct_sheet_range.beg_auth_seq_id")
    shtend_seq_id = block.find_values("_struct_sheet_range.end_auth_seq_id")
    shtchain_id = block.find_values("_struct_sheet_range.end_auth_asym_id")

    hlxbeg_res_list_all = []
    hlxend_res_list_all = []
    shtbeg_res_list_all = []
    shtend_res_list_all = []
    for ichain in chainids:
        hlxbeg_res_list_all.append([br for c, br in zip(list(hlxchain_id), list(hlxbeg_seq_id)) if c == ichain])
        hlxend_res_list_all.append([er for c, er in zip(list(hlxchain_id), list(hlxend_seq_id)) if c == ichain])
        shtbeg_res_list_all.append([br for c, br in zip(list(shtchain_id), list(shtbeg_seq_id)) if c == ichain])
        shtend_res_list_all.append([er for c, er in zip(list(shtchain_id), list(shtend_seq_id)) if c == ichain])
    return [hlxbeg_res_list_all, hlxend_res_list_all, shtbeg_res_list_all, shtend_res_list_all]


def main(halfmap1, halfmap2, modelname, resolution):
    fid = open("EMDA_atomic_cc.txt", "w")
    fid2 = open("EMDA_residue_cc.txt", "w")
    if modelname.endswith((".pdb", ".ent")):
        iotools.pdb2mmcif(modelname)
        modelname = 'out.cif'
    mapcc, mapmodelcc = get_atomic_cc(
        map1=halfmap1, 
        map2=halfmap2, 
        modelname=modelname, 
        resol=float(resolution))
    (chainids, 
    chainlist, 
    resnamelist, 
    resnumlist, 
    mapcclist,
    mapmodelcclist) = get_chains(
        modelname=modelname, 
        xyz_cc_map=mapcc, 
        xyz_cc_mapmodel=mapmodelcc,
        fid=fid, 
        fid2=fid2)
    print('Chain IDs: ', chainids)
    secondary_struc_elements = secondary_structure(chainids, modelname)
    choose_chains_to_plot(chainids, chainlist, resnamelist,
                          resnumlist, mapcclist, mapmodelcclist, secondary_struc_elements) 

def main_helper_for_rcc(uc, rcc_map, rcc_mapmodel, modelname):
    fid = open("EMDA_atomic_cc.txt", "w")
    fid.write("*** Terms and definitions ***\n")
    fid.write("-----------------------------\n")
    fid.write("halfmapCC - Halfmap correlation  values at atomic positions\n")
    fid.write("mapCC     - Fullmap correlation values at atomic positions\n")
    fid.write("mapCC = 2 x halfmapCC / (1 + halfmapCC) [Ref: J.Mol.Biol.,333(4) (2003)]\n")
    fid.write("CC*   = SQRT(mapCC)                     [Ref: JSB, 214(1), (2022)]\n")
    fid.write("\n")
    fid2 = open("EMDA_residue_cc.txt", "w")
    fid2.write("*** Terms and definitions ***\n")
    fid2.write("-----------------------------\n")
    fid2.write("halfmapCC - Halfmap correlation  values at atomic positions\n")
    fid2.write("mapCC     - Fullmap correlation values at atomic positions\n")
    fid2.write("mapCC = 2 x halfmapCC / (1 + halfmapCC) [Ref: J.Mol.Biol.,333(4) (2003)]\n")
    fid2.write("CC*   = SQRT(mapCC)                     [Ref: JSB, 214(1), (2022)]\n")
    fid2.write("\n")
    if modelname.endswith((".pdb", ".ent")):
        iotools.pdb2mmcif(modelname)
        modelname = 'out.cif'
    mapcc, mapmodelcc = get_atomic_cc_helper_for_rcc(
        uc=uc,
        rcc_map=rcc_map, 
        rcc_mapmodel=rcc_mapmodel, 
        modelname=modelname, 
        )
    (chainids, 
    chainlist, 
    resnamelist, 
    resnumlist, 
    mapcclist,
    mapmodelcclist) = get_chains(
        modelname=modelname, 
        xyz_cc_map=mapcc, 
        xyz_cc_mapmodel=mapmodelcc,
        fid=fid, 
        fid2=fid2)
    print('Chain IDs: ', chainids)
    secondary_struc_elements = secondary_structure(chainids, modelname)
    choose_chains_to_plot(chainids, chainlist, resnamelist,
                          resnumlist, mapcclist, mapmodelcclist, secondary_struc_elements)   


if __name__ == "__main__":
    map1, map2, modelname, resol = sys.argv[1:]
    main(map1, map2, modelname, resol)
    """ fid = open("EMDA_atomic_cc.txt", "w")
    fid2 = open("EMDA_residue_cc.txt", "w")
    if modelname.endswith((".pdb", ".ent")):
        iotools.pdb2mmcif(modelname)
        modelname = 'out.cif'
    mapcc, mapmodelcc = get_atomic_cc(map1, map2, modelname, float(resol))
    chainids, chainlist, resnamelist, resnumlist, mapcclist = get_chains(
        modelname, mapcc)
    chainids, chainlist, resnamelist, resnumlist, mapmodelcclist = get_chains(
        modelname, mapmodelcc)
    print('Chain IDs: ', chainids)
    secondary_struc_elements = secondary_structure(chainids, modelname)
    choose_chains_to_plot(chainids, chainlist, resnamelist,
                          resnumlist, mapcclist, mapmodelcclist, secondary_struc_elements) """

    
