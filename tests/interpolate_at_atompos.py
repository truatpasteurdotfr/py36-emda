# EMDA testcode to get atomic correlation values by interpolation
# Author: Rangana Warshamange [ranganaw@mrc-lmb.cam.ac.uk]
# v1 - 13.08.2021; v2 - 29.08.2022

import numpy as np
import gemmi
import emda.emda_methods as em
import argparse


def interpolate_cc(data):
    from scipy.interpolate import RegularGridInterpolator

    linx = np.linspace(0, data.shape[0], data.shape[0])
    liny = np.linspace(0, data.shape[1], data.shape[1])
    linz = np.linspace(0, data.shape[2], data.shape[2])
    return RegularGridInterpolator(
        (linx, liny, linz), data, method="nearest")

def get_chains(modelname, ligand_id, pixsize):
    st = gemmi.read_structure(modelname)
    ligdic = {}
    for chain in st[0]:
        for residue in chain:
            for atom in residue:
                if chain.name == 'A' and residue.name == ligand_id:
                    x = float(atom.pos.x) / pixsize
                    y = float(atom.pos.y) / pixsize
                    z = float(atom.pos.z) / pixsize
                    ligdic[atom.name] = [x, y, z]
    return ligdic

def plot_all(cc_list, lbls, ligand):
    import matplotlib
    matplotlib.use(matplotlib.get_backend())
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)
    color_list = ['blue', 'green', 'red', 'orange', 'purple', 'yellow', 'magenta']
    lbls = ['CC_wt-mel', 'CC_wt-pent', 'CC_mel-pent']
    for cc, lbl, col in zip(cc_list, lbls, color_list):
        ax1.plot(cc, marker='o', color=col, label=lbl)
    x_ticks = np.arange(0, len(cc)+1, 5)
    xminor_ticks = np.arange(0, len(cc)+1, 1)
    y_ticks = np.arange(0, 1.1, 0.1)
    ax1.set_xticks(x_ticks)
    ax1.set_xticks(xminor_ticks, minor=True)
    ax1.set_yticks(y_ticks)
    ax1.grid(which='both')
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    ax1.set_xlabel("Atom label", fontweight='bold')
    ax1.set_ylabel("Correlation", fontweight='bold')
    plt.legend()
    plt.title('Atomic CC of ligand %s' %ligand)
    plotname = "atomic_cc_%s.png" %ligand
    plt.savefig(plotname, format="png", dpi=300)
    plt.show()

def vec2string(vec):
    return " ".join(("% .3f"%x for x in vec))

def list2string(lis):
    return " ".join(("%s"%x for x in lis))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Atomic CC values')
    parser.add_argument('--maplist', metavar='', required=True, type=str, nargs='+',
                        help='list of maps for')
    parser.add_argument('--labels', metavar='', required=False, type=str, nargs='+',
                        help='list of labels for data')
    parser.add_argument('--model', metavar='', required=True, type=str,
                        help='model name pdb/cif')
    parser.add_argument('--ligand', metavar='', required=True, type=str,
                        help='ligand-id')  
    args = parser.parse_args() 

    cc_list = []
    for i, imap in enumerate(args.maplist):
        temp = []
        uc, ccarr, orig = em.get_data(imap)
        if i == 0:
            cell = uc
            pixsize = cell[0] / ccarr.shape[0]
            ligdic = get_chains(modelname=args.model, 
                                ligand_id=args.ligand, pixsize=pixsize)
        cc_interpolator = interpolate_cc(data=ccarr)
        for item in ligdic.items():
            temp.append(cc_interpolator(tuple(reversed(item[1]))))
        cc_list.append(temp)

    if args.labels is None :
        labels = ['ccmap_%i'%i for i in range(len(args.maplist))]
    else:
        labels = args.labels
    filename = '%s-emda_ligandCC.txt'%args.ligand
    fid = open(filename, "w")
    print("\n**** Atomic CC values for Ligand %s ****\n" % args.ligand)
    print('Atm#  AtmID ', list2string(labels))
    fid.write("**** Atomic CC values for Ligand %s ****\n" % args.ligand)
    fid.write('Atm#  AtmID  %s\n' %list2string(labels))
    for i, item in enumerate(ligdic.items()):
        print(
            "%i  %s  %s" %(i, item[0], vec2string([cc_list[j][i] for j in range(len(cc_list))]))
        )
        fid.write(
            "%i  %s  %s\n" %(i, item[0], vec2string([cc_list[j][i] for j in range(len(cc_list))]))
        )
    plot_all(cc_list, labels, args.ligand)