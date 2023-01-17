""" Calculate FSC between onemap against several other maps

Specific example code for EMDA-tutorial on EMDA paper

Run the code as:
$python fsc_nmaps.py reference.mrc initial_map.mrc adjusted_map.mrc"""

import sys
import numpy as np
import emda.emda_methods as em
from emda.core import plotter, fsc, restools

def plot_nlines(
    res_arr,
    list_arr,
    mapname="halfmap_fsc.eps",
    curve_label=None,
    fscline=0.143,
    plot_title=None,
):
    from matplotlib import pyplot as plt

    font = {"family": "sans-serif", "color": "black", "weight": "normal", "size": 16}
    if curve_label is None:
        curve_label = ["halfmap_fsc"]
    bin_arr = np.arange(len(res_arr))
    fig = plt.figure(figsize=(7.5, 5.3))
    ax1 = fig.add_subplot(111)
    for icurve in range(len(list_arr)):
        ax1.plot(bin_arr, list_arr[icurve], label=curve_label[icurve], linewidth=2)
    xmin = np.min(bin_arr)
    xmax = np.max(bin_arr)
    plt.plot(
        (xmin, xmax), (float(fscline), float(fscline)), color="gray", linestyle=":"
    )
    plt.plot((xmin, xmax), (0.0, 0.0), color="black", linestyle=":")
    pos = np.array(ax1.get_xticks(), dtype=int)
    n_bins = res_arr.shape[0]
    pos[pos < 0] = 0
    pos[pos >= n_bins] = n_bins - 1
    ax1.set_xticklabels(np.round(res_arr[pos], decimals=2), fontdict=font)
    ax1.set_xlabel("Resolution ($\AA$)", fontdict=font)
    ax1.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax1.set_yticklabels(labels=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), fontdict=font)
    plt.legend(loc=1, prop={'size': 16})
    plt.ylabel("Fourier Shell Correlation", fontdict=font)
    
    if plot_title is not None:
        plt.title(plot_title, fontdict=font)
    plt.savefig(mapname, format="eps", dpi=300)
    plt.close()


if __name__ == "__main__":

    ref, initial, adjusted = sys.argv[1:]
    maplist = [ref, initial, adjusted]
    nmaps = len(maplist)
    fsc_list = []
    uc, arr1, _ = em.get_data(maplist[0])
    f1 = np.fft.fftshift(np.fft.fftn(arr1))
    nbin, res_arr, bin_idx = restools.get_resolution_array(uc, f1)
    idx = np.argmin((res_arr - 1.75) ** 2)
    for i in range(1, nmaps):
        _, arr2, _ = em.get_data(maplist[i])
        f2 = np.fft.fftshift(np.fft.fftn(arr2))
        bin_fsc = fsc.anytwomaps_fsc_covariance(f1, f2, bin_idx, nbin)[0]
        fsc_list.append(bin_fsc[:idx])

    plot_nlines(
        res_arr=res_arr[:idx],
        list_arr=fsc_list,
        mapname="fsc_plot.eps",
        curve_label=[
                    "ref. vs initial map",
                    "ref. vs adjusted map",
                    ],
        fscline=0.0,
        plot_title="FSC calculated against the reference map"
    )
