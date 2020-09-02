"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# this code generates the mask if not supplied.
# then ally that mask to arr.
from __future__ import absolute_import, division, print_function, unicode_literals
from emda import core, ext
import numpy as np


class MaskedMaps:
    def __init__(self, hfmap_list=None):
        self.hfmap_list = hfmap_list
        self.mask = None

    def read_halfmaps(self):
        for n in range(0, len(self.hfmap_list), 2):
            uc, arr1, origin = core.iotools.read_map(self.hfmap_list[n])
            uc, arr2, origin = core.iotools.read_map(self.hfmap_list[n + 1])
        self.uc = uc
        self.arr1 = arr1
        self.arr2 = arr2
        self.origin = origin

    def generate_mask(self, arr1, arr2, smax=9, iter=1, threshold=None):
        kern = core.restools.create_soft_edged_kernel_pxl(
            smax
        )  # sphere with radius of n pixles
        _, fullcc3d = ext.realsp_local.get_3d_realspcorrelation(arr1, arr2, kern)
        if threshold is None:
            cc_mask, threshold = self.histogram(fullcc3d)
        else:
            cc_mask, _ = self.histogram(fullcc3d)
        print("threshold: ", threshold)
        mask = fullcc3d * (fullcc3d >= threshold)
        # dilate and softened the mask
        mask = make_soft(binary_dilation_ccmask(mask * cc_mask, iter))
        mask = mask * (mask >= 0.0)
        self.mask = mask

    def create_edgemask(self, radius):
        # Remove everything outside radius
        box_radius = radius + 1
        box_size = radius * 2 + 1
        # Creating a sphere mask
        center = [box_radius, box_radius, box_radius]
        print("boxsize: ", box_size, "boxradius: ", box_radius, "center:", center)
        radius = box_radius
        X, Y, Z = np.ogrid[:box_size, :box_size, :box_size]
        dist_from_center = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )
        mask = dist_from_center <= radius
        return mask

    def histogram(self, arr1):
        from scipy import stats

        nx, ny, nz = arr1.shape
        maxbin = np.amax(np.array([nx // 2, ny // 2, nz // 2]))
        counts = stats.binned_statistic(
            arr1.flatten(), arr1.flatten(), statistic="count", bins=maxbin, range=(0, 1)
        )[0]
        sc = counts - np.roll(counts, 1)
        ulim = len(sc) - 11 + np.argmax(sc[-10:])
        cc_arr = np.linspace(0, 1, maxbin)
        xc = cc_arr[:ulim] * counts[:ulim]
        xc_sum = np.sum(xc)
        isum = 0.0
        cc_arr = cc_arr[:ulim]
        for i in range(len(xc)):
            isum = isum + xc[i]
            if isum >= xc_sum / 2:
                threshold = cc_arr[i]
                break
        edge_mask = self.create_edgemask(ulim)
        cc_mask = np.zeros(shape=(nx, ny, nz), dtype="bool")
        cx, cy, cz = edge_mask.shape
        dx = (nx - cx) // 2
        dy = (ny - cy) // 2
        dz = (nz - cz) // 2
        print(dx, dy, dz)
        cc_mask[dx : dx + cx, dy : dy + cy, dz : dz + cz] = edge_mask
        return cc_mask, threshold

    def get_radial_sum(self, arr1):
        # this function not used
        from matplotlib import pyplot as plt

        nx, ny, nz = arr1.shape
        nbin, res_arr, bin_idx = core.restools.get_resolution_array(self.uc, arr1)
        sum_lst = []
        ibin_lst = []
        isum = 0.0
        isum_old = 0.0
        for ibin in range(nbin):
            ibin_sum = np.sum(arr1 * (bin_idx == ibin))
            ibin_lst.append(ibin_sum)
            isum_old = isum
            isum = isum + ibin_sum
            # if ibin_sum < 0.0 and ibin > 10:
            if isum <= isum_old and ibin > 10:
                break
            print(ibin, ibin_sum)
            sum_lst.append(isum)
        edge_mask = self.create_edgemask(ibin)
        cc_mask = np.zeros(shape=(nx, ny, nz), dtype="bool")
        cx, cy, cz = edge_mask.shape
        dx = (nx - cx) // 2
        dy = (ny - cy) // 2
        dz = (nz - cz) // 2
        print(dx, dy, dz)
        cc_mask[dx : dx + cx, dy : dy + cy, dz : dz + cz] = edge_mask
        core.iotools.write_mrc(arr1 * cc_mask, "maskedcc_map.mrc", self.uc, self.origin)
        plt.plot(ibin_lst, "r")
        plt.show()


def binary_dilation_ccmask(ccmask, iter=1):
    from scipy.ndimage.morphology import binary_dilation

    return binary_dilation(ccmask, iterations=iter).astype(ccmask.dtype)


def make_soft(dilated_mask, kern_rad=3):
    # convoluting with gaussian shere
    import scipy.signal

    kern_sphere = core.restools.create_soft_edged_kernel_pxl(kern_rad)
    return scipy.signal.fftconvolve(dilated_mask, kern_sphere, "same")


def mapmask(arr, uc, itr=3, kern_rad=3, prob=0.99):
    from scipy.ndimage.morphology import binary_dilation
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    arr_tmp = arr
    X2 = np.sort(arr.flatten())
    F2 = np.array(range(len(X2))) / float(len(X2) - 1)
    loc = np.where(F2 >= prob)
    thresh = X2[loc[0][0]]
    thresh = max([thresh, np.max(X2) * 0.02])
    print("threshold: ", thresh)
    # plot the sorted data:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(F2, X2)
    ax1.set_xlabel("$p$")
    ax1.set_ylabel("$x$")
    ax2 = fig.add_subplot(122)
    ax2.plot(X2, F2)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$p$")
    plt.savefig("cdf.png", format="png", dpi=300)
    binary_arr = (arr_tmp > thresh).astype(int)
    dilate = binary_dilation(binary_arr, iterations=itr)
    mask = make_soft(dilate, kern_rad)
    mask = mask * (mask >= 0.0)
    return mask


if __name__ == "__main__":
    maplist = [
        "/Users/ranganaw/MRC/REFMAC/Bianka/EMD-4572/other/run_half1_class001_unfil.mrc",
        "/Users/ranganaw/MRC/REFMAC/Bianka/EMD-4572/other/run_half2_class001_unfil.mrc",
    ]
    obj = MaskedMaps(maplist)
    obj.read_halfmaps()
    obj.generate_mask(obj.arr1, obj.arr2)
