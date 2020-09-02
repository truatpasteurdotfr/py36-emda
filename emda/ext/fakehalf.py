# Neighbor averaging method based fake half map
def fakehalf(f_map, knl=3):
    import numpy as np
    from emda import core
    import scipy.signal

    # kernel = np.ones((knl, knl, knl), dtype=int)
    # kernel[1,1,1] = 0
    # kernel = kernel/np.sum(kernel)
    # box_size = knl
    # kernel = np.ones((box_size, box_size, box_size)) / (box_size ** 3)
    # kernel[1,1,1] = 0
    kernel = core.restools.create_soft_edged_kernel_pxl(knl)
    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2] = 0
    kernel = kernel / np.sum(kernel)
    fakehalf = scipy.signal.fftconvolve(f_map, kernel, "same")
    return fakehalf
