# various functions
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def regularize_fsc_weights(fsc):
    for i, elm in enumerate(fsc):
        if elm <= 1e-5:
            elm = elm + 1e-4 
            fsc[i] = elm
    return fsc

def get_avg_fsc(binfsc, bincounts):
    fsc_filtered = filter_fsc(bin_fsc=binfsc, thresh=0.)
    fsc_avg = np.average(a=fsc_filtered[np.nonzero(fsc_filtered)], 
                         weights=bincounts[np.nonzero(fsc_filtered)])
    return fsc_avg

def filter_fsc(bin_fsc, thresh=0.1):
    bin_fsc_new = np.zeros(bin_fsc.shape, 'float')
    for i, ifsc in enumerate(bin_fsc):
        if ifsc >= thresh:
            bin_fsc_new[i] = ifsc
        else:
            if i > 1:
                break
    return bin_fsc_new