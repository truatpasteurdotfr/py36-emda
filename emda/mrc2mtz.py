import numpy as np
import argparse
import sys
from emda.iotools import read_map,write_3d2mtz
from emda.config import *

# MRC to MTZ conversion

cmdl_parser = argparse.ArgumentParser(description='MRC to MTZ conversion\n')
cmdl_parser.add_argument('-i', '--inmap', required=True, help='input map')
cmdl_parser.add_argument('-o', '--outmtz', required=True, help='outputmtz')

def main():
    args = cmdl_parser.parse_args()
    uc,ar1,origin = read_map(args.inmap)
    hf1 = np.fft.fftshift(np.fft.fftn(ar1))
    write_3d2mtz(uc,hf1,outfile=args.outmtz+'.mtz')

if(__name__ == "__main__"):
    main()

