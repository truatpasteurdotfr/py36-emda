import numpy as np
import argparse
import sys
from emda.iotools import read_map,write_mrc
from emda.maptools import mtz2map
from emda.config import *

# MRC to MTZ conversion

cmdl_parser = argparse.ArgumentParser(description='MTZ to MRC conversion\n')
cmdl_parser.add_argument('-i', '--inmtz', required=True, help='input mtz')
cmdl_parser.add_argument('-m', '--inmap', required=True, help='input map')
cmdl_parser.add_argument('-o', '--outmap', required=True, help='output mrc')

def main():
    args = cmdl_parser.parse_args()
    uc,ar,origin = read_map(args.inmap)
    dat = mtz2map(args.inmtz,ar.shape)
    outfile=args.outmap+'.mrc'
    write_mrc(dat,outfile,uc,origin)

if(__name__ == "__main__"):
    main()

