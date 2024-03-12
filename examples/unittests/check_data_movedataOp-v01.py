#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import copy
import glob
import re
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def check_golden_data():
    check = True
    tol = 1.e-2

    goldenfilename = "./output/golden.bin"

    outfiles=glob.glob("./output/param1.bin")
    # sort outfiles
    if(len(outfiles)>1):
        ii=[ int(re.search(r'\d+', fname).group()) for fname in outfiles ]
        outfiles=np.array(outfiles)[np.argsort(ii)]

    n0=int(sys.argv[1])
    n1=int(sys.argv[2])
    n2=int(sys.argv[3])

    M = n0*n1
    N = n2

    dtype = np.float16
    print("(N,M)=",N,M)
    golden = np.fromfile( goldenfilename, dtype=dtype )

    # print(f"Golden: {golden[:10]}")
    # print(f"Output: {output[:10]}")

    print("Golden:")
    reshaped_golden = np.reshape(golden, (M, N) )
    for pos, row in enumerate( reshaped_golden[[0,1,-2,-1]] ):
        print(f"{pos}: {row}")

    
    for outfilename in outfiles:
        
        output = np.fromfile( outfilename, dtype=dtype )
        

        print("Output:",outfilename)
        reshaped_output = np.reshape(output, (M, N) )
        for pos, row in enumerate( reshaped_output[[0,1,-2,-1],:20] ):
            print(f"{pos}: {row}")

        # diff = (golden.astype(float) - output.astype(float))**2
        # diff = np.cumsum((diff.flatten()))
        # print("Diff**2:")
        # reshaped_output = np.reshape(diff, (M, N) )
        # for pos, row in enumerate( reshaped_output[[0,1,-2,-1]] ):
        #     print(f"{pos}: {row}")

        norm_diff = np.linalg.norm( (golden.astype(float) - output.astype(float)) )
        norm_diff_relative=norm_diff/np.linalg.norm( golden.astype(float) )
        print(f"Comparing in {goldenfilename} and {outfilename} (size {np.shape(golden)} and type {dtype}): absolute = {norm_diff} : relative = {norm_diff_relative} ")
        check = check and (norm_diff_relative<tol)
    if(check):
        print(bcolors.OKGREEN + "Test OK!" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Test Failed" + bcolors.ENDC)
        
        sys.exit(1)


if __name__ == "__main__":
    assert(len(sys.argv)==4)
    check_golden_data()
