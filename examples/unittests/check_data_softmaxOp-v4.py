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

    # outfilename = "./output/output.bin"
    outfiles=glob.glob("./output/param1.bin")
    # sort outfiles
    if(len(outfiles)>1):
        ii=[ int(re.search(r'\d+', fname).group()) for fname in outfiles ]
        outfiles=np.array(outfiles)[np.argsort(ii)]

    n0=int(sys.argv[1])
    n1=int(sys.argv[2])
    n2=int(sys.argv[3])
    n3=int(sys.argv[4])
    n4=int(sys.argv[5])
    n5=int(sys.argv[6])

    N = n0*n1*n2*n3*n4*n5
    shape1=(n0,n1,n2,n3,n4,n5)

    dtype = np.float16

    # printblocks=[(0,0),(0,n1-1),(n0-1,n1-1),(n0-1,0)]
    
    golden = np.fromfile( goldenfilename, dtype=dtype )

    # print(f"Golden: {golden[:10]}")
    # print(f"Output: {output[:10]}")


    print("Golden:")
    reshaped_golden = np.reshape(golden, shape1 )        
    
    for outfilename in outfiles:
        
        output = np.fromfile( outfilename, dtype=dtype )
        print("Output:",outfilename)
        reshaped_output = np.reshape(output, shape1 )


        # for i0 in range(n0):
        #     for i1 in range(n1):
        #         for i2 in range(n2):
        #             for i3 in range(n3):
        #                 for i4 in range(n4):
        #                     tmp_diff=np.linalg.norm(reshaped_output[i0,i1,i2,i3,i4,:]-reshaped_golden[i0,i1,i2,i3,i4,:])
        #                     print(i0,i1,i2,i3,i4,
        #                           " d=",tmp_diff,
        #                           "  o=",reshaped_output[i0,i1,i2,i3,i4,:2],".",
        #                           "  g=",reshaped_golden[i0,i1,i2,i3,i4,:2],".")

        
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
    assert(len(sys.argv)==7)
    check_golden_data()
