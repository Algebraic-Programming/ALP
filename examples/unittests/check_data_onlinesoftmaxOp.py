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

    n0=int(sys.argv[1])
    n1=int(sys.argv[2])
    n2=int(sys.argv[3])
    n3=int(sys.argv[4])
    n4=16

    shape1=(n0,n2,n4)
    shape2=(n1,n3,n4)
    shape3=(n0,n2)
    shape4=(n0,n1,n2,n3)

    dtype = np.float16

    #############################################

    outfilename="output/m0_golden.bin"
    golden = np.fromfile( outfilename, dtype=dtype )
    print("Golden:",outfilename)
    reshaped_golden = np.reshape(golden, shape3 )

    print("golden m")
    # for i0 in range(n0):
    #     print(i0,", ".join(reshaped_golden[i0,:].astype(str)))

    outfilename="output/param0.bin"
    output = np.fromfile( outfilename, dtype=dtype )
    print("Output:",outfilename)
    reshaped_output = np.reshape(output, shape3 )

    # print("output m")
    # for i0 in range(n0):
    #     print(i0,", ".join(reshaped_output[i0,:].astype(str)))

    # print("output - golden m")
    # for i0 in range(n0):
    #     print(i0,", ".join((reshaped_golden[i0,:]-reshaped_output[i0,:]).astype(str)))

    norm_diff = np.linalg.norm( ( output.astype(float) - golden.astype(float) ) )
    check = check and ( norm_diff < 1.e-4 )
    print(f"DiffNorm(m) {outfilename,} (size {shape3} and type {dtype}): absolute = {norm_diff}")

    #############################################

    outfilename="output/l0_golden.bin"
    golden = np.fromfile( outfilename, dtype=dtype )
    print("Golden:",outfilename)
    reshaped_golden = np.reshape(golden, shape3 )

    # print("golden l")
    # for i0 in range(n0):
    #     print(i0,", ".join(reshaped_golden[i0,:].astype(str)))

    outfilename="output/param1.bin"
    output = np.fromfile( outfilename, dtype=dtype )
    print("Output:",outfilename)
    reshaped_output = np.reshape(output, shape3 )

    # print("output l")
    # for i0 in range(n0):
    #     print(i0,", ".join(reshaped_output[i0,:].astype(str)))

    norm_diff = np.linalg.norm( ( output.astype(float) - golden.astype(float) ) )
    norm_relative = norm_diff / np.linalg.norm( golden.astype(float) )
    check = check and ( norm_relative < 1.e-2 )
    print(f"DiffNorm(l) {outfilename,} (size {shape3} and type {dtype}): absolute = {norm_diff}, relative = {norm_relative}")


    outfilename="output/param3.bin"
    output = np.fromfile( outfilename, dtype=dtype )

    outfilename="output/s1_golden.bin"
    golden = np.fromfile( outfilename, dtype=dtype )


    reshaped_output = np.reshape(output, shape4 )
    reshaped_golden = np.reshape(golden, shape4 )
    # for i0 in range(n0):
    #     for i1 in range(n1):
    #         nrm1=np.linalg.norm( ( reshaped_output[i0,i1,:,:].astype(float) ) )
    #         nrm2=np.linalg.norm( ( reshaped_golden[i0,i1,:,:].astype(float) ) )
    #         nrm3=np.linalg.norm( ( reshaped_output[i0,i1,:,:].astype(float) - reshaped_golden[i0,i1,:,:].astype(float) ) )
    #         print(i0,i1,nrm1,nrm2,nrm3)
    #         # if(i0==0 and i1 == 0):
    #         #     print("o","i0 =",i0,"i1=",i1)
    #         #     for i2 in range(n2):
    #         #         print(i2,", ".join(reshaped_output[i0,i1,i2,:].astype(str)))
    #         #     print("g","i0 =",i0,"i1=",i1)
    #         #     for i2 in range(n2):
    #         #         print(i2,", ".join(reshaped_golden[i0,i1,i2,:].astype(str)))

    norm_output = np.linalg.norm( ( output.astype(float) ) )
    norm_golden = np.linalg.norm( ( golden.astype(float) ) )

    norm_diff = np.linalg.norm( ( output.astype(float) - golden.astype(float) ) )
    norm_relative = norm_diff / np.linalg.norm( ( golden.astype(float) ) )
    check = check and ( norm_relative < 1.e-4 )
    print(f"DiffNorm(s1) {outfilename,} (size {shape4} and type {dtype}): absolute = {norm_diff}, relative = {norm_relative}")


    #############################################

    # outfilename="output/output_o.bin"
    # output = np.fromfile( outfilename, dtype=dtype )
    # print("Output:",outfilename)
    # reshaped_output = np.reshape(output, shape1 )
    # norm_diff = np.linalg.norm( ( output.astype(float) ) )
    # print(f"Norm {outfilename} (size {shape1} and type {dtype}): absolute = {norm_diff}")


    # K = np.reshape( np.fromfile( "input/k_gm.bin", dtype=dtype ), (n1*n3,n4) )
    # Q = np.reshape( np.fromfile( "input/q_gm.bin", dtype=dtype ), (n0*n2,n4) )
    # V = np.reshape( np.fromfile( "input/v_gm.bin", dtype=dtype ), (n4,n1*n3) )


    # O=Q.dot(K.T).dot(V.T)
    # oflat=np.reshape( O, (n0*n2*n4) )
    # print(output.astype(float))
    # print(oflat.astype(float))


    # for outfilename in outfiles:
    #     output = np.fromfile( outfilename, dtype=dtype )
    #     print("Output:",outfilename)
    #     reshaped_output = np.reshape(output, shape3 )
    #     for i in range(n0):
    #         print(i,reshaped_output[i,:])
    #     norm_diff = np.linalg.norm( (golden.astype(float) - output.astype(float)) )
    #     norm_diff_relative = norm_diff/np.linalg.norm( golden.astype(float) )
    #     print(f"Comparing in {goldenfilename} and {outfilename} (size {np.shape(golden)} and type {dtype}): absolute = {norm_diff}, relative = {norm_diff_relative}")

    if(check):
        print(bcolors.OKGREEN + "Test OK!" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Test Failed" + bcolors.ENDC)
        
        sys.exit(1)




if __name__ == "__main__":
    assert(len(sys.argv)==5)
    check_golden_data()
