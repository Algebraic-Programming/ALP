#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import copy
import sys

def gen_golden_data():
    x1_gm_type = np.float16

    n0=int(sys.argv[1])
    n1=int(sys.argv[2])
    n2=int(sys.argv[3])
    
    M = n0*n1
    N = n2

    x1_gm = np.random.randint(1, 10, [M, N]).astype(x1_gm_type)

    S = copy.copy(x1_gm)

    # S=softmax(S)
    # rowmaxS=np.max(S,axis=1)
    # S=S-np.tile(rowmaxS, (np.shape(S)[1],1)).T
    # S=np.exp(S)
    # rowsumS=1/np.sum(S,axis=1)
    # S=S*(np.tile(rowsumS, (np.shape(S)[1],1)).T)

    golden = S

    infilename = "./input/input0.bin"
    outfilename = "./output/golden.bin"

    x1_gm.tofile( infilename )
    golden.tofile( outfilename )

    print(f"I/O of size {M} x {N} and type {x1_gm_type} generated in {infilename} and {outfilename}")
    # print( golden )

if __name__ == "__main__":
    assert(len(sys.argv)==4)
    gen_golden_data()
