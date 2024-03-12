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
    n3=int(sys.argv[4])
    
    M = n0*n1
    N = n2*n3

    x1_gm = np.random.randint(1, 10, [N, M]).astype(x1_gm_type)

    S = copy.copy(x1_gm)
    S = np.reshape(S,(n0,n1,n2,n3))

    # S=block_softmax(S)
    for i0 in range(n0):
        for i1 in range(n1):
            Stmp=S[i0,i1,:,:]
            rowmaxS=np.max(Stmp,axis=1)
            Stmp=Stmp-np.tile(rowmaxS, (np.shape(Stmp)[1],1)).T
            Stmp=np.exp(Stmp)
            rowsumS=1/np.sum(Stmp,axis=1)
            Stmp=Stmp*(np.tile(rowsumS, (np.shape(Stmp)[1],1)).T)
            S[i0,i1,:,:]=Stmp

    golden = S

    infilename = "./input/input0.bin"
    outfilename = "./output/golden.bin"

    x1_gm.tofile( infilename )
    golden.tofile( outfilename )

    print(f"I/O of size {n0} x {n1} x {n2} x {n3} and type {x1_gm_type} generated in {infilename} and {outfilename}")
    # print( golden )

if __name__ == "__main__":
    assert(len(sys.argv)==5)
    gen_golden_data()
