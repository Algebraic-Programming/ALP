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
    n4=int(sys.argv[5])
    n5=int(sys.argv[6])

    N = n0*n1*n2*n3*n4*n5

    x1_gm = np.random.randint(1, 10, [N]).astype(x1_gm_type)

    S = copy.copy(x1_gm)

    S=np.reshape(S,(n0,n1,n2,n3,n4,n5))
    for i0 in range(n0):
        for i1 in range(n1):
            for i3 in range(n3):
                for i4 in range(n4):
                    Stmp=S[i0,i1,:,i3,i4,:]
                    rowmaxStmp=np.max(Stmp,axis=1)
                    Stmp=Stmp-np.tile(rowmaxStmp, (np.shape(Stmp)[1],1)).T
                    Stmp=np.exp(Stmp)
                    rowsumStmp=1/np.sum(Stmp,axis=1)
                    Stmp=Stmp*(np.tile(rowsumStmp, (np.shape(Stmp)[1],1)).T)
                    S[i0,i1,:,i3,i4,:]=Stmp


    golden = np.reshape(S,(n0*n1*n2*n3*n4*n5))

    infilename = "./input/input0.bin"
    outfilename = "./output/golden.bin"

    x1_gm.tofile( infilename )
    golden.tofile( outfilename )

    print(f"I/O of size {n0} x {n1} x {n2} x {n3} x {n4} x {n5} and type {x1_gm_type} generated in {infilename} and {outfilename}")
    # print( golden )


if __name__ == "__main__":
    assert(len(sys.argv)==7)
    gen_golden_data()
