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
    n4=16

    N0 = n0*n2*n4
    N1 = n1*n3*n4

    shape0 = (n0,n2,n4)
    shape1 = (n1,n3,n4)
    shape2 = (n0,n1,n2,n3)
    shape3 = (n0,n2)

    Q_gm = np.random.randint(1, 4, [N0]).astype(x1_gm_type)
    V_gm = np.random.randint(1, 4, [N1]).astype(x1_gm_type)
    K_gm = np.random.randint(1, 4, [N1]).astype(x1_gm_type)

    Q = copy.copy(Q_gm)
    Q = np.reshape(Q,shape0)
    #infilename = "./input/q_gm.bin"
    #Q_gm.tofile( infilename )

    K = copy.copy(K_gm)
    K = np.reshape(K,shape1)
    #infilename = "./input/k_gm.bin"
    #K_gm.tofile( infilename )

    V = copy.copy(V_gm)
    V = np.reshape(V,shape1)
    #infilename = "./input/v_gm.bin"
    #V_gm.tofile( infilename )


    S0 = np.zeros(shape2).astype(x1_gm_type)
    for i0 in range(n0):
        for i1 in range(n1):
            S0[i0,i1,:,:]=Q[i0,:,:].dot(V[i1,:,:].T)
    S0_gm = np.reshape(S0,n0*n1*n2*n3)
    #infilename = "./input/s0_gm.bin"
    infilename = "./input/input2.bin"
    S0_gm.tofile( infilename )

    m0 = np.zeros(shape3).astype(x1_gm_type)
    l0 = np.zeros(shape3).astype(x1_gm_type)
    S1 = np.zeros(shape2).astype(x1_gm_type)
    for i0 in range(n0):
        l0[i0,:]=0
        m0[i0,:]=-65504.0
        for i1 in range(n1):
            Sij=S0[i0,i1,:,:]
            # Pi=Sij
            mi_old=copy.copy(m0[i0,:])

            rowmaxS=np.max(Sij,axis=1)
            m0[i0,:]=np.maximum(m0[i0,:],rowmaxS) # m0[i0,:]=rowmaxS  # TEMP
            mi_bcast=np.tile(m0[i0,:], (np.shape(Sij)[1],1))
            Pi=Sij-mi_bcast.T
            Pi=np.exp(Pi)

            expmidiff=np.exp(mi_old-m0[i0,:])
            l0[i0,:]*=expmidiff
            l0[i0,:]+=np.sum(Pi,axis=1)

            S1[i0,i1,:,:]=Pi




    # print("m0=",m0)
    # print("l0=",l0)
    m0_gm = np.reshape(m0,n0*n2)
    l0_gm = np.reshape(l0,n0*n2)
    s1_gm = np.reshape(S1,n0*n1*n2*n3)
    goldenfilename = "./output/m0_golden.bin"
    m0_gm.tofile( goldenfilename )
    goldenfilename = "./output/l0_golden.bin"
    l0_gm.tofile( goldenfilename )
    goldenfilename = "./output/s1_golden.bin"
    s1_gm.tofile( goldenfilename )

    S2 = np.zeros(shape2).astype(x1_gm_type)
    for i0 in range(n0):
        for i1 in range(n1):
            S2[i0,i1,:,:]=Q[i0,:,:].dot(V[i1,:,:].T)
    S2_gm = np.reshape(S2,n0*n1*n2*n3)
    infilename = "./input/s2_gm.bin"
    S2_gm.tofile( infilename )

    # S = copy.copy(x1_gm)
    # S = np.reshape(S,(n0,n1,n2,n3))

    # # S=block_softmax(S)
    # for i0 in range(n0):
    #     for i1 in range(n1):
    #         Stmp=S[i0,i1,:,:]
    #         rowmaxS=np.max(Stmp,axis=1)
    #         Stmp=Stmp-np.tile(rowmaxS, (np.shape(Stmp)[1],1)).T
    #         Stmp=np.exp(Stmp)
    #         rowsumS=1/np.sum(Stmp,axis=1)
    #         Stmp=Stmp*(np.tile(rowsumS, (np.shape(Stmp)[1],1)).T)
    #         S[i0,i1,:,:]=Stmp

    # golden = S

    # outfilename = "./output/golden.bin"
    # golden.tofile( outfilename )

    # print(f"I/O of size {n0} x {n1} x {n2} x {n3} and type {x1_gm_type} generated in {infilename} and {outfilename}")
    # # print( golden )

if __name__ == "__main__":
    assert(len(sys.argv)==5)
    gen_golden_data()
