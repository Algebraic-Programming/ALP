#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import copy

def gen_golden_data():
    x1_gm_type = np.float16

    n0=16
    n1=32
    n2=16
    n3=16
    n4=16

    N1 = n0*n1*n2*n3
    shape1 = (n0,n1,n2,n3)

    S0_gm = np.random.randint(1, 10, [N1]).astype(x1_gm_type)
    infilename = "./input/s0_gm.bin"
    S0_gm.tofile( infilename )



if __name__ == "__main__":
    gen_golden_data()
