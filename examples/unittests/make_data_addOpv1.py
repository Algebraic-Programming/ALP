#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import sys

def gen_golden_data_simple():

    n0=int(sys.argv[1])

    N = n0

    input_x = np.random.uniform(-100, 100, N ).astype(np.float16)
    input_y = np.random.uniform(-100, 100, N ).astype(np.float16)
    golden = (input_x + input_y).astype(np.float16)

    input_x.tofile("./input/input0.bin")
    input_y.tofile("./input/input1.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    assert(len(sys.argv)==2)
    gen_golden_data_simple()
