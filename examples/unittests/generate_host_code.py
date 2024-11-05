#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import copy
import re
import argparse

host_code_template=""

class Tensor:
    """A simple tensor class"""
    def __init__(self,grid,axes,inout,tid):
        assert( inout=="in" or inout=="out" )
        self.grid=grid
        self.axes=axes
        self.inout=inout
        self.tid=tid

        self.paramName="param"+str(tid)+inout
        self.paramHostname=self.paramName+"Host"
        self.paramDevicename=self.paramName+"Device"
        self.paramFileSize=self.paramName+"FileSize"

        if( inout=="out" ):
            self.paramFileNameOut='"./output/param'+str(tid)+'.bin"'
        else:
            self.paramFileNameIn='"./input/input'+str(tid)+'.bin"'

        self.paramFileSizeExpr=" * ".join(["_n"+str(k) for k in self.axes])

    def print(self):
        print(" Tensor[ grid = " ,self.grid, " axes= ", self.axes, " inout= ", self.inout, " tid = ", self.tid , " ]" )


def parse_tensor_line(problem_grid_in,LineIn):
    tensors_all_str=np.array((LineIn.split()[0]).split(","))
    i1=np.where(tensors_all_str=="in")[0]
    i2=np.where(tensors_all_str=="out")[0]
    ii=np.sort(np.concatenate((i1,i2)))
    tensors_all=np.split(tensors_all_str,ii+1)[:-1]
    tensors_axes=[ np.array(a[:-1]).astype(int) for a in tensors_all]
    tensors_inout=[ a[-1] for a in tensors_all]
    tensors_all_list=[ Tensor(problem_grid_in,a,io,tid) for tid,(a,io) in enumerate(zip(tensors_axes,tensors_inout)) ]
    return(tensors_all_list)

def get_grid_from_mcd(grid,tabs="\t"):
    s=""
    for i,n in enumerate(grid):
        s=s+tabs+"uint32_t _n"+str(n)+" = atoi(argv["+str(i+1)+"]);\n"
    return(s)

def get_declaretensorsizes(tensors,tabs="\t"):
    s=""
    for t in tensors:
        s=s+tabs+"size_t "+t.paramFileSize+" = "+t.paramFileSizeExpr+" * sizeof( DTYPE );\n"
    return(s)

def get_host_alloc(tensors,tabs="\t"):
    s=""
    for t in tensors:
        s=s+tabs+"uint8_t *"+t.paramHostname+";\n"
        s=s+tabs+"CHECK_ACL(aclrtMallocHost((void**)(&"+t.paramHostname+"), "+t.paramFileSize+"));\n"
    return(s)

def get_host_readfiles(tensors,tabs="\t"):
    s=""
    for t in tensors:
        if( t.inout == "in" ):
            s=s+tabs+'ReadFile('+t.paramFileNameIn+', '+t.paramFileSize+', '+t.paramHostname+', '+t.paramFileSize+');\n'
    return(s)

def get_device_alloc(tensors,tabs="\t"):
    s=""
    for t in tensors:
        s=s+tabs+"uint8_t *"+t.paramDevicename+";\n"
        s=s+tabs+"CHECK_ACL(aclrtMalloc((void**)(&"+t.paramDevicename+"), "+t.paramFileSize+", ACL_MEM_MALLOC_HUGE_FIRST));\n"
    return(s)

def get_host2device_move(tensors,tabs="\t\t"):
    s=""
    for t in tensors:
        if( t.inout == "in" ):
            s=s+tabs+"CHECK_ACL(aclrtMemcpy("+t.paramDevicename+", "+t.paramFileSize+", "+t.paramHostname+", "+t.paramFileSize+", ACL_MEMCPY_HOST_TO_DEVICE));\n"
    return(s)

def get_devicetensor_arglist(tensors,tabs="\t\t\t"):
    s=tabs+",".join([t.paramDevicename for t in tensors])
    return(s)

def get_alldim_list(grid,tabs="\t\t\t"):
    s=tabs+", ".join(["_n"+str(k) for k in grid])
    return(s)

def get_device2host_move(tensors,tabs="\t"):
    s=""
    for t in tensors:
        if( t.inout == "out" ):
            s=s+tabs+"CHECK_ACL(aclrtMemcpy("+t.paramHostname+", "+t.paramFileSize+", "+t.paramDevicename+", "+t.paramFileSize+", ACL_MEMCPY_DEVICE_TO_HOST));\n"
    return(s)

def get_device_free(tensors,tabs="\t"):
    s=""
    for t in tensors:
        s=s+tabs+"CHECK_ACL(aclrtFree("+t.paramDevicename+"));\n"
    return(s)

def get_host_free(tensors,tabs="\t"):
    s=""
    for t in tensors:
        s=s+tabs+"CHECK_ACL(aclrtFreeHost("+t.paramHostname+"));\n"
    return(s)

def get_host_write(tensors,tabs="\t"):
    s=""
    for t in tensors:
        if( t.inout == "out" ):
            s=s+tabs+'WriteFile('+t.paramFileNameOut+', '+t.paramHostname+', '+t.paramFileSize+');\n'
    return(s)

def get_frwdec_tensorlist(tensors,tabs="\t"):
    s=tabs+", ".join([ "uint8_t *"+t.paramName for t in tensors])
    return(s)

def get_frwdec_alldim_list(grid,tabs="\t"):
    s=tabs+", ".join(["uint32_t n"+str(k) for k in grid])
    return(s)

def get_frwdec_all_thrd_dim_list(grid,tabs="\t"):
    s=tabs+", ".join(["uint32_t _p"+str(k) for k in grid])
    return(s)

##############  cpu code gen  ##################

def get_cpu_alloc(tensors,tabs="\t"):
    s=""
    for t in tensors:
        s=s+tabs+"uint8_t* "+t.paramName+" = (uint8_t*)AscendC::GmAlloc("+t.paramFileSize+");\n"
    return(s)

def get_cpu_readfiles(tensors,tabs="\t"):
    s=""
    for t in tensors:
        if( t.inout == "in" ):
            s=s+tabs+'ReadFile('+t.paramFileNameIn+', '+t.paramFileSize+', '+t.paramName+', '+t.paramFileSize+');\n'
    return(s)

def get_cputensor_arglist(tensors,tabs="\t\t"):
    s=tabs+",".join([t.paramName for t in tensors])
    return(s)

def get_cpu_write(tensors,tabs="\t"):
    s=""
    for t in tensors:
        if( t.inout == "out" ):
            s=s+tabs+'WriteFile('+t.paramFileNameOut+', '+t.paramName+', '+t.paramFileSize+');\n'
    return(s)

def get_cpu_free(tensors,tabs="\t"):
    s=""
    for t in tensors:
        s=s+tabs+"AscendC::GmFree((void *)"+t.paramName+");\n"
    return(s)

def get_cpu_frwdec_tensorlist(tensors,tabs="\t"):
    s=tabs+", ".join([ "GM_ADDR "+t.paramName for t in tensors])
    return(s)

def get_cpu_frwdec_alldim_list(grid,tabs="\t"):
    s=tabs+", ".join(["uint32_t n"+str(k) for k in grid])
    return(s)

def get_cpu_frwdec_all_thrd_dim_list(grid,tabs="\t"):
    s=tabs+", ".join(["uint32_t _p"+str(k) for k in grid])
    return(s)

def get_analytic_mdel_arg_list(t,tabs=""):
    s=tabs+", ".join([x.split()[-1] for x in t.strip().split(",") if x])
    return(s)

def get_analytic_model_init(t,tcode,tabs="\t"):
    s=""
    s=s+"\n#ifdef _ANALYTIC_MODEL_\n"
    s=s+"\n".join(tcode)
    s=s+"\n#else\n"
    s=s+";\n ".join([tabs+x.strip()+" = 1;" for x in t.strip().split(",") if x])
    s=s+"\n#endif\n"
    return(s)



parser = argparse.ArgumentParser(description='Generate host test code.')
parser.add_argument('template_file', type=str, nargs='+',
                    help='host code will be generated from this template')
parser.add_argument('out_file', type=str, nargs='+',
                    help='generated host code file name')
parser.add_argument('in_file', type=str, nargs='+',
                    help='input paramters')
parser.add_argument('repeats', type=str, nargs='+', default="10",
                    help='number or repeats in the unit tests')
parser.add_argument('device_id', type=str, nargs='+', default="0",
                    help='device id used in tests')
parser.add_argument('nthreads', type=str, nargs='+', default="8",
                    help='number of threads used in tests')
args = parser.parse_args()

template_file=args.template_file[0]
out_file=args.out_file[0]
in_file=args.in_file[0]
repeats=args.repeats[0]
device_id=args.device_id[0]
nthreads=args.nthreads[0]

print("args.template_file=",args.template_file)
print("args.out_file=",args.out_file)
print("args.in_file=",args.in_file)
print("args.repeats=",args.repeats)
print("args.device_id=",args.device_id)
print("args.nthreads=",args.nthreads)

file1 = open(in_file, 'r')
Lines = file1.readlines()
file1.close()

thread_grid=np.array((Lines[0].split()[0]).split(",")).astype(int)
problem_grid=np.array((Lines[1].split()[0]).split(",")).astype(int)
tensors_all=parse_tensor_line(problem_grid,Lines[2])
kernel_name=Lines[3].split()[0]
analyticModelFormalParams=Lines[4]
i1=np.where( [ "BEGIN_ANALYTIC_MODEL" in l for l in Lines ] )[0][0]
i2=np.where( [ "END_ANALYTIC_MODEL" in l for l in Lines ] )[0][0]
analyticModelInitCode=Lines[i1+1:i2]
print("kernel_name =",kernel_name)
print("thread_grid =",thread_grid)
print("problem_grid =",problem_grid)
print("analyticModelFormalParams =",analyticModelFormalParams)
print("analyticModelInitCode =",analyticModelInitCode)
print("tensors =")
for t in tensors_all:
    t.print()

replace_rules=[
    ("##KERNELNAME##",kernel_name),
    ("##REPEATS##",repeats),
    ("##DEVICEID##",device_id),
    ("##DECLARESIZES##",get_grid_from_mcd(problem_grid)),
    ("##NTHREADS##",nthreads),
    ("##DECLARETENSORSIZES##",get_declaretensorsizes(tensors_all)),
    ("##HOSTDECLARETENSOR##",get_host_alloc(tensors_all)),
    ("##HOSTREADFILES##",get_host_readfiles(tensors_all)),
    ("##DEVICEDECLARETENSOR##",get_device_alloc(tensors_all)),
    ("##HOST2DEVICEMOVE##",get_host2device_move(tensors_all)),
    ("##DEVICETENSORLIST##",get_devicetensor_arglist(tensors_all)),
    ("##ALLDIMENSIONSLIST##",get_alldim_list(problem_grid)),
    ("##DEVICE2HOSTMOVE##",get_device2host_move(tensors_all)),
    ("##DEVICEFREETENSOR##",get_device_free(tensors_all)),
    ("##HOSTFREETENSOR##",get_host_free(tensors_all)),
    ("##WRITETENSOR##",get_host_write(tensors_all)),
    ("##FRWDECTENSORALLLIST##",get_frwdec_tensorlist(tensors_all)),
    ("##FRWDECTENSORSIZESLIST##",get_frwdec_alldim_list(problem_grid)),
    ("##FRWDECTHRDGRIDLIST##",get_frwdec_all_thrd_dim_list(thread_grid)),
    ("##CPUDECLARETENSOR##",get_cpu_alloc(tensors_all)),
    ("##CPUREADFILES##",get_cpu_readfiles(tensors_all)),
    ("##CPUTENSORLIST##",get_cputensor_arglist(tensors_all)),
    ("##CPUWRITETENSOR##",get_cpu_write(tensors_all)),
    ("##CPUFREETENSOR##",get_cpu_free(tensors_all)),
    ("##CPUFRWDECTENSORALLLIST##",get_cpu_frwdec_tensorlist(tensors_all)),
    ("##CPUFRWDECTENSORSIZESLIST##",get_cpu_frwdec_alldim_list(problem_grid)),
    ("##CPUFRWDECTHRDGRIDLIST##",get_cpu_frwdec_all_thrd_dim_list(thread_grid)),
    ("##ANALYTICMODELFORMALPARAMS##","\t"+analyticModelFormalParams),
    ("##ANALYTICMODELPARAMS##",get_analytic_mdel_arg_list(analyticModelFormalParams)),
    ("##DECLAREANALYTICMODELPARAMS##",get_analytic_model_init(analyticModelFormalParams,analyticModelInitCode))

]

file1 = open(template_file, 'r')
Lines = file1.readlines()

text=copy.copy(Lines)
for old,new in replace_rules:
    for i in range(len(text)):
        text[i] = text[i].replace(old,new)

# writing to file
file1 = open(out_file, 'w')
file1.writelines(text)
file1.close()
