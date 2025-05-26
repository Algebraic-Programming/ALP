import pyalp_ref as pyalp
import numpy as np

N, M = 5 , 5
idata = np.array([ 0, 1, 2, 3, 3, 4, 2, 3, 3, 4, 1, 4, 1, 4, 4 ],dtype=np.int32)
jdata = np.array([ 0, 1, 2, 3, 2, 2, 1, 4, 1, 1, 0, 3, 0, 3, 4 ],dtype=np.int32)
vdata = np.array([ 1, 1, 1, 1, .5, 2, 1, 4, 4.4, 1, 0, 3.5, 0, 3, 1 ], dtype=np.float64)
b = np.array([ 1., 1., 1., 1., 1. ], dtype=np.float64)
x = np.array([ 1,  1., 0., 0.3, -1. ], dtype=np.float64)
r = np.zeros(5)
u = np.zeros(5)
tmp = np.zeros(5)


A=np.zeros((M,N))
for i,j,v in zip(idata,jdata,vdata):
    A[i,j]=v

pyalp.print_my_numpy_array(b)

maxiterations = 2000
verbose = 1

#########################
alpmatrixA = pyalp.Matrix(5,5,idata,jdata,vdata)
alpvectorx = pyalp.Vector(5,x)
alpvectorb = pyalp.Vector(5,b)
alpvectorr   = pyalp.Vector(5,r)
alpvectoru   = pyalp.Vector(5,u)
alpvectortmp = pyalp.Vector(5,tmp)

iterations,residual = pyalp.conjugate_gradient( alpmatrixA, alpvectorx, alpvectorb, alpvectorr, alpvectoru, alpvectortmp, maxiterations, verbose )
print(" iterations = ", iterations )
print(" residual = ", residual )

x_result=alpvectorx.to_numpy()
print(x_result)

assert(np.allclose(x_result,np.array([ 1., 1.,  0., 0.13598679, -0.88396565])))
