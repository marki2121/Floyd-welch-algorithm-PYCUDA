import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule

from scipy.sparse.csgraph import floyd_warshall
from random import seed, randint

import time

V = int(input("Kolika je matrica? \n>"))

VV = numpy.int32(V)
m = numpy.empty((V,V), dtype=numpy.float32)
rez = numpy.empty_like(m)
tmp = numpy.empty_like(m)


def rand_unos():
    seed(1)
    for i in range(V):
        for j in range(V):
            if i==j:
                m[i][j] = 0
            else:
                n = randint(1, 20)
                if float(n) > 16:
                    m[i][j] = numpy.Inf
                else:
                    m[i][j] = float(n)


rand_unos()

print(m)
print("\n\n\n")

t1 = time.time()
dist_mat, pred = floyd_warshall(csgraph=m, return_predecessors=True)
t2 = time.time()
print("CPU: \n")
print(dist_mat)
print("\n Vrijeme: " + str(t2-t1))

mod = SourceModule(open("main.cu").read())
magic = mod.get_function("funkcija")

block = V / 32

if (block > 1):
    if(block % 2 != 0):
        block = int(block) + 1
    V = 32
else:
    block = 1 

print("\n\nBlock size (" + str(V), ", " + str(V) + ", 1)")
print("Grid size (" + str(int(block)) + ", " + str(int(block)) + ")")

t1 = time.time()
for k in range(V):
    magic(drv.InOut(m), drv.In(VV), drv.In(numpy.int32(k)), block=(int(V), int(V), 1), grid=(int(block), int(block)))
t2 = time.time()

print("\n\n GPU: \n")
print(m)
print("\n Vrijeme: " + str(t2-t1))