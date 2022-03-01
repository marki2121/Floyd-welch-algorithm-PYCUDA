import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule

from random import seed, randint

import time

V = int(input("Kolika je matrica? \n>"))

VV = numpy.int32(V)
m = numpy.empty((V,V), dtype=numpy.float32)
rez_cpu = numpy.empty_like(m)

rez_cpu = m

#Kreiranje matrice
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


#CPU solution
t1 = time.time()
for k in range(V):
    for i in range(V):
        for j in range(V):
            if((rez_cpu[i][k] + rez_cpu[k][j]) < rez_cpu[i][j]):
                rez_cpu[i][j] = rez_cpu[i][k] + rez_cpu[k][j]

t2 = time.time()
print("CPU: \n")
print(rez_cpu)
print("\n Vrijeme: " + str(t2-t1))

numpy.savetxt("outfile_cpu.txt", rez_cpu, fmt="%2.0f")

mod = SourceModule(open("main.cu").read())
magic = mod.get_function("funkcija")

block = V / 32

#Racunanje velicine blokova i gridova
if (block > 1):
    if(block % 2 != 0):
        block = int(block) + 1
    V = 32
else:
    block = 1 

print("\n\nBlock size (" + str(V), ", " + str(V) + ", 1)")
print("Grid size (" + str(int(block)) + ", " + str(int(block)) + ")")

#GPU solution
t1 = time.time()
for k in range(V):
    magic(drv.InOut(m), drv.In(VV), drv.In(numpy.int32(k)), block=(int(V), int(V), 1), grid=(int(block), int(block)))
t2 = time.time()

print("\n\n GPU: \n")
print(m)
print("\n Vrijeme: " + str(t2-t1))

numpy.savetxt("outfile_gpi.txt", m, fmt="%2.0f")