import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule

from scipy.sparse.csgraph import floyd_warshall
from random import seed, randint

V = int(input("Kolika je matrica? \n>"))

m = numpy.empty((V,V), dtype=numpy.float32)
rez = numpy.empty_like(m)
tmp = numpy.empty_like(m)

def sam_unesi():
    for i in range(V):
        for j in range(V):
            if i==j:
                m[i][j] = 0
            else:
                n = input("Kolika je velicina u tocki x(" + str(i) + "," + str(j) + ")?\nAko je nema ostavi prazno!!!\n>")

                if n != "":
                    if float(n) > 10:
                        m[i][j] = numpy.Inf
                    else:
                        m[i][j] = float(n)
                else: 
                    m[i][j] = numpy.NaN

def rand_unos():
    seed(1)
    for i in range(V):
        for j in range(V):
            if i==j:
                m[i][j] = 0
            else:
                n = randint(0, 10)

                if n > 1:
                    if float(n) > 8:
                        m[i][j] = numpy.Inf
                    else:
                        m[i][j] = float(n)
                else: 
                    m[i][j] = numpy.NaN

o = input("Dali zelite da matica bude random d/n?\n>")

if o.lower() == "d":
    rand_unos()
else:
    sam_unesi()

print(m)
print("\n\n\n")

dist_mat, pred = floyd_warshall(csgraph=m, return_predecessors=True)

print("CPU: \n")
print(dist_mat)

#mod = SourceModule(open("main.cu").read())
#magic = mod.get_function("funkcija")

#for k in range(V):
  #  magic(drv.Out(rez), drv.In(m), drv.In(k), block=(V,V,1), grid=(1,1))
 #   m = rez


print("\n\nGPU:  \n")


