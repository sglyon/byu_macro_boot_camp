from mpi4py import MPI
import numpy
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

A = numpy.array(numpy.random.randint(0,10,(3,3)), dtype=float)
local_a = numpy.zeros(3)
comm.Scatter(A,local_a)
print "process", rank, "has", local_a
