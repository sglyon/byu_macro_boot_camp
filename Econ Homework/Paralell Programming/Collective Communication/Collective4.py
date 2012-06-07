"""
Created May 17, 2012

Author: Spencer Lyon
"""

#Collective3.py
from mpi4py import MPI
import numpy
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#read from command line
n = int(sys.argv[1])

#arbitrary example vectors, generated to be evenly divided by the number of
#processes for convenience

x = numpy.random.rand(n) if comm.rank == 0 else None

#initialize as numpy arrays
maxVal = numpy.array([0.])

leftOvers = n % size

sendcount = numpy.array([n//size],dtype = float)

if rank < leftOvers:
    sendcount += 1.0


send_count_tuple = numpy.zeros(size)
comm.Allgather(sendcount, send_count_tuple)
send_count_tuple = tuple(send_count_tuple)


displacement = numpy.zeros(1)

if rank < leftOvers:
    displacement = numpy.array([rank * (n//size + 1)], dtype = float)
else:
    displacement = numpy.array([rank * n//size + leftOvers], dtype = float)

send_displacement_tuple = numpy.zeros(size)
comm.Allgather(displacement, send_displacement_tuple)
send_displacement_tuple = tuple(send_displacement_tuple)

print 'rank __ has the displacement __ ', rank, send_count_tuple



#initialize as numpy arrays
local_x = numpy.zeros(sendcount)

#divide up vectors

comm.Scatterv([x, send_count_tuple, send_displacement_tuple, MPI.DOUBLE],
    local_x, root=0)


#local computation of dot product
local_max = numpy.array([numpy.max(local_x)])


#sum the results of each
comm.Reduce(local_max, maxVal, op = MPI.MAX)

if (rank == 0):
        print "The maximum value is ", maxVal[0], "computed in parallel"
        print "and", numpy.max(x), "computed serially"
