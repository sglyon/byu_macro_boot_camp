"""
Created May 16, 2012

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

x = numpy.linspace(0,100,n) if comm.rank == 0 else None
y = numpy.linspace(20,300,n) if comm.rank == 0 else None

#initialize as numpy arrays
dot = numpy.array([0.])

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


local_x = numpy.zeros(sendcount)
local_y = numpy.zeros(sendcount)

#divide up vectors

comm.Scatterv([x, send_count_tuple, send_displacement_tuple, MPI.DOUBLE],
    local_x, root=0)
comm.Scatterv([y, send_count_tuple, send_displacement_tuple, MPI.DOUBLE],
    local_y, root=0)

#local computation of dot product
local_dot = numpy.array([numpy.dot(local_x, local_y)])


#sum the results of each
comm.Reduce(local_dot, dot, op = MPI.SUM)

if (rank == 0):
        print "The dot product is", dot[0], "computed in parallel"
        print "and", numpy.dot(x,y), "computed serially"
