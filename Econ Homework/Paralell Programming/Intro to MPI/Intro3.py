#hello.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 5:
    print "Error, This program must run with 5 processes!"
    comm.Abort()
else:
    print 'Success'
