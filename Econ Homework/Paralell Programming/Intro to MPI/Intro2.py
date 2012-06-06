#hello.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank % 2 == 0:
    print 'Hello from process', rank
else:
    print 'Goodbye from process', rank
