#passRandomDrawVector.py
import numpy as np
import scipy as sp
from scipy import random
import sys
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
destination = 0

randNum = sp.rand(1)
#randNum = random.randint(10)

if rank == size - 1:
    destination = 0
else:
    destination = rank + 1

print 'Process ', rank, 'drew the number ', randNum, 'to pass to ', destination


if rank == 0 :
    sourceProcess = size -1
else:
    sourceProcess = rank - 1

comm.Send(randNum, dest = destination)
comm.Recv(randNum, source = sourceProcess)

print 'Process ', rank, 'recieved the number', randNum, 'from process',\
       sourceProcess
