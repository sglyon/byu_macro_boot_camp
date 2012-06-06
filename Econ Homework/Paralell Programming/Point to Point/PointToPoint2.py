"""
Created May 16, 2012

Author: Spencer Lyon
"""

#Ch2Prob2.py
import numpy as np
import scipy as sp
import sys
from mpi4py import MPI

n = int(sys.argv[1])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

randNum = np.zeros(n)

if rank == 1:
	randNum = sp.rand(n)
	print "Process", rank, "drew the number", randNum
	comm.Send(randNum, dest = 0, tag = 33)

if rank == 0:
	print "Process", rank, "before receiving has the number", randNum
	comm.Recv(randNum, source=1)
	print "Process", rank, "received the number", randNum

# If I change the dest or source parameters the program doesn't run.
# If I change the tag parameter nothing happens on the surface, we just have
## one more way to identify where it came from or what it is. 
