from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = int(sys.argv[1])

if rank ==0:
    x = np.array(np.random.rand(n), dtype = float)
    A = np.array(np.random.randint(0,10,(n,n)), dtype=float)
else:
    x = np.zeros(n)
    A = np.zeros((n,n))

local_a = np.zeros(n)

comm.Scatter(A,local_a,root=0)
comm.Bcast(x, root = 0)

#local_product = np.zeros(1)



local_product = np.array([np.dot(x, local_a)])

solution = np.zeros(n)

for i in range(0,size):
    comm.Gather(local_product, solution, root =0)

if rank ==0:
    print 'the solution is ', solution, 'as computed in paralell'
    print 'the solution in serial is', np.dot(A,x)


# This is how to use Bcast and gather.

# To use Scatter and Reduce we would have scattered columns of A by scattering
# A.T and then given each process one element of the vector and told it to find
# the product of that element and each element in its column of A. We then would
# use Redcue to sum accross each of the columns and get the answer.
