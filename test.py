import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf=None
if comm.rank==0:
    sendbuf=np.array(
            [[i+j for i in range(10)] for j in range(size)]
            ,dtype='f')

recvbuf=None

comm.Scatter(sendbuf,recvbuf,root=0)

print(rank,recvbuf)
#output=comm.Gather(outputs,root=0)

#if comm.rank==0:
#    print(output)
