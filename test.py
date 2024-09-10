import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf=None
if comm.rank==0:
    sendbuf=np.array(
            [[i+j for i in range(10)] for j in range(size)]
            ,dtype='i')

recvbuf = np.empty(10, dtype='i')
comm.Scatter(sendbuf,recvbuf,root=0)

recvbuf=recvbuf**2

output=np.empty([size,10],dtype='i')
print(rank,output)

comm.Gather(recvbuf,output,root=0)

print(rank,output)

#if comm.rank==0:
#    print(output)
