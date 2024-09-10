# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:31:48 2024

@author: jason
"""
import numpy as np
import itertools
import time
from mpi4py import MPI

#Simulation functions
#======================
def pmap(rho):
    return rho/(1+rho)

def rhomap(p):
    return p/(1-p)

def p_prime_sel_opt(p,delt_opt,gam,sign,V_s):
    S=1/(2*V_s)
    p=pmap(rhomap(p)*np.exp(2*S*gam*sign*(delt_opt+0.5*gam*sign*(2*p-1))))    
    return p

def simulate(param):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=param
    a=np.sqrt(a2)

    sign=2*np.random.randint(0,2,[L,rep])-1
    
    opt=np.zeros(rep)
    p=np.zeros([L,rep])
    maxiter=int(10*N)
    
    for t in range(maxiter):
        
        fixed_loci_1=(p==1)
        
        #Reset fixed loci and remove them from optimum
        p[fixed_loci_1]=0
        opt=opt-2*a*np.sum(sign*fixed_loci_1,0)
        
        zbar=np.sum(2*a*sign*p**2+a*sign*p*(1-p),0)
        
        fixed_loci_0=(p==0)
        mutation_mask=((np.random.rand(L,rep)<N*mu) & fixed_loci_0)
        
        np.place(p,mutation_mask,1/N)
        np.place(
                sign,mutation_mask,
                2*np.random.randint(0,2,np.sum(mutation_mask))-1
                )
        
        poly_loci=np.logical_not(fixed_loci_0) & (p<1-1/N) 
        #new mutants excluded also
        
        #p[poly_loci]=p[poly_loci]+0*mu*(1-2*p[poly_loci])
        p[poly_loci]=p[poly_loci]+\
            (np.random.rand(np.sum(poly_loci))<N*mu*(1-p[poly_loci]))/N-\
                (np.random.rand(np.sum(poly_loci))<N*mu*p[poly_loci])/N
        
        p=np.random.binomial(N,p_prime_sel_opt(p,opt-zbar,a,sign,V_s))/N
        #p=np.random.binomial(N,p,size=sim_L)/N
        
        opt=(1-theta)*opt + np.random.normal(0,np.sqrt(sigma_e2),rep)

    return 2*a**2*np.sum(p*(1-p),0)

####################################
#Parallel handling of replicates with MPI.

#sigma_e2s=np.array([0,1e-4,5e-4,1e-3,5e-3,1e-2])
sigma_e2s=np.array([0,1e-4])
Ls=np.array([1000])
Ns=np.array([1000])
Vs=np.array([5])
mus=np.array([5e-6])
thetas=np.array([0e-2])
#a2s=np.array([0.01,0.02,0.04,0.06,0.08,0.1])
a2s=np.array([0.01])
all_reps=100

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

rep_local=int(all_reps/size)
params=[_ for _ in 
        itertools.product(Ls,sigma_e2s,Ns,Vs,mus,a2s,thetas,[rep_local])
        ]

output=[]
for param in params:
    start=time.time()
    Vg_local=simulate(param)

    recvbuf=None
    if rank==0:
        recvbuf=np.empty([size,rep_local],dtype='d')
    comm.Gather(Vg_local,recvbuf,root=0)

    if rank==0:
        print((time.time()-start)/60)
        output.append(recvbuf.flatten())

params=[_ for _ in
        itertools.product(Ls,sigma_e2s,Ns,Vs,mus,a2s,thetas,reps)
        ]

np.savetxt("Vg_sims",output,header=str(params))
