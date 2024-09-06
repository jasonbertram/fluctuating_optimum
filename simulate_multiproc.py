# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:31:48 2024

@author: jason
"""
import numpy as np
import itertools
import time
import ray
import os

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

@ray.remote
def simulate(param):
    
    start=time.time()
    
    L,sigma_e2,N,V_s,mu,a2,theta,rep=param
    
    #a=np.sqrt(Vm/(2*L*mu))
    a=np.sqrt(a2)

    sign=2*np.random.randint(0,2,[L,rep])-1
    
    opt=np.zeros(rep)
    
    maxiter=int(10*N)
    
    p=np.zeros([L,rep])
    
    for t in range(maxiter):
        
        fixed_loci_1=(p==1)
        
        #Reset fixed loci and remove them from optimum
        p[fixed_loci_1]=0
        opt=opt-2*a*np.sum(sign*fixed_loci_1,0)
        
        zbar=np.sum(2*a*sign*p**2+a*sign*p*(1-p),0)
        
        fixed_loci_0=(p==0)
        mutation_mask=((np.random.rand(L,rep)<N*mu) & fixed_loci_0)
        
        np.place(p,mutation_mask,1/N)
        np.place(sign,mutation_mask,2*np.random.randint(0,2,np.sum(mutation_mask))-1)
        
        poly_loci=np.logical_not(fixed_loci_0) & (p<1-1/N) #new mutants excluded also!
        
        #p[poly_loci]=p[poly_loci]+0*mu*(1-2*p[poly_loci])
        p[poly_loci]=p[poly_loci]+\
            (np.random.rand(np.sum(poly_loci))<N*mu*(1-p[poly_loci]))/N-\
                (np.random.rand(np.sum(poly_loci))<N*mu*p[poly_loci])/N
        
        p=np.random.binomial(N,p_prime_sel_opt(p,opt-zbar,a,sign,V_s))/N
        #p=np.random.binomial(N,p,size=sim_L)/N
        
        opt=(1-theta)*opt + np.random.normal(0,np.sqrt(sigma_e2),rep)

    with open('log.txt','a') as f:
        f.write(str(param)+": " + "time="+str(time.time()-start))
    
    print("================================"+str(param))
    return 2*a**2*np.sum(p*(1-p),0)

####################################
#Parallel simulator. One core per parameter vector.
#Uses ray for multi-node

sigma_e2s=np.array([0,1e-4,5e-4,1e-3,5e-3,1e-2])
Ls=np.array([100])
Ns=np.array([1000])
Vs=np.array([5])
mus=np.array([5e-6])
thetas=np.array([0e-2])
a2s=np.array([0.01,0.02,0.04,0.06,0.08,0.1])
#Mutational heritability = 2 L mu alpha**2
#Vms=np.array([1e-4])
reps=np.array([100])

params=[_ for _ in itertools.product(Ls,sigma_e2s,Ns,Vs,mus,a2s,thetas,reps)]

# Connect to Ray cluster
ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'])

futures = [simulate.remote(_) for _ in params]
output = np.array(ray.get(futures))

np.savetxt("Vg_sims",output,header=str(params))
