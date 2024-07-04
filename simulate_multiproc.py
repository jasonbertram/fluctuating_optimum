# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:31:48 2024

@author: jason
"""



import numpy as np
import itertools
from multiprocessing import Pool
import time

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
    
    start=time.time()
    
    L,sigma_e2,N,V_s,mu,Vm,theta,rep=param
    
    a=np.sqrt(Vm/(2*L*mu))
    
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
        
        # zbar_hist[t]=zbar
        # opt_hist[t]=opt
        # numfix_hist[t]=np.sum(fixed_loci_1,0)
        
             
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
    
    print(param,"time="+str(start-time.time()))
    
    return 2*a**2*np.sum(p*(1-p),0)



####################################
#Parallel simulator
#Will not run in interactive mode



sigma_e2s=np.array([0,1e-6,1e-5,1e-4,1e-3,1e-2])

Ls=np.array([100,1000])
Ns=np.array([1000,10000])
Vs=np.array([5])
mus=np.array([5e-7,5e-6])
thetas=np.array([1e-2])

#Mutational heritability = 2 L mu alpha**2
Vms=np.array([1e-4])

reps=np.array([1000])

params=[_ for _ in itertools.product(Ls,sigma_e2s,Ns,Vs,mus,Vms,thetas,reps)]



if __name__ == '__main__':
    p=Pool(16)
    output=np.array(p.map(simulate, params))
    

np.savetxt("Vg_sims",output,header=str(params))

#plot_params=[_ for _ in itertools.product(Ls,Ns,Vs,mus,Vms,reps)]
#np.savetxt("Vg_sims",output,header=str(list(sigma_e2s))+'\n'+str(plot_params))


    
