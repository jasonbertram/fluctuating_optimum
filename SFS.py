# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:07:53 2022

@author: jason
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


#Simulation functions
#======================
def pmap(rho):
    return rho/(1+rho)

def rhomap(p):
    return p/(1-p)

#simplified selection mapping assuming zbar=opt
def p_prime_sel_opt(p,delt_opt,gam,S):
    
    #exclude loci fixed at p=1 where rhomap diverges
    nf=(p!=1)
    
    p[nf]=pmap(rhomap(p[nf])*np.exp(2*S*gam[nf]*(delt_opt[nf]+0.5*gam[nf]*(2*p[nf]-1))))
    
    return p


#Theory functions
#======================
from scipy.special import hyp2f1,erf
from scipy import integrate
from scipy.optimize import minimize

def int_fac(x,a,b):
    a_fac=np.sqrt(a/(a+4))
    
    # return (1/a_fac)*(2**(b-1)/(b+1)) \
    #     *(1-a_fac*(1-2*x)) \
    #         *(1+a_fac*(1-2*x))**(-b) \
    #             *(1+a*x*(1-x))**b \
    #                 *hyp2f1(-b,b+1,b+2,(1-a_fac*(1-2*x))/2)
    return x#**(0*b+1)*(1+0*(-b)/(1+1/(b+1))*x+0*(-b)*(-b+1)/(1+1/(b+1)+1/(b+2))*x**2/2)
    
                    

def phi_not_normed(x,N,sigma_c2,s):
    
    C=int_fac(1,N*sigma_c2,2*s/sigma_c2-1)
    
    a=N*sigma_c2
    
    x_m=0.5-0.5*np.sqrt(1+4/a)
    x_p=0.5+0.5*np.sqrt(1+4/a)
    
    #return (1+N*sigma_c2*x*(1-x))**(-2*s/sigma_c2)/(x*(1-x))*(C-int_fac(x,N*sigma_c2,2*s/sigma_c2-1))
    #return (1+N*sigma_c2*x*(1-x))**(-2*s/sigma_c2)/(x*(1-x))*(C-int_fac(x,N*sigma_c2,2*s/sigma_c2-1))
    return (1/x)*(1/(x-x_m)-0*1/(x-x_p))**(2*s/sigma_c2)
    

def phi_norm_const(N,sigma_c2,s,mu):
    return 1/(integrate.quad(lambda x: phi_not_normed(x,N,sigma_c2,s),1/N,1-1/N)[0]+0.*2/((1-np.exp(-1))*mu*N**2)*(np.exp(-1)*phi_not_normed(1/N,N,sigma_c2,s)+1*np.exp(-2)*phi_not_normed(2/N,N,sigma_c2,s)))

def E_H(N,sigma_c2,s,mu):
    return integrate.quad(lambda x: x*(1-x)*phi_not_normed(x,N,sigma_c2,s),1/N,1)[0]*phi_norm_const(N,sigma_c2,s,mu)


# def int_fac_nofluc(x,N,s):
#     a=2*N*s
#     return erf(0.5*np.sqrt(a)*(2*x-1))

# def phi_nofluc(x,N,s):
#     C=int_fac_nofluc(1,N,s)
#     norm_const=1/integrate.quad(lambda x: np.exp(-2*N*s*x*(1-x))/(x*(1-x))*(C-int_fac_nofluc(x,N,s)),1/N,1)[0]
#     return norm_const*np.exp(-2*N*s*x*(1-x))/(x*(1-x))*(C-int_fac_nofluc(x,N,s))


def Vg_pred(Vg,N,alpha,L,sigma_e2,V_s):
    s=alpha**2/(2*V_s)
    sigma_c2=alpha**2*sigma_e2/Vg**2
    return alpha**2*L*E_H(N,sigma_c2,s,mu)

def Vg_pred_consistent(init,N,alpha,L,sigma_e2,V_s,trunc):
    res = minimize(lambda x: (Vg_pred(x,N,alpha,L,sigma_e2,V_s)-x)**2, init)
    return res.x[0]

#%%%

Vg_sims={}


L=1000
rep=100
sim_L=rep*L

sigma_e2s=np.array([0,1e-4,1e-2])
#sigma_e2s=np.array([1e-2])

#Ns=np.array([100,1000,10000])
Ns=np.array([1000])
Vs=np.array([20])
#mus=np.array([1e-6,1e-5,1e-4])
mus=np.array([1e-5])

#Mutational heritability = 2 L mu alpha**2 (2 comes from diploidy)  
Vm=1e-2

#If dt<1, iterate in fraction of a generation
dt=1

for sigma_e2,N,V_s,mu in itertools.product(sigma_e2s,Ns,Vs,mus):
    
    
    alpha=np.sqrt(Vm/(2*L*mu))
    
    gam=alpha*np.ones(sim_L)
    
    zbar=0
    opt=0
    
    #print(sigma_e2,N,V_s,mu,alpha**2/(2*V_s))
    
    maxiter=int(10*N/dt)
    
    p=np.ones(sim_L)/N
    for _ in range(rep):
        p[_*L+0:_*L+int(L/2)]=1-np.ones(int(L/2))/N
        
    phist=np.zeros([1,sim_L])
    
    for t in tqdm(range(maxiter)):
        
        p_Vg=np.array(p.reshape(L,int(rep)))
        
        V_g=np.ravel(np.outer(alpha**2*np.sum(p_Vg*(1-p_Vg),0),np.ones(L)))
        
        #zbar=np.ravel(np.outer(alpha*np.mean(p_Vg,0),np.ones(L)))
        
        zbar=zbar+V_g/V_s*(opt-zbar)*dt
        opt=np.random.normal(opt,np.sqrt(sigma_e2*dt),sim_L)
        
        
        
        fixed_loci_0=(p==0)
        fixed_loci_1=(p==1)

        p[fixed_loci_0]=(np.random.rand(np.sum(fixed_loci_0))<N*mu)/N
        p[fixed_loci_1]=1-(np.random.rand(np.sum(fixed_loci_1))<N*mu)/N

        
        
        p=np.random.binomial(int(N/dt),p_prime_sel_opt(p,opt-zbar,gam,dt/(2*V_s)),size=sim_L)/(N/dt)
        #p=np.random.binomial(N,p,size=sim_L)/N
        
        phist[0]=p
        
    
    Vg_sims[str([sigma_e2,N,V_s,mu])]=V_g

#Vg_sims=np.array(Vg_sims)


#%%%
#Genetic variance versus sigma_e2

params=np.array(list(itertools.product(sigma_e2s,Ns,Vs,mus)))

for N in Ns:
    for V_s in Vs:
        for mu in mus:
            plt.figure()
            
            Vg_mean=np.array([np.mean(Vg_sims[str([_,N,V_s,mu])]) for _ in sigma_e2s])
            
            #Environmental variance set to 1
            h_2=Vg_mean/(Vg_mean+1)
            
            plt.plot(sigma_e2s,h_2,label=str(N)+','+str(V_s)+','+str(mu))
            plt.ylim([0,1])
            plt.legend()
            

            #SFS from diffusion approximation
            alpha=np.sqrt(Vm/(2*L*mu))
            sigma_c2s=alpha**2*sigma_e2s[1:]/Vg_mean[1:]**2
            print(sigma_c2s)
            s=alpha**2/(2*V_s)
            
            H_theory=np.array([E_H(N,sigma_c2,s,mu) for sigma_c2 in sigma_c2s])
            Vg_theory=2*alpha**2*L*H_theory

            Vg_theory_consistent=np.array([Vg_pred_consistent(Vg_mean[_],N,alpha,L,sigma_e2s[_],V_s,1/N) for _ in range(len(sigma_e2s))])
            
            #plt.semilogx(sigma_e2s,alpha**2*L*H_theory)
            
            plt.plot(sigma_e2s[1:],Vg_theory/(Vg_theory+1),'k')
            #plt.semilogx(sigma_e2s,Vg_theory_consistent)
            plt.plot(sigma_e2s,Vg_theory_consistent/(Vg_theory_consistent+1))
            
            
            


#%%%
#SFS
#

V_g_theory=np.mean(V_g)
#V_g_theory=0.8

sigma_c2=alpha**2*sigma_e2/V_g_theory**2
#sigma_c2=0.008
s=alpha**2/(2*V_s)

x=np.linspace(1/N,1-1/N,1000)

#u=log(x)
u=np.linspace(np.log(1/N),np.log(1-1/N),31)

plt.figure()
poly_loci=((p<1) & (p>0))
plt.hist(np.log(phist[-1][poly_loci]),u,density=True)
#plt.hist(phist[-1][poly_loci],x,density=True)

#neutral
#plt.semilogy(x,0.9/(np.log(N)*x),'k--')
#ignoring frequency dependence of average selection pressure
#plt.semilogy(x,0.2*np.exp(-2*N*s*x)/(x*(1-x)),'k--')
#complete
u=np.linspace(np.log(1/N),np.log(1-1/N),101)
if sigma_e2>0:
    plt.semilogy(u,np.array([phi_not_normed(np.exp(_),N,sigma_c2,s)*np.exp(_) for _ in u])*phi_norm_const(N,sigma_c2,s,mu))
    #plt.semilogy(x,np.array([phi_not_normed(_,N,sigma_c2,s) for _ in x])*phi_norm_const(N,sigma_c2,s,mu))
    
#V->0
#plt.semilogy(x,phi_nofluc(x,N,s),'r')

#plt.ylim([10**-2,10**3])

#print(np.sum((phist[-1]>0.1) & (phist[-1]<0.9))/L)

# plt.figure()
# p_ordered=np.sort(phist[-1])[::-1]/np.sum(p_ordered)
# plt.plot(np.array(range(len(p_ordered))),np.cumsum(p_ordered))
#plt.ylim([0,1])

#%%%

from scipy.integrate import quad
from scipy.optimize import minimize

trunc=1/N

# neutral_norm=quad(lambda x: 1/x,trunc,1-trunc)[0]

# print(quad(lambda x: x*(1-x)*1/x,trunc,1-trunc)[0]/neutral_norm)

# MS_norm=quad(lambda x: np.exp(-2*N*s*x)/x,trunc,1-trunc)[0]
# print(quad(lambda x: x*(1-x)*np.exp(-2*N*s*x)/x,trunc,1-trunc)[0]/MS_norm)

# noV_norm=quad(lambda x: phi_noV(x,N,s),trunc,1-trunc)[0]
# print(quad(lambda x: x*(1-x)*phi_noV(x,N,s),trunc,1-trunc)[0]/noV_norm)

def Vg_pred_MS_drift(N,alpha,V_s,trunc):
    s=alpha**2/(2*V_s)
    
    norm=quad(lambda x: np.exp(-2*N*s*x)/(x*(1-x)),trunc,1-trunc)[0]
    return alpha*(quad(lambda x: x*(1-x)*np.exp(-2*N*s*x)/(x*(1-x)),trunc,1-trunc)[0])/norm

def Vg_pred_nofluc(N,alpha,V_s,trunc):
    s=alpha**2/(2*V_s)
    
    norm=quad(lambda x: phi_nofluc(x,N,s),trunc,1-trunc)[0]
    return alpha*(quad(lambda x: x*(1-x)*phi_nofluc(x,N,s),trunc,1-trunc)[0])/norm



#print(Vg_pred(Vg_sim,N,alpha,sigma_e2,V_s,trunc))

#Vg_theory=np.array([Vg_pred_consistent(alpha*0.5/np.log(N),N,alpha,sigma_e2,_,trunc) for _ in Vs_range])
Vg_theory_nofluc=np.array([Vg_pred_nofluc(N,alpha,_,trunc) for _ in Vs_range])
Vg_theory_MS_drift=np.array([Vg_pred_MS_drift(N,alpha,_,trunc) for _ in Vs_range])

plt.figure()
#plt.plot(np.sqrt(Vs_range),Vg_theory)
plt.plot(np.sqrt(Vs_range),Vg_theory_nofluc,'k--')
#plt.plot(np.sqrt(Vs_range),Vg_theory_MS_drift,'k')
plt.plot(np.sqrt(Vs_range),len(Vs_range)*[0.5*alpha/np.log(N)])
gamma=N*alpha**2/Vs_range
#plt.plot(np.sqrt(Vs_range),0.075*alpha/(gamma/2 + gamma/(np.exp(gamma)-1)))
plt.plot(np.sqrt(Vs_range),0.06*alpha*(2/gamma)*(1-np.exp(-gamma))/(np.exp(-gamma)+1),'k')
plt.plot(np.sqrt(Vs_range),Vg_sims,'.')

mu=5*10**-6
p_MS=mu/(1/(2*Vs_range))
plt.plot(np.sqrt(Vs_range),alpha*p_MS*(1-p_MS))


# plt.figure()
# plt.plot(np.sqrt(Vs_range),Vg_theory/(Vg_theory+Vs_range/10))
# plt.plot(np.sqrt(Vs_range),Vg_theory_nofluc/(Vg_theory_nofluc+Vs_range/10))
# plt.plot(np.sqrt(Vs_range),Vg_theory_MS_drift/(Vg_theory_MS_drift+Vs_range/10))
# plt.plot(np.sqrt(Vs_range),alpha*p_MS*(1-p_MS)/(alpha*p_MS*(1-p_MS)+Vs_range/10))

# plt.plot(V_g_range,Vg_pred)
# plt.plot(V_g_range,V_g_range)

