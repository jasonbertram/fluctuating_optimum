# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:27:44 2023

@author: jason
"""
#Theory functions
#======================
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.optimize
import ast
                  

def phi_not_normed(x,N,mu,sigma_c2,s):
    
    if sigma_c2==0:
        return np.exp(-2*N*s*x*(1-x))*(x*(1-x))**(2*N*mu-1)
    else:
        return (1+1*N*sigma_c2*x*(1-x))**(-2*s/sigma_c2-2*N*mu)/(x*(1-x))**(1-2*N*mu)
    

def phi_norm_const(N,mu,sigma_c2,s):
    return 0.5/(integrate.quad(lambda x: phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]+integrate.quad(lambda x: phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0])

def E_H(N,mu,sigma_c2,s):
    return 2*(integrate.quad(lambda x: x*(1-x)*phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]+integrate.quad(lambda x: x*(1-x)*phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0])*phi_norm_const(N,mu,sigma_c2,s)


def Vg_pred(Vg,N,mu,a,L,sigma_e2,V_s):
    s=a**2/(2*V_s)
    sigma_c2=a**2*sigma_e2/(Vg+0.01)**2
    return 2*a**2*L*E_H(N,mu,sigma_c2,s)

def Vg_pred_consistent(init,N,mu,a,L,sigma_e2,V_s):
    res = scipy.optimize.minimize(lambda x: (Vg_pred(x,N,mu,a,L,sigma_e2,V_s)-x)**2, init)
    return res.x[0]


def Vg_theory_opt(x,N,a,so2,L,mu,Vs):
    
    if so2==0:
        return x-4*L*mu*V_s
    
    else: 
        sc2=a**2*so2/x**2
        d=np.abs(0.5*(1-np.sqrt(1+4/(1*N*sc2))))
        b=x**2/(Vs*so2)-2*N*mu
        b_t=b/(d+1)
    
        return (2*N*mu)*(2*L*a**2)*d**b_t*(d**(1-b_t)-(0.5+d)**(1-b_t))/(b_t-1)-x


#%%%
#Load data

with open("Vg_sims",'r') as fin:
    params=ast.literal_eval(fin.readline()[2:-1])

Vg_sims=np.loadtxt("Vg_sims")

#%%%
#Generate figures
offset=1e-5
for i in range(len(params)):

    L,sigma_e2,N,V_s,mu,Vm,rep=params[i]

    a=np.sqrt(Vm/(2*L*mu))

    if a**2<1 and mu<5e-5 and V_s<20 and Vm>5e-5:

        plt.figure(str([L,N,V_s,mu,Vm,rep]))
        
        plt.gca().set_title(str([L,N,V_s,mu,Vm,rep]))
        
        
        #Environmental variance set to 1
        plt.violinplot(Vg_sims[i]/(Vg_sims[i]+1),positions=[np.log10(sigma_e2+offset)],widths=5e-1,showmeans=True)
        
        
        #h_2=Vg_mean/(Vg_mean+1)
        #plt.plot(sigma_e2s,h_2,label=str(N)+','+str(V_s)+','+str(mu))
        
        #Vg_theory=0.5*2*L*mu*V_s*(1+np.sqrt(1+sigma_e2s/(V_s*L**2*mu**2)))
        
        Vg_theory=scipy.optimize.fsolve(lambda x: Vg_theory_opt(x,N,a,sigma_e2,L,mu,V_s),0.025)[0]
        plt.plot(np.log10(sigma_e2+offset), Vg_theory/(Vg_theory+1),'o')
        
        
        Vg_numerical=Vg_pred_consistent(2e-1,N,mu,a,L,sigma_e2,V_s)
        plt.plot(np.log10(sigma_e2+offset), Vg_numerical/(Vg_numerical+1),'kx',label='numerical')
        
        plt.ylim([0,.6])
        
        



#%%

fig, ax1=plt.subplots()

def heritability(V_s,sigma_e2s,Lmu):
    Vg=0.5*4*Lmu*V_s*(1+np.sqrt(1+sigma_e2s/(4*V_s*Lmu**2)))
    return Vg/(1+Vg)

xs=np.linspace(1,20)
ys=np.linspace(1e-4,1e-2)

h2_array=np.array([[heritability(x,y,5e-2) for x in xs] for y in ys])
pos=plt.contourf(xs,ys,h2_array,levels=np.arange(0,np.max(h2_array)+0.05,0.05))
fig.colorbar(pos)

fig, ax1=plt.subplots()
xs=10**(np.linspace(-4,-1))

h2_array=np.array([[heritability(10,y,x) for x in xs] for y in ys])
pos=plt.contourf(np.linspace(-4,-1),np.log10(ys),h2_array,levels=np.arange(0,np.max(h2_array)+0.05,0.05))
fig.colorbar(pos)