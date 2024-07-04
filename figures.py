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
        return (1+N*sigma_c2*x*(1-x))**(-2*s/sigma_c2-2*N*mu)/(x*(1-x))**(1-2*N*mu)
    

def phi_norm_const(N,mu,sigma_c2,s):
    return 0.5/(integrate.quad(lambda x: phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]+integrate.quad(lambda x: phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0])

def E_H(N,mu,sigma_c2,s):
    return 2*(integrate.quad(lambda x: x*(1-x)*phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]+integrate.quad(lambda x: x*(1-x)*phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0])*phi_norm_const(N,mu,sigma_c2,s)


def Vg_pred(Vg,N,mu,a,L,sigma_e2,V_s):
    s=a**2/(2*V_s)
    sigma_c2=a**2*sigma_e2/(Vg)**2
    return 2*a**2*L*E_H(N,mu,sigma_c2,s)

def Vg_pred_consistent(init,N,mu,a,L,sigma_e2,V_s):
    res = scipy.optimize.minimize(lambda x: (Vg_pred(x,N,mu,a,L,sigma_e2,V_s)-x)**2, init)
    return res.x[0]


def Vg_theory_opt(x,N,a,so2,L,mu,Vs):
    
    if so2==0:
        return x-4*L*mu*V_s
    
    else: 
        sc2=a**2*so2/x**2
        d=np.abs(0.5*(1-np.sqrt(1+4/(N*sc2))))
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

offset=1e-7

unique_params=set([str([p[0],p[2],p[3],p[4],p[5]]) for p in params if p[5]/(2*p[0]*p[4])<0.5])

fig, axs=plt.subplots(3,2,figsize=[7,7])
axs=axs.flat

fig_dict=dict(zip(unique_params,axs))


for i in range(len(params)):

    L,sigma_e2,N,V_s,mu,Vm,theta,rep=params[i]

    a=np.sqrt(Vm/(2*L*mu))
        
    if a<0.5:

        ax=fig_dict[str([L,N,V_s,mu,Vm])]
        
        
        #Environmental variance set to 1
        ax.violinplot(Vg_sims[i]/(Vg_sims[i]+1),positions=[np.log10(sigma_e2+offset)],widths=0.75,showmeans=True)
        
        
        #h_2=Vg_mean/(Vg_mean+1)
        #plt.plot(sigma_e2s,h_2,label=str(N)+','+str(V_s)+','+str(mu))
        
        #Vg_theory=0.5*2*L*mu*V_s*(1+np.sqrt(1+sigma_e2s/(V_s*L**2*mu**2)))
        
        Vg_theory=scipy.optimize.fsolve(lambda x: Vg_theory_opt(x,N,a,sigma_e2,L,mu,V_s),0.025)[0]
        ax.plot(np.log10(sigma_e2+offset), Vg_theory/(Vg_theory+1),'ko',fillstyle='none',markersize=8,label=r'Theory analytical',alpha=0.7)
        
        
        Vg_numerical=Vg_pred_consistent(2e-1,N,mu,a,L,sigma_e2,V_s)
        ax.plot(np.log10(sigma_e2+offset), Vg_numerical/(Vg_numerical+1),'kx',markersize=10,label=r'Theory numerical',alpha=0.7)
        
        
        
        ax.set_ylim([0,.5])
        ax.set_title(r'$L=$'+str(L)+r'$,N=$'+str(N)+r'$,\mu=$'+str(mu),y=.83,fontsize=11)
                    


axs[2].set_ylabel(r'Heritability $h^2$',fontsize=14)

axs[1].set_yticklabels([])
axs[3].set_yticklabels([])
axs[5].set_yticklabels([])

axs[0].set_yticklabels([r'$0.0$',r'$0.1$',r'$0.2$',r'$0.3$',r'$0.4$',r'$0.5$'],fontsize=12)
axs[2].set_yticklabels([r'$0.0$',r'$0.1$',r'$0.2$',r'$0.3$',r'$0.4$',r'$0.5$'],fontsize=12)
axs[4].set_yticklabels([r'$0.0$',r'$0.1$',r'$0.2$',r'$0.3$',r'$0.4$',r'$0.5$'],fontsize=12)

for _ in axs:
    _.set_xticks([np.log10(x+offset) for x in [0,1e-6,1e-5,1e-4,1e-3,1e-2]])

axs[0].set_xticklabels([])
axs[1].set_xticklabels([])
axs[2].set_xticklabels([])
axs[3].set_xticklabels([])

axs[4].set_xticklabels([r'$0$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'],fontsize=12)
axs[5].set_xticklabels([r'$0$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'],fontsize=12)

axs[4].set_xlabel(r'Fluctuation intensity $\sigma^2$',x=1.2,fontsize=14)

#remove duplicate legend entries
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[0].legend(by_label.values(), by_label.keys(),loc=[0.3,.55],fontsize=10)

plt.savefig('violinplot.pdf')

#ax.set_xlabel(r'Fluctuation intensity $\sigma^2$',fontsize=14)

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