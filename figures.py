# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:27:44 2023

@author: jason
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.optimize
import ast

#
#Theory functions
#======================                  

def Vg_LB(mu,L,Vs):
    return 4*mu*L*Vs

def phi_not_normed(x,N,mu,sigma_c2,s):
    if sigma_c2==0:
        return np.exp(-2*N*s*x*(1-x))*(x*(1-x))**(2*N*mu-1)
    else:
        return ((1+N*sigma_c2*x*(1-x))**(-2*s/sigma_c2-2*N*mu)
                /(x*(1-x))**(1-2*N*mu))

def phi_norm_const(N,mu,sigma_c2,s):
    return (0.5/(integrate.quad(lambda x: 
            phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]
            +integrate.quad(lambda x: 
            phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0]))

def E_H(N,mu,sigma_c2,s):
    return (2*(integrate.quad(lambda x: x*(1-x)
            *phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]
            +integrate.quad(lambda x: x*(1-x)
            *phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0])
            *phi_norm_const(N,mu,sigma_c2,s))

def Vg_pred(Vg,N,mu,a,L,sigma2,V_s):
    s=a**2/(2*V_s)
    sigma_c2=a**2*sigma2/(Vg)**2
    return 2*a**2*L*E_H(N,mu,sigma_c2,s)

def Vg_pred_consistent(init,N,mu,a,L,sigma2,V_s):
    res = (scipy.optimize.minimize(lambda x: 
            (Vg_pred(x,N,mu,a,L,sigma2,V_s)-x)**2, init))
    return res.x[0]

def Vg_theory_opt(x,N,a,sigma_e2,L,mu,Vs):
    if sigma_e2==0:
        return x-4*L*mu*Vs
    else: 
        sc2=a**2*sigma_e2/x**2
        d=np.abs(0.5*(1-np.sqrt(1+4/(N*sc2))))
        b=x**2/(Vs*sigma_e2)-2*N*mu
        b_t=b/(d+1)
        return ((2*N*mu)*(2*L*a**2)*d**b_t*(d**(1-b_t)
                -(0.5+d)**(1-b_t))/(b_t-1)-x)


#%%
#Load data

fname="Vg_sims_th0"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep

Vg_sims=np.loadtxt(fname)

#%%
########################################
#Exploratory plots

#small offset to avoid log(0)
offset=1e-5

#index of variable on x axis
xvar=1

#log x axis option
logx=True

indices=[_ for _ in range(len(params)) if params[_][-3]==0.1]

for i in indices:

    plt.figure(str([_ for ind,_ in enumerate(params[i]) if ind != xvar]))
    ax=plt.gca() 

    L,sigma_e2,N,V_s,mu,a2,theta,rep=params[i]
    a=np.sqrt(a2)
    
    #Vg_theory=scipy.optimize.fsolve(
    #        lambda x: Vg_theory_opt(x,N,a,sigma_e2,L,mu,V_s),0.025)[0]
    
    Vg_numerical=Vg_pred_consistent(2e-1,N,mu,a,L,sigma_e2,V_s)
    
    h2_LB=Vg_LB(mu,L,V_s)/(1+Vg_LB(mu,L,V_s))
    ax.axhline(y=h2_LB,color='k',ls='--')
    ax.axhspan(0.1,0.6,color='k',alpha=0.02)

    #Simulated heritability violinplots
    #Environmental variance = 1
    if logx==True:  
        ax.violinplot(
                Vg_sims[i]/(Vg_sims[i]+1),
                positions=[np.log10(params[i][xvar]+offset)],
                widths=2e-1,showmeans=True)
        
        #ax.plot(
        #        np.log10(params[i][xvar]+offset),
        #        Vg_theory/(Vg_theory+1),
        #        'ko',fillstyle='none',markersize=8,
        #        label=r'Theory analytical',alpha=0.7)
        
        ax.plot(
                np.log10(params[i][xvar]+offset),
                Vg_numerical/(Vg_numerical+1),
                'kx',markersize=10,
                label=r'Diffusion approximation MSB',alpha=0.7)
        
    else:
        ax.violinplot(
                Vg_sims[i]/(Vg_sims[i]+1),
                positions=[params[i][xvar]+offset],
                widths=1e-3,showmeans=True)
        
        ax.plot(
                params[i][xvar]+offset, Vg_theory/(Vg_theory+1),
                'ko',fillstyle='none',markersize=8,
                label=r'Theory analytical',alpha=0.7)
        
        ax.plot(
                params[i][xvar]+offset, Vg_numerical/(Vg_numerical+1),
                'kx',markersize=10,
                label=r'Theory numerical',alpha=0.7)
       
    ax.set_ylim([0,0.8])

for _ in plt.get_figlabels():
    plt.figure(_)
    plt.savefig('/home/jason/git/fluctuating_optimum/'+_+'.png')

plt.close('all')



#%%
#===================================================
#multipanel sigma2 dependency
offset=1e-5

#index of variable on x axis
xvar=1

N=10000
fig, axs=plt.subplots(2,2,figsize=[7,7])
axs=axs.flat

unique_params=set([_[:1]+_[2:6] for _ in params if _[-3]==0.1 and _[2]==N])
indices=[_ for _ in range(len(params)) if params[_][-3]==0.1 and params[_][2]==N]
fig_dict=dict(zip(unique_params,axs))

for i in indices:

    L,sigma_e2,N,V_s,mu,a2,theta,rep=params[i]

    a=np.sqrt(a2)

    ax=fig_dict[(L,N,V_s,mu,a2)]
    
    ax.violinplot(
            Vg_sims[i]/(Vg_sims[i]+1),
            positions=[np.log10(sigma_e2+offset)],
            widths=0.25,showmeans=True)
   
    scaleVg=Vg_LB(mu,L,V_s)+np.sqrt(V_s*sigma_e2)
    ax.plot(
            np.log10(params[i][xvar]+offset),
            scaleVg/(scaleVg+1),
            'ko',markersize=10,
            label=r'Diffusion approx. MSB',alpha=0.7)
    
    Vg_numerical=Vg_pred_consistent(5e-1,N,mu,a,L,sigma_e2,V_s)
    
    ax.plot(
            np.log10(params[i][xvar]+offset),
            Vg_numerical/(Vg_numerical+1),
            'kx',markersize=10,
            label=r'Diffusion approx. MSB',alpha=0.7)
        
    ax.set_ylim([0,.8])
    ax.set_title(r'$L=$'+str(L)+r'$,V_s=$'+str(V_s),y=.99,fontsize=11)
                    
    h2_LB=Vg_LB(mu,L,V_s)/(1+Vg_LB(mu,L,V_s))
    ax.axhline(y=h2_LB,color='k',ls='--',label='Latter-Bulmer')
    ax.axhspan(0.1,0.6,color='k',alpha=0.02)

axs[2].set_ylabel(r'Heritability $h^2$',fontsize=14)
axs[2].yaxis.set_label_coords(-0.2,1.1)

tick_list=[r'$0.0$','',r'$0.2$','',r'$0.4$','',r'$0.6$','',r'$0.8$']
axs[0].set_yticklabels(tick_list,fontsize=12)
axs[2].set_yticklabels(tick_list,fontsize=12)
axs[1].set_yticklabels([])
axs[3].set_yticklabels([])

tick_list=['',r'$0$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$']
axs[2].set_xticklabels(tick_list,fontsize=12)
axs[3].set_xticklabels(tick_list,fontsize=12)
axs[0].set_xticklabels([])
axs[1].set_xticklabels([])

axs[2].set_xlabel(r'Fluctuation intensity $\sigma^2$',x=1.15,fontsize=14)

#remove duplicate legend entries
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[0].legend(by_label.values(), by_label.keys(),loc=[0.1,.8],fontsize=10)

#plt.savefig('violinplot_N'+str(N)+'.pdf',bbox_inches='tight')

#ax.set_xlabel(r'Fluctuation intensity $\sigma^2$',fontsize=14)

###################
#%%
#Stablizing picture

x=np.linspace(-2,2.1,1000)

fig,ax=plt.subplots(figsize=[4,2.7])

ax.plot(x,np.exp(-x**2),'r',label='Fitness')
ax.plot(x,np.exp(-10*x**2),'b',label='Population')

ax.set_xticklabels([])
ax.set_xlabel(r'Trait value',fontsize=14)

ax.set_yticklabels([])
ax.set_ylabel(r'Distribution',fontsize=14)

xpos=0.4
plt.annotate(text='', xy=(-xpos,np.exp(-10*xpos**2)), xytext=(xpos,np.exp(-10*xpos**2)), arrowprops=dict(arrowstyle='<->'))
plt.annotate(text=r'$V_g+V_e$',xy=(-0.435,np.exp(-0.15*xpos**2)-0.15),xytext=(-0.435,np.exp(-10*xpos**2)-0.15),fontsize=14)


xpos=0.6
plt.annotate(text='', xy=(-xpos,np.exp(-xpos**2)), xytext=(xpos,np.exp(-xpos**2)), arrowprops=dict(arrowstyle='<->'))
plt.annotate(text=r'$V_s$',xy=(-0.15,np.exp(-xpos**2)-0.15),xytext=(-0.15,np.exp(-xpos**2)-0.15),fontsize=14)


plt.annotate(text=r'Trait optimum',xy=(0,1),xytext=(0,1.1),fontsize=12,arrowprops=dict(arrowstyle='->'))

plt.tight_layout()

plt.legend(loc='upper left',frameon=False)

#plt.savefig("stablizing.eps")

#%%
#Latter-Bulmer predictions

mu=6.6e-9

Ls=np.linspace(1.2e4,1.2e8,10000)
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,20)/(1+Vg_LB(mu,Ls,20)),label=r'$V_s=20V_e$')
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,5)/(1+Vg_LB(mu,Ls,5)),label=r'$V_s=5V_e$')

plt.fill_between(np.log10(Ls),[0.1],[0.6],alpha=0.5)

plt.xlabel(r'Target size (fraction of euchromatic genome)',fontsize=14)
plt.ylabel(r'Heritability',fontsize=14)

plt.gca().set_xticklabels(['',r'$10^{-4}$','',r'$10^{-3}$','',r'$10^{-2}$','',r'$10^{-1}$','',r'$1$'],fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)


plt.ylim([0,1])
plt.tight_layout()
#plt.savefig("LB.pdf")

#%%
#Finite N Latter-Bulmer predictions
mu=6.6e-9

Ls=np.linspace(1.2e4,1.2e8,10000)
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,20)/(1+Vg_LB(mu,Ls,20)),label=r'$V_s=20V_e, N\rightarrow\infty$')
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,5)/(1+Vg_LB(mu,Ls,5)),label=r'$V_s=5V_e, N\rightarrow\infty$')

plt.fill_between(np.log10(Ls),[0.1],[0.6],alpha=0.5)

plt.xlabel(r'Target size (fraction of euchromatic genome)',fontsize=14)
plt.ylabel(r'Heritability',fontsize=14)

plt.gca().set_xticklabels(['',r'$10^{-4}$','',r'$10^{-3}$','',r'$10^{-2}$','',r'$10^{-1}$','',r'$1$'],fontsize=14)
plt.yticks(fontsize=14)

plt.gca().set_prop_cycle(None)

Ls=np.array([10,50,1e2,5e2,1e3,5e3,1e4,5e4,1e5])
mu=mu*1e3

h2_theory=[]
for _ in Ls:

    temp=Vg_pred_consistent(1e-1,1000,mu,0.3,_,0,20)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,h2_theory,'--',label=r'$V_s=20V_e, N=1000$')


h2_theory=[]
for _ in Ls:

    temp=Vg_pred_consistent(1e-1,1000,mu,0.3,_,0,5)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,h2_theory,'--',label=r'$V_s=5V_e, N=1000$')
plt.ylim([0,1])
plt.tight_layout()
plt.legend(fontsize=14)

#plt.savefig("LB_drift.pdf")

#%%
#Moving optimum picture

x=np.linspace(-2,2.1,1000)

fig,ax=plt.subplots(figsize=[4,2.7])

ax.plot(x,np.exp(-(x-0.6)**2),'r',label='Fitness')
ax.plot(x,np.exp(-10*x**2),'b',label='Population')

ax.set_xticklabels([])
ax.set_xlabel(r'Trait value',fontsize=14)

#ax.set_yticklabels([])
ax.set_ylabel(r'Distribution',fontsize=14)

xpos=0.4
plt.annotate(s=r'', xy=(-xpos+0.6,1.05), xytext=(xpos+0.6,1.05), arrowprops=dict(arrowstyle='<->',linewidth='2'))

plt.annotate(s='Trait\noptimum',xy=(0.6,1),xytext=(0.22,.6),fontsize=12,arrowprops=dict(arrowstyle='->'))

plt.ylim([0,1.1])

plt.tight_layout()

plt.legend(loc='upper left',frameon=False)

plt.savefig("stablizing_fluc.eps")

#%%
#Chasing optimum picture

x=np.linspace(-2,2.1,1000)

fig,ax=plt.subplots(figsize=[4,2.7])

ax.plot(x,np.exp(-(x-1.5)**2),'r',label='Fitness')
ax.plot(x,np.exp(-10*x**2),'b',label='Population')

ax.set_xticklabels([])
ax.set_xlabel(r'Trait value',fontsize=14)

#ax.set_yticklabels([])
ax.set_ylabel(r'Distribution',fontsize=14)

xpos=0.4
plt.annotate(s='', xy=(-0.5,1.0), xytext=(0.8,1.0), arrowprops=dict(arrowstyle='<-',linewidth='2'))

plt.annotate(s=r'Rate $\propto V_g$',xy=(0.,1.025), xytext=(-0.4,1.025))

plt.ylim([0,1.1])

plt.tight_layout()

plt.legend(loc='upper left',frameon=False)

plt.savefig("stablizing_lande.eps")


#%%

sigma2=1e-2
theta=0e-2

plt.figure(figsize=[3,2])
ax=plt.gca()

opts=np.zeros(1000)
for _ in range(1,1000):
    opts[_]=(1-theta)*opts[_-1] + np.random.normal(0,np.sqrt(sigma2))

ax.set_yticklabels([])
ax.set_xticklabels([])

ax.set_ylabel('Trait optimum')
ax.set_xlabel('Time')

plt.plot(opts)
plt.tight_layout()

%plt.savefig("random_walk.pdf")

#%%
#Latter-Bulmer_fluctuation predictions 

mu=6.6e-9
sigma2=1e-2

Ls=np.exp(np.linspace(np.log(1.2e4),np.log(1.2e8),20))
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,20)/(1+Vg_LB(mu,Ls,20)),label=r'$V_s=20, \sigma^2=0$')
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,5)/(1+Vg_LB(mu,Ls,5)),label=r'$V_s=5, \sigma^2=0$ ')


#rescale for computational stability
Ls=1e-3*Ls
mu=mu*1e3

plt.gca().set_prop_cycle(None)

h2_theory=[]
for _ in Ls:
    temp=Vg_pred_consistent(1e-1,10000,mu,0.3,_,sigma2,20)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,h2_theory,'--',label=r'$V_s=20, \sigma^2=10^{-2}$')

h2_theory=[]
for _ in Ls:
    temp=Vg_pred_consistent(1e-1,10000,mu,0.3,_,sigma2,5)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,h2_theory,'--',label=r'$V_s=5, \sigma^2=10^{-2}$')

plt.fill_between(np.log10(Ls)+3,[0.1],[0.6],alpha=0.5)

plt.xlabel(r'Target size (fraction of euchromatic genome)',fontsize=14)
plt.ylabel(r'Heritability',fontsize=14)

plt.gca().set_xticklabels(['',r'$10^{-4}$','',r'$10^{-3}$','',r'$10^{-2}$','',r'$10^{-1}$','',r'$1$'],fontsize=14)
plt.yticks(fontsize=14)

plt.ylim([0,1])
plt.tight_layout()
plt.legend(fontsize=12)

#plt.figure()
#plt.plot(h2_theory/(Vg_LB(mu,Ls,5)/(1+Vg_LB(mu,Ls,5))))


#plt.savefig("LB_fluc.pdf")
