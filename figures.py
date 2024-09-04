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


def Vg_pred(Vg,N,mu,a,L,sigma_e2,V_s):
    s=a**2/(2*V_s)
    sigma_c2=a**2*sigma_e2/(Vg)**2
    return 2*a**2*L*E_H(N,mu,sigma_c2,s)

def Vg_pred_consistent(init,N,mu,a,L,sigma_e2,V_s):
    res = (scipy.optimize.minimize(lambda x: 
            (Vg_pred(x,N,mu,a,L,sigma_e2,V_s)-x)**2, init))
    return res.x[0]


def Vg_theory_opt(x,N,a,so2,L,mu,Vs):
    
    if so2==0:
        return x-4*L*mu*V_s
    
    else: 
        sc2=a**2*so2/x**2
        d=np.abs(0.5*(1-np.sqrt(1+4/(N*sc2))))
        b=x**2/(Vs*so2)-2*N*mu
        b_t=b/(d+1)
    
        return ((2*N*mu)*(2*L*a**2)*d**b_t*(d**(1-b_t)
                -(0.5+d)**(1-b_t))/(b_t-1)-x)


#%%
#Load data

with open("Vg_sims",'r') as fin:
    params=ast.literal_eval(fin.readline()[2:-1])

Vg_sims=np.loadtxt("Vg_sims")

#parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep

#%%
########################################
#small offset to avoid log(0)
offset=1e-7

#index of variable on x axis
xvar=5

for i in range(len(params)):

    plt.figure(str([_ for ind,_ in enumerate(params[i]) if ind != xvar]))
    ax=plt.gca() 
     
    #Simulated heritability violinplots
    #Environmental variance = 1
    ax.violinplot(
            Vg_sims[i]/(Vg_sims[i]+1),
            positions=[params[i][xvar]+offset],
            widths=1e-2,showmeans=True)
    
    L,sigma_e2,N,V_s,mu,a2,theta,rep=params[i]
    a=np.sqrt(a2)
    
    Vg_theory=scipy.optimize.fsolve(
            lambda x: Vg_theory_opt(x,N,a,sigma_e2,L,mu,V_s),0.025)[0]
    ax.plot(
            params[i][xvar]+offset, Vg_theory/(Vg_theory+1),
            'ko',fillstyle='none',markersize=8,
            label=r'Theory analytical',alpha=0.7)
    
    Vg_numerical=Vg_pred_consistent(2e-1,N,mu,a,L,sigma_e2,V_s)
    ax.plot(
            params[i][xvar]+offset, Vg_numerical/(Vg_numerical+1),
            'kx',markersize=10,
            label=r'Theory numerical',alpha=0.7)
    
    ax.set_ylim([0,.5])

for _ in plt.get_figlabels():
    plt.figure(_)
    plt.savefig('/home/jason/git/fluctuating_optimum/'+_+'.png')

plt.close('all')

#%%
#multipanel sigma2 dependency
offset=1e-7


fig, axs=plt.subplots(3,2,figsize=[7,7])
axs=axs.flat

fig_dict=dict(zip(unique_params,axs))


for i in range(len(params)):

    L,sigma_e2,N,V_s,mu,a2,theta,rep=params[i]

    #a=np.sqrt(Vm/(2*L*mu))
    a=np.sqrt(a2)

    ax=fig_dict[str([L,N,V_s,mu,a2])]
    
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

#%%
#Stablizing picture

def Vg(mu,L,Vs):
    return 4*mu*L*Vs

x=np.linspace(-2,2.1,1000)

fig,ax=plt.subplots(figsize=[4,2.7])

ax.plot(x,np.exp(-x**2),'r',label='Fitness')
ax.plot(x,np.exp(-10*x**2),'b',label='Population')

ax.set_xticklabels([])
ax.set_xlabel(r'Trait value',fontsize=14)

ax.set_yticklabels([])
ax.set_ylabel(r'Distribution',fontsize=14)

xpos=0.4
plt.annotate(s='', xy=(-xpos,np.exp(-10*xpos**2)), xytext=(xpos,np.exp(-10*xpos**2)), arrowprops=dict(arrowstyle='<->'))
plt.annotate(s=r'$V_g+V_e$',xy=(-0.435,np.exp(-0.15*xpos**2)-0.15),xytext=(-0.435,np.exp(-10*xpos**2)-0.15),fontsize=14)


xpos=0.6
plt.annotate(s='', xy=(-xpos,np.exp(-xpos**2)), xytext=(xpos,np.exp(-xpos**2)), arrowprops=dict(arrowstyle='<->'))
plt.annotate(s=r'$V_s$',xy=(-0.15,np.exp(-xpos**2)-0.15),xytext=(-0.15,np.exp(-xpos**2)-0.15),fontsize=14)


plt.annotate(s=r'Trait optimum',xy=(0,1),xytext=(0,1.1),fontsize=12,arrowprops=dict(arrowstyle='->'))

plt.tight_layout()

plt.legend(loc='upper left',frameon=False)

plt.savefig("stablizing.eps")

#%%
#Latter-Bulmer predictions

mu=6.6e-9

Ls=np.linspace(1.2e4,1.2e8,10000)
plt.plot(np.log10(Ls),Vg(mu,Ls,20)/(1+Vg(mu,Ls,20)),label=r'$V_s=20V_e$')
plt.plot(np.log10(Ls),Vg(mu,Ls,5)/(1+Vg(mu,Ls,5)),label=r'$V_s=5V_e$')

plt.fill_between(np.log10(Ls),[0.1],[0.6],alpha=0.5)

plt.xlabel(r'Target size (fraction of euchromatic genome)',fontsize=14)
plt.ylabel(r'Heritability',fontsize=14)

plt.gca().set_xticklabels(['',r'$10^{-4}$','',r'$10^{-3}$','',r'$10^{-2}$','',r'$10^{-1}$','',r'$1$'],fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)


plt.ylim([0,1])
plt.tight_layout()
plt.savefig("LB.pdf")

#%%
#Finite N Latter-Bulmer predictions
mu=6.6e-9

Ls=np.linspace(1.2e4,1.2e8,10000)
plt.plot(np.log10(Ls),Vg(mu,Ls,20)/(1+Vg(mu,Ls,20)),label=r'$V_s=20V_e, N\rightarrow\infty$')
plt.plot(np.log10(Ls),Vg(mu,Ls,5)/(1+Vg(mu,Ls,5)),label=r'$V_s=5V_e, N\rightarrow\infty$')

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
#Chasing optimim picture

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
#Latter-Bulmer_fluctuation predictions 

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

plt.savefig("random_walk.pdf")

#%%
mu=6.6e-9

Ls=np.linspace(1.2e4,1.2e8,10000)
plt.plot(np.log10(Ls),Vg(mu,Ls,20)/(1+Vg(mu,Ls,20)),label=r'$V_s=20V_e, \sigma^2=0$')
plt.plot(np.log10(Ls),Vg(mu,Ls,5)/(1+Vg(mu,Ls,5)),label=r'$V_s=5V_e, \sigma^2=0$ ')


Ls=np.array([10,25,50,75,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,5e4,1e5])
mu=mu*1e3

# h2_theory=[]
# for _ in Ls:
#     temp=Vg_pred_consistent(1e-1,1000,mu,0.3,_,0,20)
#     h2_theory.append(temp/(1+temp))

# plt.plot(np.log10(Ls)+3,h2_theory,label=r'$V_s=20V_e, N=1000$')


# h2_theory=[]
# for _ in Ls:
#     temp=Vg_pred_consistent(1e-1,1000,mu,0.3,_,0,5)
#     h2_theory.append(temp/(1+temp))

# plt.plot(np.log10(Ls)+3,h2_theory,label=r'$V_s=5V_e, N=1000$')

plt.gca().set_prop_cycle(None)

h2_theory=[]
for _ in Ls:

    temp=Vg_pred_consistent(1e-1,10000,mu,0.3,_,1e-3,20)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,h2_theory,'--',label=r'$V_s=20V_e, \sigma^2=10^{-2}V_e$')


h2_theory=[]
for _ in Ls:

    temp=Vg_pred_consistent(1e-1,10000,mu,0.3,_,1e-3,5)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,h2_theory,'--',label=r'$V_s=5V_e, \sigma^2=10^{-2}V_e$')


plt.fill_between(np.log10(Ls)+3,[0.1],[0.6],alpha=0.5)

plt.xlabel(r'Target size (fraction of euchromatic genome)',fontsize=14)
plt.ylabel(r'Heritability',fontsize=14)

plt.gca().set_xticklabels(['',r'$10^{-4}$','',r'$10^{-3}$','',r'$10^{-2}$','',r'$10^{-1}$','',r'$1$'],fontsize=14)
plt.yticks(fontsize=14)

plt.ylim([0,1])
plt.tight_layout()
plt.legend(fontsize=12)

#plt.savefig("LB_fluc.pdf")
