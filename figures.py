# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:27:44 2023

@author: jason
"""

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