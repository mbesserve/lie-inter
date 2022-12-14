#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:02:17 2021

@author: besserve
"""
import deepeq as deq
import numpy as np
from torch import nn
import torch 
import torch.optim as optim
import copy

countryList = ['FR','DE','IT','US','GB']
year = 2018


 

# create the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

f = deq.leontief().double() 


residOpt = []
dimRange =[5,10,20,50,100,200]
betaRange = [1.0,2.0]
max_iterRange = [20,200]

for ky, year in enumerate(range(2012,2019)):
    residOpt.append([])
    for kc,country in enumerate(countryList):
        residOpt[ky].append([])
        dat = np.load('countTableUAI_'+country+'_'+str(year)+'.npz')
        Aorg = dat['A']
        yorg = dat['y']
        for kbeta,beta in enumerate(betaRange):
            residOpt[ky][kc].append([])
            for kiter,maxiter in enumerate(max_iterRange):
                residOpt[ky][kc][kbeta].append([])
                model = deq.DEQFixedPoint(f, deq.anderson,  m=5, lam=1e-4,tol=1e-12, max_iter=max_iterRange[kiter], beta=betaRange[kbeta]).to(device)
        #model = nn.Sequential(deq.diagmulsel(Aorg.shape[0],Aorg.shape[0]),deq.DEQFixedPoint(f, deq.anderson, tol=1e-4, max_iter=2000, beta=2.0)).to(device)
                
                for kdim,dim in enumerate(dimRange):
                    #residOpt[kbeta][kiter].append([])
                    A = torch.tensor(copy.deepcopy(np.concatenate((Aorg[:dimRange[kdim],:dimRange[kdim]],np.sum(yorg[:dimRange[kdim],:],1,keepdims=True)),1)),requires_grad=True).unsqueeze(0)
                
                    x = model(A)
            
                    dx= x- f(x,A)
                    residOpt[ky][kc][kbeta][kiter].append(copy.deepcopy((torch.norm(dx)/torch.norm(x)).detach().numpy()))
    
    



ropt = np.array(residOpt)
import matplotlib.pyplot as plt
plt.figure(20)
plt.clf()
plt.semilogy(np.mean(ropt,axis=(0,1))[0,1,:].T,label=r'Anderson $\beta=1.0$')
plt.semilogy(np.mean(ropt,axis=(0,1))[0,1,:].T+np.std(ropt,axis=(0,1))[0,1,:].T,linestyle='--',color=plt.gca().lines[-1].get_color(),alpha=.5)
#plt.semilogy(np.mean(ropt,axis=(0,1))[0,1,:].T-np.std(ropt,axis=(0,1))[0,1,:].T,linestyle='--',color=plt.gca().lines[-1].get_color())

#plt.figure(21)
#plt.clf()
plt.semilogy(np.mean(ropt,axis=(0,1))[1,1,:].T,label=r'Anderson $\beta=2.0$')
plt.semilogy(np.mean(ropt,axis=(0,1))[1,1,:].T+np.std(ropt,axis=(0,1))[1,1,:].T,linestyle='--',color=plt.gca().lines[-1].get_color(),alpha=.5)
#plt.semilogy(np.mean(ropt,axis=(0,1))[1,1,:].T-np.std(ropt,axis=(0,1))[1,1,:].T,linestyle='--',color=plt.gca().lines[-1].get_color())




residOpt = []
dimRange =[5,10,20,50,100,200]
betaRange = [1.0,2.0]
max_iterRange = [20,200,2000,20000]
for ky, year in enumerate(range(2012,2019)):
    residOpt.append([])
    for kc,country in enumerate(countryList):
        residOpt[ky].append([])
        dat = np.load('countTableUAI_'+country+'_'+str(year)+'.npz')
        Aorg = dat['A']
        yorg = dat['y']
        #residOpt.append([])
        for kbeta,beta in enumerate(betaRange):
            residOpt[ky][kc].append([])
            for kiter,maxiter in enumerate(max_iterRange):
                residOpt[ky][kc][kbeta].append([])
                model = deq.DEQFixedPoint(f, deq.forward_iteration,tol=1e-12, max_iter=max_iterRange[kiter]).to(device)
        #model = nn.Sequential(deq.diagmulsel(Aorg.shape[0],Aorg.shape[0]),deq.DEQFixedPoint(f, deq.anderson, tol=1e-4, max_iter=2000, beta=2.0)).to(device)
                
                for kdim,dim in enumerate(dimRange):
                    #residOpt[kbeta][kiter].append([])
                    A = torch.tensor(copy.deepcopy(np.concatenate((Aorg[:dimRange[kdim],:dimRange[kdim]],np.sum(yorg[:dimRange[kdim],:],1,keepdims=True)),1)),requires_grad=True).unsqueeze(0)
                
                    x = model(A)
            
                    dx= x- f(x,A)
                    residOpt[ky][kc][kbeta][kiter].append(copy.deepcopy((torch.norm(dx)/torch.norm(x)).detach().numpy()))
        
    
    


roptf = np.array(residOpt)
import matplotlib.pyplot as plt
#plt.figure(22)
#plt.clf()
plt.semilogy(np.mean(roptf,axis=(0,1))[0,1,:].T,label=r'forward iteration')
plt.semilogy(np.mean(roptf,axis=(0,1))[0,1,:].T+np.std(roptf,axis=(0,1))[0,1,:].T,linestyle='--',color=plt.gca().lines[-1].get_color(),alpha=.5)
#plt.semilogy(np.mean(roptf,axis=(0,1))[0,1,:].T-np.std(roptf,axis=(0,1))[0,1,:].T,linestyle='--',color=plt.gca().lines[-1].get_color())


plt.legend()#['Anderson \beta=1.0','Anderson \beta=2.0','forward iteration'])

plt.xticks(ticks=range(len(dimRange)),labels=[str(num) for num in dimRange])
plt.xlabel('SCM dimension')
plt.title('Equilibrium relative error')




# residOpt = []
# dimRange =[5,10,20,50,100,200]
# betaRange = [1.0,2.0]
# max_iterRange = [20,200,2000,20000]

# for kc,country in enumerate(countryList):
#     residOpt.append([])
#     dat = np.load('countTableUAI_'+country+'_'+str(year)+'.npz')
#     Aorg = dat['A']
#     yorg = dat['y']
#     #residOpt.append([])
#     for kbeta,beta in enumerate(betaRange):
#         residOpt[kc].append([])
#         for kiter,maxiter in enumerate(max_iterRange):
#             residOpt[kc][kbeta].append([])
#             model = deq.DEQFixedPoint(f, deq.forward_iteration,tol=1e-12, max_iter=max_iterRange[kiter]).to(device)
#     #model = nn.Sequential(deq.diagmulsel(Aorg.shape[0],Aorg.shape[0]),deq.DEQFixedPoint(f, deq.anderson, tol=1e-4, max_iter=2000, beta=2.0)).to(device)
            
#             for kdim,dim in enumerate(dimRange):
#                 #residOpt[kbeta][kiter].append([])
#                 A = torch.tensor(copy.deepcopy(np.concatenate((Aorg[:dimRange[kdim],:dimRange[kdim]],np.sum(yorg[:dimRange[kdim],:],1,keepdims=True)),1)),requires_grad=True).unsqueeze(0)
            
#                 x = torch.linalg.solve(torch.tensor(copy.deepcopy(np.eye(dimRange[kdim])-Aorg[:dimRange[kdim],:dimRange[kdim]])  ),torch.tensor(np.sum(yorg[:dimRange[kdim],:],1,keepdims=True)))
        
#                 dx= x.unsqueeze(0)- f(x.unsqueeze(0),A)
#                 residOpt[kc][kbeta][kiter].append(copy.deepcopy((torch.norm(dx)/torch.norm(x)).detach().numpy()))
    




# roptinv = np.array(residOpt)
# import matplotlib.pyplot as plt
# #plt.figure(22)
# #plt.clf()
# plt.semilogy(np.mean(roptinv,axis=0)[0,1,:].T,label=r'linear equation solver')


# plt.legend()#['Anderson \beta=1.0','Anderson \beta=2.0','forward iteration'])

# plt.xticks(ticks=range(len(dimRange)),labels=[str(num) for num in dimRange])
# plt.xlabel('SCM dimension')
# plt.title('Equilibrium relative error')

    

    
