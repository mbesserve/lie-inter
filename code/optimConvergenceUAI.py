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

country = countryList[0]

dat = np.load('countTableUAI_'+country+'_'+str(year)+'.npz')
Aorg = dat['A']
yorg = dat['y']
S = dat['S']
units = dat['fpUnits']

impacts = dat['impacts']
sectors = dat['sectors']
 

# create the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

f = deq.leontief().double() 


Stens = torch.tensor(copy.deepcopy(S)).unsqueeze_(0)

co2Opt = []
emplOpt = []
parsOpt = []
residOpt = []
dimRange =[5,10,20,50,100,200]
betaRange = [1.0,2.0]
max_iterRange = [20,200,2000,20000]

residOpt.append([])
for kbeta in betaRange:
    residOpt[kbeta].append([])
    for kiter in max_iterRange:
        residOpt[kbeta][kiter].append([])
        model = deq.DEQFixedPoint(f, deq.anderson, tol=1e-4, max_iter=max_iterRange[kiter], beta=betaRange[kbeta]).to(device)
#model = nn.Sequential(deq.diagmulsel(Aorg.shape[0],Aorg.shape[0]),deq.DEQFixedPoint(f, deq.anderson, tol=1e-4, max_iter=2000, beta=2.0)).to(device)
        
        for kdim in dimRange:
            A = torch.tensor(copy.deepcopy(np.concatenate((Aorg[:dimRange[kdim],:dimRange[kdim]],np.sum(yorg,1,keepdims=True)),1)),requires_grad=True).unsqueeze(0)
        
            x = model(A[])
    
            dx= x- f(x,A)
            residOpt[kbeta][kiter][kdim].append(copy.deepcopy((torch.norm(dx)/torch.norm(x)).detach().numpy()))

        F = torch.matmul(Stens, x)
        empl = torch.sum(Stens[:,9:14,:],axis = 1)* x[:,:,0]
        #GWP 100 for CO2, CH4 and N2O
        co2 = 1*F[:,23]+25*F[:,24]+298*F[:,25]
        co20 = co2.detach()
        # optimize    
        opt = optim.Adam(model.parameters(), lr=1e-3)
        x0 = x.detach()
        empl0 = empl.detach()    
        params = [param for param in model.parameters()]
        
    #    co2Opt.append([])
    #    emplOpt.append([])
        for kite in range(Niter):
            x = model(A)
            F = torch.matmul(Stens, x)
            empl = torch.sum(Stens[:,9:14,:],axis = 1)* x[:,:,0]
            #GWP 100 for CO2, CH4 and N2O
            co2 = 1*F[:,23]+25*F[:,24]+298*F[:,25]
       
            # optimize
            eloss = nn.L1Loss()(empl,empl0)
            
            loss = co2+lmbda*eloss
            opt.zero_grad()
            loss.backward()
            opt.step()
        co2Opt.append(copy.deepcopy(co2.detach().numpy()))
        emplOpt.append(copy.deepcopy(empl.detach().numpy()))#empl.detach().numpy()))
        parsOpt.append(copy.deepcopy(params[0].detach().numpy()))
        xOpt.append(copy.deepcopy(x.detach().numpy()))
    
    co20mat = copy.deepcopy(co20.numpy())
    co2Optmat = np.array(co2Opt)
    emplOptmat = np.array(emplOpt)
    parsOptmat = np.array(parsOpt)
#    emplVec = S[[1],:]*xOptmat[:,0,:,0]
    empl0mat = copy.deepcopy(empl0.numpy())
    xOptmat = np.array(xOpt)
    x0mat = copy.deepcopy(x0.numpy())
    
    np.savez('resOptUAI_'+country+'_'+str(year)+'.npz',xOptmat=xOptmat,x0mat=x0mat,
             emplOptmat=emplOptmat,empl0mat=empl0mat,parsOptmat=parsOptmat,
             co2Optmat=co2Optmat,co20mat=co20mat,lambdaList=lambdaList,Niter=Niter)
    
