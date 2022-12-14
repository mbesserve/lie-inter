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
for year in range(2012,2019):

    for country in countryList:
    
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
        
        model = nn.Sequential(deq.diagmulsel(Aorg.shape[0],Aorg.shape[0]),deq.DEQFixedPoint(f, deq.anderson, tol=1e-4, max_iter=2000, beta=2.0)).to(device)
        
        A = torch.tensor(copy.deepcopy(np.concatenate((Aorg,np.sum(yorg,1,keepdims=True)),1))).unsqueeze_(0)
        Stens = torch.tensor(copy.deepcopy(S)).unsqueeze_(0)
        
        
        lambdaList = np.logspace(7, 14,12)
        Niter = 20000
        co2Opt = []
        emplOpt = []
        parsOpt = []
        xOpt = []
        for klmbda, lmbda in enumerate(lambdaList):
            model = nn.Sequential(deq.diagmulsel(Aorg.shape[0],Aorg.shape[0]),deq.DEQFixedPoint(f, deq.anderson, tol=1e-4, max_iter=2000, beta=2.0)).to(device)
            x = model(A)
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
        
