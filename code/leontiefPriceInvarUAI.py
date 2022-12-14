#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 09:34:09 2021

@author: anonymous
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    
def interMatrix(interList,outDim):
    
    if len(interList)>0:
        if max(interList)>outDim+1: 
            raise ValueError
    
    M = torch.zeros(outDim,len(interList),dtype=torch.double)
    cVec = torch.ones(outDim,1,dtype=torch.double)
    for kinter,inter in enumerate(interList):
        M[inter,kinter].fill_(1.)
        cVec[inter].fill_(0.)
        
    return M,cVec
    
class priceLeontief(nn.Module):
    """This implements a simplified shadow price rebound model:
        
    .. math::
        x^* = A x^* + y (p^*)
        
        p^* = A^T p^* + A^T \delta_E

    """
    def __init__(self,n_sectors,deltaE,dCurve,interSec):
        super().__init__()
        self.deltaE = deltaE
        self.n_sectors = n_sectors
        self.demandCurve = DemCurve(self.n_sectors,dCurve,interSec)
        
    def forward(self,z,A) :
        return torch.cat((
            torch.matmul(A,z[:,0:self.n_sectors,:]) + self.demandCurve(z[:,self.n_sectors:,:]),
            torch.matmul(A.transpose(1,2),z[:,self.n_sectors:,:])+ torch.matmul(A.transpose(1,2),deltaE)
            ),1)
        
    
def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, d1 = x0.shape
    X = torch.zeros(bsz, m, d*d1, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*d1, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        #deprecated
        #alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res


import matplotlib.pyplot as plt


def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res


import torch.autograd as autograd

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, A):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(
                lambda z : self.f(z,A), 
                torch.cat((torch.zeros_like(A[:,:,[1]]),torch.zeros_like(A[:,:,[1]])),1),
                **self.kwargs)
        z = self.f(z,A)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,A)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z


from torch.autograd import gradcheck
# run a very small network with double precision, iterating to high precision
def dCurve(p) :
#    return 1/(p+1)
#    return 1.42/((p/.41)*(p/.4)*(p/.4)*(p/.4)+1)
    return 1.42/(torch.exp(10*((p/.4)-1))+1)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
                
        self.input_fc = nn.Linear(input_dim, 20).double()
        self.hidden_fc = nn.Linear(20, 10).double()
        self.output_fc = nn.Linear(10, output_dim).double()
        
    def forward(self, x):
        
        #x = [batch size, height, width]
        
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        #x = [batch size, height * width]
        
        h_1 = F.relu(self.input_fc(x))
        #h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))
        #h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)
        #y_pred = [batch size, output dim]
        
        return y_pred


class DemCurve(nn.Module):
    def __init__(self,n_sectors,dCurve,interSect):
        super(DemCurve, self).__init__()
        self.dCurve = dCurve
        self.softInt = nn.ModuleList([ MLP(1,1) for i in range(len(interSect)) ])
        self.interSect = interSect
        self.M, self.cVec = interMatrix(self.interSect, n_sectors)
        
    def forward(self,p):
        if len(self.interSect)>0:
            intVec = torch.cat([f(p[:,self.interSect[k],0]) for k,f in enumerate( self.softInt)],dim = 1)
            p = p + torch.matmul(torch.ones(p.shape[0],1,1)*self.M.unsqueeze(0), intVec.unsqueeze(2)) 
        #for ksec, sec in enumerate(self.interSect):
        #    pint[:,sec] = pint[:,sec]+self.softInt[ksec](pint[:,sec])
        return dCurve(p)

class diagmul(nn.Module):
    def __init__(self, n_sectors, interSect):
        super().__init__()
        self.interSect = interSect
        self.inter = torch.nn.Parameter(torch.ones( len(self.interSect),1).double() )
        
        #self.d = torch.ones( n_sectors, requires_grad = False, dtype=torch.double)
        self.M, self.cVec = interMatrix(self.interSect, n_sectors)
        
        #for ksec, sec in enumerate(self.interSect):
         #   self.d[sec] = self.inter[ksec]

    def forward(self,A):
        d = torch.matmul(self.M,self.inter)+self.cVec
        
        return torch.matmul(torch.diag(1-torch.relu(1-torch.relu(d.squeeze()))).unsqueeze(0),A)

        
torch.manual_seed(0)


nsec = 3
deltaE = torch.zeros(1,3,1,dtype=torch.double)
enSecs = [1]
for ksec in range(nsec):
    if ksec in enSecs:
        deltaE[:,ksec,:] = 1.
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


auxVar = [0]
#deq = DEQFixedPoint(f, anderson, tol=1e-10, max_iter=5000).double()
A = torch.tensor([[[0,0.1,0],[.5,0,0],[.1,.2,.0]]]).double()



interVar = [1]

lieModel = diagmul(3,interVar)

f = priceLeontief(nsec, deltaE, dCurve, auxVar).double() #ResNetLayer(2,2, num_groups=2).double

model = nn.Sequential(lieModel,
                      DEQFixedPoint(f, anderson, tol=1e-4, max_iter=5000, beta=2.0)
                      ).to(device)

fref = priceLeontief(nsec, deltaE, dCurve, []).double() #ResNetLayer(2,2, num_groups=2).double

modelRef = nn.Sequential(
                      DEQFixedPoint(fref, anderson, tol=1e-4, max_iter=5000, beta=2.0)
                      ).to(device)

modelLie = nn.Sequential(lieModel,
                      DEQFixedPoint(fref, anderson, tol=1e-4, max_iter=5000, beta=2.0)
                      ).to(device)

import torch.optim as optim

import copy
dc = copy.deepcopy

pars = dict([parms for parms in model.named_parameters()])
interPars = pars['0.inter']
auxFunW = pars['1.f.demandCurve.softInt.0.input_fc.weight']
auxFunB = pars['1.f.demandCurve.softInt.0.input_fc.bias']
auxFunWh = pars['1.f.demandCurve.softInt.0.hidden_fc.weight']
auxFunBh = pars['1.f.demandCurve.softInt.0.hidden_fc.bias']
auxFunWo = pars['1.f.demandCurve.softInt.0.output_fc.weight']
auxFunBo = pars['1.f.demandCurve.softInt.0.output_fc.bias']
lieInter = pars['0.inter']

# checking parameters are shared
#parsLie = dict([parms for parms in lieModel.named_parameters()])
#id(parsLie['inter'])


parsRef = dict([parms for parms in modelRef.named_parameters()])

Aref = A
Aref.requires_grad_(True)
x = modelRef(Aref)
x0 = x.detach()
opt = optim.Adam([auxFunW,auxFunB,auxFunWh,auxFunBh,auxFunWo,auxFunBo], lr=1e-3)



lieIntList = torch.arange(.2,1,.2)
    
mse = []
Xlie = []
Xopt = []
Xref = []
PriceInvar = []
mask = torch.zeros_like(A,requires_grad=False)
mask[:,0,1].fill_(1.)
Niter = 10000
invarVar = 3
nbatch = 10

for lieInt in lieIntList:
    with torch.no_grad():
        lieInter.fill_(lieInt)
    for kite in range(Niter):
        Arnd = A+(.18*torch.randn(nbatch,A.shape[1],A.shape[2])-.09)*mask
        Arnd.requires_grad_(True)
        x = model(Arnd)
        xref = modelRef(Arnd)
        loss = nn.MSELoss()(model[1].f.demandCurve.softInt[0](x[:,invarVar,:])+x[:,invarVar,:],xref[:,invarVar,:])#+torch.norm(params[0][:2])
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        
    Aref = A+torch.arange(-.09,.09,.002).unsqueeze(1).unsqueeze(2)*mask
    Aref.requires_grad_(True)
    x = model(Aref)
    xopt = x.detach().numpy()    
    PriceInvar.append(dc((model[1].f.demandCurve.softInt[0](x[:,invarVar,:])+x[:,invarVar,:]).detach().numpy()))
    
    x = modelRef(Aref)
    xref = x.detach().numpy()
    x = modelLie(Aref)
    xlie = x.detach().numpy()
    mse.append(dc(loss.detach().numpy()))
    Xopt.append(dc(xopt))
    Xlie.append(dc(xlie))
    Xref.append(dc(xref))
    

    
import numpy as np
#priceInvar = np.array([(model[1].f.demandCurve.softInt[0](torch.tensor(x)[:,invarVar,:])+torch.tensor(x)[:,invarVar,:]).detach().numpy() for x in Xopt])
#priceLie = 
        
        
priceInvar = np.array(PriceInvar)
xlieA = np.array(Xlie)
xrefA = np.array(Xref)
xoptA = np.array(Xopt)


# plt.figure()
# plt.plot(xrefA[:,:,invarVar,0].T,priceInvar[:,:,0].T)
# plt.plot(xrefA[:,:,invarVar,0].T,xlieA[:,:,invarVar,0].T,'--')
    
klie = 2
# plt.figure()
# plt.plot(xrefA[klie,:,invarVar,0].T,priceInvar[klie,:,0].T)
# plt.plot(xrefA[klie,:,invarVar,0].T,xlieA[klie,:,invarVar,0].T)
# plt.plot(xrefA[klie,:,invarVar,0].T,xrefA[klie,:,invarVar,0].T,'k--')
# plt.legend(['intervened invariant price','lie intervened price','reference price'])
# plt.xlabel('reference price')    


Arange = Aref[:,0,1].detach().numpy()
klie = 2
plt.figure()
plt.plot(Arange,priceInvar[klie,:,0].T)
plt.plot(Arange,xlieA[klie,:,invarVar,0].T)
plt.plot(Arange,xrefA[klie,:,invarVar,0].T,'k--')
plt.legend(['intervened invariant price','lie intervened price','reference price'])
plt.xlabel('parameter value')    

plt.figure()
plt.plot(Arange,xoptA[klie,:,interVar[0],0].T)
plt.plot(Arange,xlieA[klie,:,interVar[0],0].T)
plt.plot(Arange,xrefA[klie,:,interVar[0],0].T,'k--')
plt.legend(['invar. intervened energy','lie intervened energy','reference energy'])
plt.xlabel('parameter value')    


np.savez('priceInvarUAInonlin',PriceInvar=PriceInvar,Xlie=Xlie,Xref=Xref,Xopt=Xopt,invarVar=invarVar,Niter=Niter,nbatch=nbatch)

    