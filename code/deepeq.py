#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 09:34:09 2021

@author: besserve
"""

import torch
import torch.nn as nn


import torch.autograd as autograd
from torch.autograd import gradcheck

import torch.optim as optim
    
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
        #torch.solve deprecated
        #alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        alpha = torch.linalg.solve( H[:,:n+1,:n+1],y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        #print(alpha.shape)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res



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



class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, A):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z,A), torch.zeros_like(A[:,:,[1]]), **self.kwargs)
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


class diagmulsel(nn.Module):
    def __init__(self,n_sectors,n_sel):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_sel = n_sel
        
        self.d = torch.nn.Parameter(torch.zeros(self.n_sel).double())
        
    # use a true multiplicative lie intervention (applies to vector y too), that is the difference with previous version of this code (leontiefCusterv3)
    def forward(self,A):
        # return torch.cat((torch.matmul(torch.diag(torch.cat((torch.sigmoid(self.d),torch.ones(self.n_sectors-self.n_sel).double()),0)),A[:,:,:-1]),A[:,:,[-1]]),2)
        return torch.matmul(torch.diag(torch.cat((2*torch.sigmoid(self.d),torch.ones(self.n_sectors-self.n_sel, device=A.device).double()),0)),A)
               

class leontief(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,z,A):
        return torch.matmul(A,torch.cat((z,torch.ones(1,1,1,device=A.device).double()),1))
    
    
    
def clusterMatrix(Xfeats,Nclust,clustMethod='NMF',selectPureClusters=False,purityThres=.1) :
    # clusters the rows of the matrix-> returns nrow labels
    if clustMethod == 'NMF' :
            from sklearn.decomposition import NMF
            model = NMF(n_components=Nclust, init='random', random_state=0)
            W = model.fit_transform(Xfeats)
            H = model.components_
            meanPatts = H
            clustLabels = np.argmax(W,axis=1)
            
            # compute purity ratios
            if selectPureClusters :
                purity = np.zeros([W.shape[0]])
                for mapIdx in range(W.shape[0]) :
                    purity[mapIdx] = (np.sum((W[mapIdx,clustLabels[mapIdx]]*H[clustLabels[mapIdx],:])**2)/
                                      np.sum(Xfeats[mapIdx,:]**2))
                    
                clustLabels[purity<purityThres]=[-1]*np.sum(purity<purityThres)
        
    else: # use kmeans...
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=Nclust, random_state=0).fit(Xfeats)
        meanPatts = kmeans.cluster_centers_
        clustLabels = kmeans.labels_
        purity = []
    return clustLabels, meanPatts
