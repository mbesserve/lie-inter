#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 08:49:37 2021

@author: besserve
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

countryList = ['FR','DE']#,'US']
Nc = len(countryList)
plt.close('all')
fig, ax  = plt.subplots(2,Nc,figsize = [10,10])

for kcount, country in enumerate(countryList):
    for year in range(2012,2019):
    
#    res = np.load('resOpt_' + country + '_' + str(year) + '.npz')
        res = np.load('resOptUAI_' + country + '_' + str(year) + '.npz')
    
        xOptmat = res['xOptmat']
        x0mat = res['x0mat']
        emplOptmat = res['emplOptmat']
        empl0mat = res['empl0mat']
        parsOptmat = res['parsOptmat']
        co2Optmat = res['co2Optmat']
        co20mat = res['co20mat']
        lambdaList = res['lambdaList']
        Niter = res['Niter']
        
    #    dat = np.load('countTable_'+country+'_'+str(year)+'.npz')
        dat = np.load('countTableUAI_'+country+'_'+str(year)+'.npz')
    
        Aorg = dat['A']
        yorg = dat['y']
        S = dat['S']
        units = dat['fpUnits']
        
        impacts = dat['impacts']
        sectors = dat['sectors']
        
        
        maxRank = 8
        
        # largest employement intensity sectors
        empIdx = np.argsort(S[1,:])[-maxRank:]
        print([sectors[k] for k in empIdx])
        
        # largest carbon intensity sectors
        footIdx = np.argsort(S[3,:])[-maxRank:]
        print('carbon intensity')
        print([sectors[k] for k in footIdx])
        
        # largest carbon/emp intensity ratio
        footempratIdx = np.argsort(S[3,:]/S[1,:])[-maxRank:]
        print('carbon/employment ratio')
        print([sectors[k] for k in footempratIdx])
        
        
        # largest activity deacrease sectors
        lambdaIdx = -8
        optIdx = np.argsort(xOptmat.squeeze()[lambdaIdx,:]/(x0mat).squeeze())[:maxRank]
        print('optim: largest activity reduction (largest first)')
        print([sectors[k] for k in optIdx])
        
        optparIdx = np.argsort(parsOptmat.squeeze()[lambdaIdx,:])[:maxRank]
        print('....')
        print('optim: largest reduction (largest first)')
        print([sectors[k] for k in optparIdx])
        print(np.sort(parsOptmat.squeeze()[lambdaIdx,:])[:maxRank])
        print('....')
        
        print('co2 init/reduced')
        print((co20mat, str(100-100*co2Optmat.squeeze()[lambdaIdx]/co20mat)))
        print('....')
        
        ax[0,kcount].plot(np.sum(emplOptmat.squeeze(),axis=1),co2Optmat.squeeze(),label=str(year))
        empldiff = emplOptmat[-1,0,:]-emplOptmat[lambdaIdx,0,:]
        emplredsort = np.argsort(empldiff)
        if year == 2018:
            ax[0,kcount].plot([np.sum(emplOptmat.squeeze(),axis=1)[-1]*.95,
                               np.sum(emplOptmat.squeeze(),axis=1)[-1]*.95],[co2Optmat.squeeze()[0],co2Optmat.squeeze()[-1]],'--k',label=str(year)+' 5% reduction')
            ax[0,kcount].plot(np.sum(emplOptmat.squeeze(),axis=1)[lambdaIdx],
                              co2Optmat.squeeze()[lambdaIdx],'+k',label=str(year)+' choice')
        
        if kcount==0:
            ax[0,kcount].legend()#loc='center left', bbox_to_anchor=(1, 0.5))
        if kcount ==0: 
            ax[0,kcount].set_ylabel('GHG ('+units[24]+')')
        ax[0,kcount].set_xlabel('employmt. ('+units[9]+')')
        ax[0,kcount].set_title(country)
    
        ax[1,kcount].axis('off')
#        ax[1,kcount].table([[sectors[k][:min(60,len(sectors[k]))],
#                             np.round(100-100*2*sigmoid(parsOptmat.squeeze()[lambdaIdx,k]),2)] for k in optparIdx],loc='center',colWidths=[.65,.1])
        tab=ax[1,kcount].table([['Sector','Empl. red. (1000 p)']]+[[sectors[k][:min(39,len(sectors[k]))],
                            np.round(empldiff[k],2)] for k in emplredsort[-1:-7:-1]],loc='center',colWidths=[.65,.35])
        tab.scale(1.15,1.2)
#        tab.set_fontsize(34)
        
    #ax[1,0].set_title('Largest act. reduction (largest first)')



# plt.figure()

# plt.table([[sectors[k][:min(40,len(sectors[k]))],
#                     np.round(S[3,k]/1000,2)] for k in range(127,140)],loc='center',colWidths=[.6,.4])
# plt.gca().axis('off')