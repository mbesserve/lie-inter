#!/usr/bin/env python
# coding: utf-8

# In[148]:


import pymrio
import copy
import numpy as np
for year in range(1995,2012):

    dataRoot = '/is/projects/netfmri/mrio/'
    mrioDat = pymrio.load(dataRoot+'exiobase_v38/pkl/'+str(year)+'/')
    
    mrioImp = pymrio.load(dataRoot+'exiobase_v38/pkl/'+str(year)+'/satellite')
    
    sectors = mrioDat.get_sectors()
    impacts = mrioImp.get_rows()
    sectors=list(sectors)
    
    
    impacts = list(impacts)
    
    
    fpUnits = list(mrioImp.unit['unit'])
    
    
    
    countryList = ['FR','DE','IT','US','GB']
    
    for kcount, country in enumerate(countryList):
        S = mrioImp.S[country].to_numpy()
        
        S.shape
        
        
        xred=mrioDat.x.xs(country,level='region')
        
        Ared = copy.deepcopy(mrioDat.A.xs(country,level='region'))
        
        Act = copy.deepcopy(Ared[:][country])
        
        A = Act.to_numpy()
        
        # check A is in the right transposition
        np.round(A[1:10,1:10],3),sectors[1:10]
        
        #print([(k,"".join([words[:min(len(words),6)]+'. '   for words in sec.split()])) for k,sec in enumerate(sectors)])
        
        yred=mrioDat.Y.xs(country,level='region')
        
        yred=yred.xs(country,level='region',axis=1)
        
        finCateg = yred.columns.get_level_values('category')
        
        finCateg = list(finCateg)
        
        y= yred.to_numpy()
        
        y.shape
        
        
        
        np.savez('countTableUAI_'+country+'_'+str(year),A=A,y=y,S=S,sectors=sectors,impacts=impacts,finCateg=finCateg,fpUnits=fpUnits)
    


