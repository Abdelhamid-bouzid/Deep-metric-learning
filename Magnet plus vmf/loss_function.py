# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:06:38 2020

@author: abdelhamid
"""
import torch
from torch.autograd import Function
    
class loss_function(Function):
    def __init__(self, Con,alpha):
        self.Con    = Con
        self.alpha  = alpha
        
    def forward(self, output, labels, clusters_centers,clusters_labels,device):
        
        mat1 = torch.exp(torch.mm(output,clusters_centers)*self.Con)
        mat2 = torch.zeros(mat1.shape[0]).to(device)
        for i in range(mat1.shape[0]):
            inds = torch.where(clusters_labels[:,0]!=clusters_labels[labels.type(torch.long)[i],0])[0]
            
            #num  = mat1[i,labels.type(torch.long)[i]]*torch.exp(torch.Tensor([self.alpha])).to(device)
            num  = mat1[i,labels.type(torch.long)[i]]
            den  = mat1[i,inds].sum(0)+num
            mat2 = -torch.log(num/den)
        
        #mat3 = torch.clamp(mat2, min = 0.0)
        loss = torch.sum(mat2)
        
        return loss    