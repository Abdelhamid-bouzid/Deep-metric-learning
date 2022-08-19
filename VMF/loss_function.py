# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:06:38 2020

@author: abdelhamid
"""
import torch
from torch.autograd import Function

class loss_function(Function):
    def __init__(self, Con):
        self.Con  = Con
        
    def forward(self, output, labels, mean_dir,device):
        
        mat1 = torch.exp(torch.mm(output,mean_dir)*self.Con)
        mat2 = torch.gather(mat1, 1, labels.unsqueeze(-1)).squeeze()
        
        mat3 = -torch.log(mat2/torch.sum(mat1,1))
        
        loss = torch.sum(mat3)
        
        return loss
    
