# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:06:38 2020

@author: abdelhamid
"""
import torch
from torch.autograd import Function
from config import Facnet_config

class loss_function(Function):
    def __init__(self, alpha):
        self.alpha  = alpha
        self.M      = Facnet_config['M']
        self.D      = Facnet_config['D']
        self.pdist  = torch.nn.PairwiseDistance(p=2)
        
        self.epsilon      = Facnet_config['epsilon']
        
    def mean_sigma(self, output, batch_label,chosen_clusters,device):
        
        sigma         = torch.zeros(chosen_clusters.shape[0]).to(device)
        mean_clusters = torch.zeros((chosen_clusters.shape[0],output.shape[1])).to(device)
        for i in range(chosen_clusters.shape[0]):
            inds               = torch.where((batch_label[:,0]==chosen_clusters[i,0]) & (batch_label[:,1]==chosen_clusters[i,1]) )[0]
            mean_clusters[i,:] = output[inds].mean(0)
            sigma[i]           = self.pdist(output[inds].mean(0),output[inds]).pow(2).mean(0)  
 
        sigma = sigma.mean(0)
        return sigma, mean_clusters    
    
    def forward(self, output, batch_label,chosen_clusters,device):
        
        sigma, mean_clusters = self.mean_sigma(output, batch_label,chosen_clusters,device)
        loss1 = torch.zeros(output.shape[0]).to(device)
        for s in range(output.shape[0]):
            dis        = -(1/(2*sigma.pow(2)))*self.pdist(output[s],mean_clusters).pow(2)
            inds1      = torch.where((chosen_clusters[:,0]==batch_label[s,0]) & (chosen_clusters[:,1]==batch_label[s,1]) )[0]
            inds2      = torch.where(chosen_clusters[:,0]!=batch_label[s,0])[0]
            
            num        = torch.exp(dis[inds1]-self.alpha)
            den        = torch.exp(dis[inds2]).sum(0)
            loss1[s]   = -torch.log(num/(den+self.epsilon) + self.epsilon)
                
        loss2 = torch.clamp(loss1, min = 0.0)
        loss  = loss2.mean(0)
        
        return loss
    