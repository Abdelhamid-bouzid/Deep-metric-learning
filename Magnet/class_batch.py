# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:16:10 2020

@author: abdelhamid
"""
import numpy as np
from config import Facnet_config
from sklearn.metrics.pairwise import euclidean_distances

class class_batch():
    def __init__(self, clusters_centers,clusters_labels,samples_indices):
        
        self.M                 = Facnet_config['M']
        self.D                 = Facnet_config['D']
        
        self.clusters_centers  = clusters_centers
        self.clusters_labels   = clusters_labels
        self.samples_indices   = samples_indices
        self.P_dis             = euclidean_distances(clusters_centers,clusters_centers)
        self.S_dis             = np.argsort(self.P_dis,axis=1)[:,1:]
        
    def construct_batch(self,inds1):
        batch_inds  = []
        batch_label = []
        
        if self.samples_indices[inds1].shape[0]<self.D:
            batch_inds.append(self.samples_indices[inds1])
            batch_label.append(np.repeat(np.expand_dims(self.clusters_labels[inds1],axis=0),self.samples_indices[inds1].shape[0],axis=0))
        else:
            batch_inds.append(self.samples_indices[inds1][np.random.randint(low=0, high=self.samples_indices[inds1].shape[0], size=self.D)])   
            batch_label.append(np.repeat(np.expand_dims(self.clusters_labels[inds1],axis=0),self.D,axis=0))
        
        clusters_inds = self.choose_nearest(inds1)
        for i in range(clusters_inds.shape[0]):
            inds2 = clusters_inds[i]
            if self.samples_indices[inds2].shape[0]<self.D:
                batch_inds.append(self.samples_indices[inds2])
                batch_label.append(np.repeat(np.expand_dims(self.clusters_labels[inds2],axis=0),self.samples_indices[inds2].shape[0],axis=0))
            else:
                batch_inds.append(self.samples_indices[inds2][np.random.randint(low=0, high=self.samples_indices[inds2].shape[0], size=self.D)])
                batch_label.append(np.repeat(np.expand_dims(self.clusters_labels[inds2],axis=0),self.D,axis=0))
                
        batch_inds      = np.concatenate(batch_inds,axis=0)        
        batch_label     = np.concatenate(batch_label,axis=0) 
        chosen_clusters = np.unique(batch_label, axis=0)
        return batch_inds,batch_label,chosen_clusters
    
    def choose_nearest(self,inds1):
        
        pos = int((2/3)*self.M)
        neg = int((1/3)*self.M)
        
        labels  = self.clusters_labels[self.S_dis[inds1,:],0]
        test    = np.where(labels[:self.M]!=self.clusters_labels[inds1,0])[0].shape[0]
        
        if test>neg:
            clusters_inds = self.S_dis[inds1,:self.M]
        else:
            inds_eq   = np.where(labels==self.clusters_labels[inds1,0])[0]
            inds_ineq = np.where(labels!=self.clusters_labels[inds1,0])[0]
            
            clusters_inds = np.concatenate([inds_eq[:pos],inds_ineq[:neg]],axis=0)
        
        return clusters_inds
    
        