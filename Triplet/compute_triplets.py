# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:45:11 2019

@author: abdelhamid
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from forward_all_images import forward_all_images
from config import Facnet_config

def compute_triplets(model,device,images_train, labels_train):
    K                   = Facnet_config['triplet_K']
    
    train_features = forward_all_images(model,device,images_train)
    D  = euclidean_distances(train_features,train_features)
    D  = np.square(D)
    I  = np.argsort(D,axis=1)
    L  = label_mat(I[:,:K+1], labels_train)
    
    anch_inds = []
    pos_inds  = []
    neg_inds  = []
    
    triplet_label = []
    
    for i in range(train_features.shape[0]):
        inds1 = np.where(L[i,1:]==L[i,0])[0]
        inds2 = np.where(L[i,1:]!=L[i,0])[0]
        
        if inds2.shape[0]!=0:
            for j in range(inds1.shape[0]):
                aa = np.where(inds2<inds1[j])[0]
                if aa.shape[0]!=0:
                    #for k in range(1):
                    for k in range(aa.shape[0]):
                        anch_inds.append(i)
                        pos_inds.append(I[i,inds1[j]+1])
                        neg_inds.append(I[i,inds2[aa[k]]+1])
                        
                        triplet_label.append(labels_train[i])
    if len(anch_inds)>0:                    
        anch_inds = np.stack(anch_inds,axis=0)
        pos_inds  = np.stack(pos_inds,axis=0)
        neg_inds  = np.stack(neg_inds,axis=0)
        triplet_label  = np.stack(triplet_label,axis=0)
    
    return triplet_label,anch_inds,pos_inds,neg_inds

def label_mat(mat, labels):
    
    L = np.zeros((mat.shape))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            L[i,j] = labels[mat[i,j]]
        
    return L