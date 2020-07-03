# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:09:18 2020

@author: abdelhamid
"""
import numpy as np
from config import Facnet_config
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def mean_dicrections(train_features,train_labels):
    
    list_classes    = Facnet_config['list_classes']
    mean_dir = np.zeros((len(list_classes),train_features.shape[1]))
    dis_inf  = np.zeros((len(list_classes),train_features.shape[0]))
    R_estimations = np.zeros((len(list_classes),))
    for c in list_classes:
        inds_c               = np.where(train_labels==c)[0]
        class_features       = train_features[inds_c]
        c_mean_direction     = np.sum(class_features,axis=0)
        mean_dir[c,:]        = c_mean_direction/np.linalg.norm(c_mean_direction)
        
        #dis_inf[c,:]     = cosine_similarity(mean_dir[c,:].reshape(1, -1),train_features)
        dis_inf[c,:]     = euclidean_distances(mean_dir[c,:].reshape(1, -1),train_features)
        R_estimations[c] = np.linalg.norm(c_mean_direction)/inds_c.shape[0]
    return mean_dir,dis_inf,R_estimations