# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:09:18 2020

@author: abdelhamid
"""
import numpy as np
from config import Facnet_config
from sklearn.cluster import KMeans

def cluster_training(train_features,train_labels):
    
    nb_clusters     = Facnet_config['nb_clusters']
    list_classes    = Facnet_config['list_classes']
    
    clusters_labels  = []
    clusters_centers = []
    
    m = 0
    samples_labels = np.zeros(train_labels.shape[0])
    for c in list_classes:
        inds_c               = np.where(train_labels==c)[0]
        class_features       = train_features[inds_c]
        
        kM = KMeans(n_clusters=nb_clusters, random_state=0).fit(class_features)
        for i in range(nb_clusters):
            clusters_centers.append(kM.cluster_centers_[i]/np.linalg.norm(kM.cluster_centers_[i]))
            clusters_labels.append(np.array([c,i]))
            
            inds2 = inds_c[np.where(kM.labels_==i)[0]]
            for j in range(inds2.shape[0]):
                samples_labels[inds2[j]] = m
            m = m+1
            
    clusters_centers = np.stack(clusters_centers,axis=0)   
    clusters_labels  = np.stack(clusters_labels,axis=0)
    return clusters_centers,clusters_labels,samples_labels