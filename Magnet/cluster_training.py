# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:09:18 2020

@author: abdelhamid
"""
import numpy as np
from config import Facnet_config
from sklearn.cluster import KMeans

def cluster_training(train_features,train_labels):
    
    nb_clusters_list = Facnet_config['nb_clusters']
    list_classes     = Facnet_config['list_classes']
    
    clusters_labels  = []
    samples_indices  = []
    clusters_centers = []
    sigma = 0
    for c in list_classes:
        nb_clusters          = nb_clusters_list[c]
        inds_c               = np.where(train_labels==c)[0]
        class_features       = train_features[inds_c]
        
        kM = KMeans(n_clusters=nb_clusters, random_state=0).fit(class_features)
        sigma = sigma + kM.inertia_
        for i in range(nb_clusters):
            clusters_centers.append(kM.cluster_centers_[i])
            clusters_labels.append(np.array([c,i]))
            samples_indices.append(inds_c[np.where(kM.labels_==i)[0]])
        
    clusters_centers = np.stack(clusters_centers,axis=0)   
    clusters_labels  = np.stack(clusters_labels,axis=0)
    sigma = sigma/(nb_clusters*len(list_classes))
    return clusters_centers,clusters_labels,samples_indices,sigma