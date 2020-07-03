# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:01:24 2020

@author: abdelhamid
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from config import Facnet_config
from sklearn.metrics import accuracy_score

def KNC(clusters_centers, clusters_labels,test_features,labels_test,sigma):
    
    L = Facnet_config['L']
    D  = euclidean_distances(test_features,clusters_centers)
    D  = np.square(D)
    I  = np.argsort(D,axis=1)
    D1 = np.sort(D,axis=1)
    
    D2  = -(1/(2*sigma ))*D1
    D2  = np.exp(D1)
    
    label_pred = np.zeros(I.shape[0])
    for i in range(I.shape[0]):
        cl = clusters_labels[I[i,:L]]
        X  = np.zeros(len(Facnet_config['list_classes']))
        for j in Facnet_config['list_classes']:
            inds = np.where(cl==j)[0]
            X[j] = np.sum(D2[i,inds])
            
        label_pred[i] = np.argmax(X/X.sum())
        
    acc = accuracy_score(labels_test, label_pred)
    return labels_test,acc
        