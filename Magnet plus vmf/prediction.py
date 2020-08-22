# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:23:40 2020

@author: abdelhamid
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

def prediction(train_features,train_labels,clusters_centers,clusters_labels):
    
    y_pred = np.zeros(train_features.shape[0])
    for i in range(train_features.shape[0]):
        feature = train_features[i]
        similarity = np.zeros(clusters_centers.shape[0])
        for j in range(clusters_centers.shape[0]):
            mean_direction = clusters_centers[j]
            similarity[j] = cosine_similarity(feature.reshape(1, -1),mean_direction.reshape(1, -1))
            
        y_pred[i] = clusters_labels[np.argmax(similarity),0]
        
    acc = accuracy_score(train_labels, y_pred)
    return y_pred,acc
            
        