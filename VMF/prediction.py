# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:23:40 2020

@author: abdelhamid
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

def prediction(train_features,train_labels,mean_dir):
    
    pred_mat = cosine_similarity(train_features,mean_dir)
    
    y_pred = np.argmax(pred_mat,axis=1)
    conf   = np.max(pred_mat,axis=1)
    acc = accuracy_score(train_labels, y_pred)
    return y_pred,acc
        