# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:09:18 2020

@author: abdelhamid
"""
import numpy as np
from config import config

def mean_dicrections(train_features,train_labels):
    
    list_classes    = config['list_classes']
    mean_dir = np.zeros((len(list_classes),train_features.shape[1]))
    for c in list_classes:
        inds_c               = np.where(train_labels==c)[0]
        class_features       = train_features[inds_c]
        c_mean_direction     = np.sum(class_features,axis=0)
        mean_dir[c,:]        = c_mean_direction/np.linalg.norm(c_mean_direction)
    return mean_dir