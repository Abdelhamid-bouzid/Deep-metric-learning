# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:22:36 2019

@author: abdelhamid
"""
import numpy as np
from config import Facnet_config

def select_classes(images,labels):
    list_classes  = Facnet_config['list_classes']
    out_images = []
    out_labels = []
    for l in list_classes:
        inds = np.where(labels==l)[0]
        out_images.append(images[inds])
        out_labels.append(labels[inds])
        
    out_images = np.concatenate(out_images,axis=0)
    out_labels = np.concatenate(out_labels,axis=0)
    return out_images, out_labels