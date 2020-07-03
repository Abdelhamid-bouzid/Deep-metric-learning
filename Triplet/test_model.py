# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:21:58 2019

@author: abdelhamid
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
from config import Facnet_config

def test_model(train_features, labels_train,test_features,labels_test):
    K                   = Facnet_config['K']
    
    D  = euclidean_distances(test_features,train_features)
    D  = np.square(D)
    I  = np.argsort(D,axis=1)
    Conf1     = KNN_conf(I[:,1:K+1], labels_train)
    Conf      = np.max(Conf1,axis=1)
    label_pred = np.argmax(Conf1,axis=1)
    acc = accuracy_score(labels_test, label_pred)
    
    return label_pred,acc
def KNN_conf(mat, labels_train):
    
    conf_vector1 = np.zeros((mat.shape[0],len(Facnet_config['list_classes'])))
    for i in range(mat.shape[0]):
        for c,j in enumerate(Facnet_config['list_classes']):
            conf_vector1[i,c] = np.where(labels_train[mat[i,:]]==j)[0].shape[0]/mat.shape[1]
                
    return conf_vector1