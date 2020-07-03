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
    #K                   = Facnet_config['K']
    K                   = 1
    
    D  = euclidean_distances(test_features,train_features)
    D  = np.square(D)
    I  = np.argsort(D,axis=1)
    y_pred     = KNN_conf(I[:,1:K+1], labels_train)
    label_pred = np.argmax(y_pred,axis=1)
    acc = accuracy_score(labels_test, label_pred)
    
    return y_pred,acc
def KNN_conf(mat, labels_train):
    
    conf_vector1 = np.zeros((mat.shape[0],len(Facnet_config['list_classes'])))
    for i in range(mat.shape[0]):
        for j in Facnet_config['list_classes']:
            conf_vector1[i,j] = np.where(labels_train[mat[i,:]]==j)[0].shape[0]/mat.shape[1]
                
    return conf_vector1