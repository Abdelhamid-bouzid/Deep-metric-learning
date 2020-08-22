"""
Created on Sun Nov 17 15:19:44 2019

@author: Abdelhamid Bouzid
"""

import numpy as np
import os

Facnet_config = {
    "step_size" : 9,
    "Epochs" : 1000, 
    "learning_rate" : 10**-4,
    "Con" : 15,
    "alpha" : 0,
    "pre_tr" : 2, 
    "nb_clusters" : 8,
    "list_classes" : [0,1,2,3,4,5,6,7,8,9],
    "batch_size" : 32,
    "optimizer_flag" : 'Adam',
    "width" : 2,
}
