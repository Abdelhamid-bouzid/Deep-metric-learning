"""
Created on Sun Nov 17 15:19:44 2019

@author: Fadoua Khmaissia
"""

import numpy as np
import os
data_config = {1000 : ['cegr02003*.arf', 'cegr01923*.arf'],
             1500 : ['cegr02005*.arf', 'cegr01925*.arf'],
             2000 : ['cegr02007*.arf', 'cegr01927*.arf'],
             2500 : ['cegr02009*.arf', 'cegr01929*.arf'],
             3000 : ['cegr02011*.arf', 'cegr01931*.arf'],
             3500 : ['cegr02013*.arf', 'cegr01933*.arf'],
             4000 : ['cegr02015*.arf', 'cegr01935*.arf'],
             4500 : ['cegr02017*.arf', 'cegr01937*.arf'],
             5000 : ['cegr02019*.arf', 'cegr01939*.arf']
}

VehicleNameDict={'PICKUP':0,'SUV':1,'BTR70':2,'BRDM2':3,'BMP2':4,
                    'T72':5,'ZSU23':6,'2S3':7,'MTLB':8,'D20':9}

shared_config = {
    "root" : r'C:\Users\abdelhamid\Desktop\ATR Database\cegr\arf', 
    "labels" : r'.\inputs\original_labels\labels',
    "zone" : r'C:\Users\abdelhamid\Desktop\ATR Database\cegr\agt',
    "yolo_labels" : r'.\inputs\yolo_labels',
    "testing_range" : 1500,
    "max_tr_range" : 2000,
    "min_tr_range" : 1000,
    "val_pct" : 0.1, # validation percentage
    "ul_pct" : 0.2, # shifting param for unlabeled data
    "ul_shift" : 0.2, # percentage per target for added unlabeled data
    "rnd_seed" : 1, #random seed for shuffeling the data

}

### pre-processing algorithms ###
invert_config = {
    "select" : 0,
}

clip1_config = {
    "select" : 0, 
    "lower_thresh" : 0.5,
    "upper_thresh" : 99.5,
}


normalize_config = {
    "select" : 1,
    "technique" : 3,
    #opts=1 ==> mean, std per frame 
    #opts=2 ==> mean, std per pixel (using mean and std images computed across all training set) 
    #opts=3 ==> med, mad normalization per frame 
    #opts=4 ==> med, mad normalization per pixe (using med and mad frames computed across all training set)
    "averaged_frames" : dict(),
}

clip2_config = {
    "select" : 1, 
    "lower_lim" : -5,
    "upper_lim" : 10,
}

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


### master ###
config = {
    "setting" : 'Regular', # 'Regular' or 'YOLO'
    "shared" : shared_config,
    "invert" : invert_config ,
    "clip1" : {"select" : clip1_config['select'], "threshold" : np.array([clip1_config['lower_thresh'], clip1_config['upper_thresh']])},
    "normalize" : normalize_config,
    "clip2" : {"select" : clip2_config['select'], "lims" : np.array([clip2_config['lower_lim'], clip2_config['upper_lim']])},
    "scenario" : data_config,
    "exp_id" : 'Ts'+str(shared_config['testing_range'])+'_Tr'+str(shared_config['min_tr_range'])+'To'+str(shared_config['max_tr_range']),
    "facenet" : Facnet_config,
    "cls_id" : VehicleNameDict,
}

if not os.path.exists('./exp_res/'+config['exp_id']):
    os.mkdir('./exp_res/'+config['exp_id'])