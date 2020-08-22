# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:19:35 2019

@author: Fadoua Khmaissia
"""
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse, math, time, json, os
import os
from PIL import Image, ImageOps
from matplotlib.pyplot import imshow
from lib import arf, preprocess
from lib.preprocess import *
from config import config
from prep_cnn_data import prepare_data, gen_CNN_test_data
#from prep_yolo_data import prepare_data_YOLO, gen_yolo_test_data
import cv2 
#from train import train,plot_confusion_matrix,test

import create_model
from learning_function import learning_function
from config import Facnet_config
from select_classes import select_classes

###############################################################################################################################
##--------Generate list of testing and training files based on setting predeifned in config.py file--------------------------##
###############################################################################################################################
training_files = getTrFilesNames(config['shared']['root'] , config['shared']['testing_range'] , config['shared']['min_tr_range'], config['shared']['max_tr_range'], config['scenario'])
testing_files  = getTsFilesNames(config['shared']['root'] , config['shared']['testing_range'] , config['scenario'])


################################################################################################################################
## -------------- Define normalization parameters and Compute averaged normalization frames across all training ---------------#
################################################################################################################################
## Only execute line below to compute avge normalization frames for the first time
normalize_opts = config['normalize']
#normalize_opts['averaged_frames'], avge_frames_path = preprocess.compute_Norm_params(config['shared']['root'], training_files ,config['invert'], config['clip1'],config['exp_id'])
## To load already computed avereged frames uncomment the line below (change path accordingly)
avge_frames_path = './exp_res/Ts1500_Tr1000To1500/norm_avge_frames___Inv-0_Clip-0'

#######################################################################################################################
##----------------------------- To Load existing data------(CNN)-----------------------------------------------------##
#######################################################################################################################
_DATA_DIR_ = "./exp_res/"+config['exp_id']+'/CNN_data/'
l_train_set = np.load(os.path.join(_DATA_DIR_ ,  "l_train" +".npy"), allow_pickle=True).item()
u_train_set = np.load(os.path.join(_DATA_DIR_ ,  "u_train" +".npy"), allow_pickle=True).item()
test_set    = np.load(os.path.join(_DATA_DIR_ ,  "test" +".npy"), allow_pickle=True).item()

##########################################################################################
## ------------------ Regular CNN setting ----Prepare data, Train and Test--------------##
##########################################################################################

config['setting'] == 'Regular'
## Preprocess training and testing sets based on the settings defined in config.py file
#l_train_set, u_train_set, test_set = prepare_data(avge_frames_path,  training_files, testing_files)

images_train = l_train_set['images']
labels_train = l_train_set['labels']

images_test  = test_set['images']
labels_test  = test_set['labels']

images_train, labels_train = select_classes(images_train,labels_train)
images_test, labels_test   = select_classes(images_test,labels_test)

##########################################################################################
## ------------------------------------- Creating the model-----------------------------##
##########################################################################################
learning_rate       = Facnet_config['learning_rate']
width               = Facnet_config['width']

model  = create_model.WRN2(width)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device=device, dtype=torch.float)
model.eval()

##########################################################################################
## ---------------------------------- Creating the optimizer----------------------------##
##########################################################################################
optimizer     = torch.optim.Adam(model.parameters(),lr = learning_rate)

learning_function(model,optimizer,device,images_train, labels_train,images_test,labels_test)