# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:19:35 2019

@author: Abdelhamid bouzid
"""
import torch

import create_model
from learning_function import learning_function
from config import Facnet_config


##########################################################################################
## ------------------ Import train/test data --------------##
##########################################################################################

#images_train, labels_train =
#images_test, labels_test   = 

##########################################################################################
## ------------------------------------- Creating the model-----------------------------##
##########################################################################################
learning_rate       = Facnet_config['learning_rate']
width               = Facnet_config['width']

model     = create_model.WRN2(width)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device=device, dtype=torch.float)
model.eval()

##########################################################################################
## ---------------------------------- Creating the optimizer----------------------------##
##########################################################################################
optimizer     = torch.optim.Adam(model.parameters(),lr = learning_rate)

learning_function(model,optimizer,device,images_train, labels_train,images_test,labels_test)
