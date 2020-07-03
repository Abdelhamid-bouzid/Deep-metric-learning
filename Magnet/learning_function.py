from forward_all_images import forward_all_images
from loss_function import loss_function
from test_model import test_model

import torch
import numpy as np
from config import Facnet_config
from cluster_training import cluster_training
from class_batch import class_batch

def learning_function(model,optimizer,device,images_train, labels_train,images_test,labels_test):
    
    
    ''' #################################################  initialization  ################################################### '''
    Epochs              = Facnet_config['Epochs']
    nb_batches          = Facnet_config['nb_batches']
    alpha               = Facnet_config['alpha']
        
    train_acc, test_acc, loss = [], [], []
    
    ''' #################################################  test the initial model  ################################################### '''
    train_features = forward_all_images(model,device,images_train)
    test_features  = forward_all_images(model,device,images_test)
    
    clusters_centers,clusters_labels,samples_indices,sigma = cluster_training(train_features,labels_train)
    CB      = class_batch(clusters_centers,clusters_labels,samples_indices)
    
    _,train_acc_1  = test_model(train_features, labels_train,train_features,labels_train)
    _,test_acc_1   = test_model(train_features, labels_train,test_features,labels_test) 
    
    train_acc.append(train_acc_1)
    test_acc.append(test_acc_1)
    
    torch.save(model.state_dict(), r'models\model'+str(0)+'.pth')
    print("   #######################  Train Epoch: {} train_acc: {:0.4f}  test_acc: {:0.4f} ###################       ".format(0,train_acc[-1],test_acc[-1]))
          
    ''' #################################################### block for learning  ########################################################## '''  
    #torch.autograd.set_detect_anomaly(True)    
    for i in range(1,Epochs+1):
        
        '''##################################################### Learning ################################################################################'''
        
        
        #indices = np.random.randint(low=0, high=clusters_centers.shape[0], size=nb_batches)
        indices = np.arange(clusters_centers.shape[0])
        
        for s in range(indices.shape[0]):
            batch_inds,batch_label,chosen_clusters = CB.construct_batch(indices[s])
            targets                = images_train[batch_inds]
            
            targets         = torch.from_numpy(targets).to(device=device, dtype=torch.float)
            batch_label     = torch.from_numpy(batch_label).to(device=device, dtype=torch.float)
            chosen_clusters = torch.from_numpy(chosen_clusters).to(device=device, dtype=torch.float)
            
            model.train()
            output  = model(targets)
            loss   = loss_function(alpha).forward(output, batch_label,chosen_clusters,device).to(device)
            
            model.eval()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()

        
        '''######################################################## Forwad the data  ########################################################################'''
        train_features = forward_all_images(model,device,images_train)
        test_features  = forward_all_images(model,device,images_test)
        
        '''########################################################## Clustering  ##########################################################################'''
        clusters_centers,clusters_labels,samples_indices,sigma = cluster_training(train_features,labels_train)
        CB      = class_batch(clusters_centers,clusters_labels,samples_indices)
        
        '''##################################################### testing with KNN ##########################################################################'''       
        _,train_acc_1  = test_model(train_features, labels_train,train_features,labels_train)
        _,test_acc_1   = test_model(train_features, labels_train,test_features,labels_test)      
       
        '''###################################################################################################################################################'''
        train_acc.append(train_acc_1)
        test_acc.append(test_acc_1)
        
        torch.save(model.state_dict(), r'models\model'+str(i)+'.pth')
        print("   #######################  Train Epoch: {} train_acc: {:0.4f}  test_acc: {:0.4f} ###################       ".format(i,train_acc[-1],test_acc[-1]))
              
        
    
    train_acc = np.stack(train_acc)
    test_acc  = np.stack(test_acc)
    
    np.save(r'data1\train_acc',train_acc)
    np.save(r'data1\test_acc',test_acc)