from forward_all_images import forward_all_images
from loss_function import loss_function
from prediction import prediction

import torch
import numpy as np
from config import Facnet_config
from cluster_training import cluster_training

def learning_function(model,optimizer,device,images_train, labels_train,images_test,labels_test):
    
    
    ''' #################################################  initialization  ################################################### '''
    Epochs              = Facnet_config['Epochs']
    batch_size          = Facnet_config['batch_size']
    Con                 = Facnet_config['Con']
    alpha               = Facnet_config['alpha']
        
    train_acc, test_acc, loss = [], [], []
    
    ''' #################################################  test the initial model  ################################################### '''
    train_features = forward_all_images(model,device,images_train)
    test_features  = forward_all_images(model,device,images_test)
    
    clusters_centers,clusters_labels,samples_labels = cluster_training(train_features,labels_train)
    
    _,train_acc_1  = prediction(train_features,labels_train,clusters_centers,clusters_labels)
    _,test_acc_1   = prediction(test_features,labels_test,clusters_centers,clusters_labels) 
    
    train_acc.append(train_acc_1)
    test_acc.append(test_acc_1)
    
    torch.save(model.state_dict(), r'models\model'+str(0)+'.pth')
    print("   #######################  Train Epoch: {} train_acc: {:0.4f}  test_acc: {:0.4f} ###################       ".format(0,train_acc[-1],test_acc[-1]))
          
    ''' #################################################### block for learning  ########################################################## '''  
    #torch.autograd.set_detect_anomaly(True)    
    for i in range(1,Epochs+1):
        
        '''##################################################### Learning ################################################################################'''
        arr = np.arange(images_train.shape[0])
        np.random.shuffle(arr)
        
        images_train_1   = images_train[arr]
        samples_labels_1 = samples_labels[arr]
        
        list_inds  = [s for s in range(0,images_train.shape[0],batch_size)]
        
        for s in list_inds:
            if s+batch_size<images_train.shape[0]:
                targets = images_train_1[s:s+batch_size]
                labels  = samples_labels_1[s:s+batch_size]
            else:
                targets = images_train_1[s:]
                labels  = samples_labels_1[s:]
            
            targets            = torch.from_numpy(targets).to(device=device, dtype=torch.float)
            labels             = torch.from_numpy(labels).to(device=device, dtype=torch.float)
            clusters_centers_1 = torch.from_numpy(clusters_centers).to(device=device, dtype=torch.float)
            clusters_labels_1  = torch.from_numpy(clusters_labels).to(device=device, dtype=torch.float)
            
            model.train()
            output  = model(targets)
            loss    = loss_function(Con,alpha).forward(output, labels, torch.t(clusters_centers_1),clusters_labels_1,device).to(device)
            
            model.eval()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()

        
        '''######################################################## Forwad the data  ########################################################################'''
        train_features = forward_all_images(model,device,images_train)
        test_features  = forward_all_images(model,device,images_test)
        
        '''########################################################## Clustering  ##########################################################################'''
        clusters_centers,clusters_labels,samples_labels = cluster_training(train_features,labels_train)
        
        '''##################################################### testing with KNN ##########################################################################'''  
        _,train_acc_1  = prediction(train_features,labels_train,clusters_centers,clusters_labels)
        _,test_acc_1   = prediction(test_features,labels_test,clusters_centers,clusters_labels)      
       
        '''###################################################################################################################################################'''
        train_acc.append(train_acc_1)
        test_acc.append(test_acc_1)
        
        torch.save(model.state_dict(), r'models\model'+str(i)+'.pth')
        print("   #######################  Train Epoch: {} train_acc: {:0.4f}  test_acc: {:0.4f} ###################       ".format(i,train_acc[-1],test_acc[-1]))
        
        if i==-1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 10**-5
        
    
    train_acc = np.stack(train_acc)
    test_acc  = np.stack(test_acc)
    
    np.save(r'data1\train_acc',train_acc)
    np.save(r'data1\test_acc',test_acc)