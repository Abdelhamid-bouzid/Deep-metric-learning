from mean_dicrections import mean_dicrections
from forward_all_images import forward_all_images
from prediction import prediction
from loss_function import loss_function
from test_model import test_model

import torch
import numpy as np
import scipy.io as sio
from config import Facnet_config

def learning_function(model,optimizer,device,images_train, labels_train,images_test,labels_test):
    
    
    ''' #################################################  initialization  ################################################### '''
    Epochs              = Facnet_config['Epochs']
    batch_size          = Facnet_config['batch_size']
    Con                 = Facnet_config['Con']
        
    train_acc, test_acc, Loss_fn = [], [], []
    
    ''' #################################################  test the initial model  ################################################### '''
    train_features = forward_all_images(model,device,images_train)
    test_features  = forward_all_images(model,device,images_test)
    
    mean_dir,_,_       = mean_dicrections(train_features,labels_train)
    
    _,train_acc_1  = prediction(train_features,labels_train,mean_dir)
    _,test_acc_1   = prediction(test_features,labels_test,mean_dir)
    
# =============================================================================
#     _,train_acc_2  = test_model(train_features, labels_train,train_features,labels_train)
#     _,test_acc_2   = test_model(train_features, labels_train,test_features,labels_test) 
# =============================================================================
    
    train_acc.append(train_acc_1)
    test_acc.append(test_acc_1)
    
    torch.save(model.state_dict(), r'models\model'+str(0)+'.pth')
    #print("   #######################  Train Epoch: {} train_acc: {:0.4f} val_acc: {:0.4f} test_acc: {:0.4f} ###################       ".format(0,train_acc[-1],val_acc[-1],test_acc[-1]))
    print("######################################################################################################################")
    print("   #####  Cosine: Train Epoch: {} train_acc: {:0.4f} test_acc: {:0.4f} #####".format(0,train_acc_1,test_acc_1))
    #print("   #####  KNN: Train Epoch: {} train_acc: {:0.4f} val_acc: {:0.4f} test_acc: {:0.4f}  #####".format(0,train_acc_2,val_acc_2,test_acc_2))
    print("######################################################################################################################")
          
          
    ''' #################################################### block for learning  ########################################################## '''      
    for i in range(1,Epochs+1):
        
        '''##################################################### Learning ################################################################################'''
        arr = np.arange(images_train.shape[0])
        np.random.shuffle(arr)
        
        images_train_1 = images_train[arr]
        labels_train_1 = labels_train[arr]
        
        list_inds  = [s for s in range(0,images_train.shape[0],batch_size)]
        
        m = 0
        ll = 0
        for s in list_inds:
            if s+batch_size<images_train.shape[0]:
                targets = images_train_1[s:s+batch_size]
                labels  = labels_train_1[s:s+batch_size]
            else:
                targets = images_train_1[s:]
                labels  = labels_train_1[s:]
            targets    = torch.from_numpy(targets).to(device=device, dtype=torch.float)
            labels     = torch.from_numpy(labels).to(device=device, dtype=torch.float)
            mean_dir1  = torch.from_numpy(mean_dir).to(device=device, dtype=torch.float)
            
            model.train()
            output  = model(targets)
            loss   = loss_function(Con).forward(output, labels, torch.t(mean_dir1),device).to(device)
            
            if loss.item()==0:
                model.eval()
                break;
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            #print("   ######### iteration percentage: {:0.4f} loss: {:0.8f}  ".format((m/len(list_inds))*100,loss.item()))
            ll = ll + loss.item()
            m = m+1
        
        '''######################################################## Forwad the data  ########################################################################'''
        train_features = forward_all_images(model,device,images_train)
        test_features  = forward_all_images(model,device,images_test)
        
        '''##################################################### Update mean directions ###################################################################'''
        mean_dir,_,_       = mean_dicrections(train_features,labels_train)
        
        '''##################################################### testing with cosine ########################################################################'''
        _,train_acc_1  = prediction(train_features,labels_train,mean_dir)
        _,test_acc_1   = prediction(test_features,labels_test,mean_dir)
        
        '''##################################################### testing with KNN ##########################################################################'''       
# =============================================================================
#         _,train_acc_2  = test_model(train_features, labels_train,train_features,labels_train)
#         _,test_acc_2   = test_model(train_features, labels_train,test_features,labels_test)       
# =============================================================================
       
        '''###################################################################################################################################################'''
        train_acc.append(train_acc_1)
        test_acc.append(test_acc_1)
        Loss_fn.append(ll)
        
        torch.save(model.state_dict(), r'models\model'+str(i)+'.pth')
        #print("   #######################  Train Epoch: {} train_acc: {:0.4f} val_acc: {:0.4f} test_acc: {:0.4f} ###################       ".format(i,train_acc[-1],val_acc[-1],test_acc[-1]))
        print("######################################################################################################################")
        print("   #####  Cosine: Train Epoch: {} train_acc: {:0.4f} test_acc: {:0.4f} #####".format(i,train_acc_1,test_acc_1))
        #print("   #####  KNN: Train Epoch: {} train_acc: {:0.4f} val_acc: {:0.4f} test_acc: {:0.4f}  #####".format(i,train_acc_2,val_acc_2,test_acc_2))
        print("######################################################################################################################")
              
        
    
    train_acc = np.stack(train_acc)
    test_acc  = np.stack(test_acc)
    Loss_fn   = np.stack(Loss_fn)
    
    np.save(r'data1\train_acc',train_acc)
    np.save(r'data1\test_acc',test_acc)
    np.save(r'data1\Loss_fn',Loss_fn)