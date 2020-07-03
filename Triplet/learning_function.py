from cross_val_function import cross_val_function
from compute_triplets import compute_triplets
from test_model import test_model
import torch
import numpy as np
from config import Facnet_config
from forward_all_images import forward_all_images

def learning_function(model,optimizer,device,images_train, labels_train,images_test,labels_test):
    
    
    ''' #################################################  initialization  ################################################### '''
    Epochs              = Facnet_config['Epochs']
        
    train_acc, test_acc, loss = [], [], []
    
    ''' #################################################  test the initial model  ################################################### '''
    train_features = forward_all_images(model,device,images_train)
    test_features  = forward_all_images(model,device,images_test)
    
    _,train_acc_1  = test_model(train_features, labels_train,train_features,labels_train)
    _,test_acc_1   = test_model(train_features, labels_train,test_features,labels_test)
    
    train_acc.append(train_acc_1)
    test_acc.append(test_acc_1)
    
    torch.save(model.state_dict(), r'models\model'+str(0)+'.pth')
    print("   #######################  Train Epoch: {} train_acc: {:0.4f}  test_acc: {:0.4f} ###################       ".format(0,train_acc[-1],test_acc[-1]))
    
          
          
    ''' #################################################### block for learning  ########################################################## '''      
    for i in range(1,Epochs+1):
        print("###################### Train Epoch: {} #############################".format(i))
        triplet_label,anch_inds,pos_inds,neg_inds = compute_triplets(model,device,images_train, labels_train)
        if len(triplet_label)<2:
            break

        train_acc1,test_acc1,loss1 = cross_val_function(i,model,device,triplet_label,anch_inds,pos_inds,neg_inds,optimizer,images_train, labels_train,images_test,labels_test)
        
        train_acc.extend(train_acc1)
        test_acc.extend(test_acc1)
        loss.extend(loss1)
        
        torch.save(model.state_dict(), r'models\model'+str(i)+'.pth')    
        
    ''' #################################################### Saving block  ########################################################## ''' 
    
    train_acc   = np.stack(train_acc)
    test_acc    = np.stack(test_acc)
    loss        = np.stack(loss)
    
    np.save(r'data1\train_acc',train_acc)
    np.save(r'data1\test_acc',test_acc)
    np.save(r'data1\loss',loss)
