import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from loss_function import loss_function
from test_model import test_model
from config import Facnet_config
from forward_all_images import forward_all_images
          
def cross_val_function(epoch,model,device,triplet_label,anch_inds,pos_inds,neg_inds,optimizer,images_train, labels_train,images_test,labels_test):
    
    ''' #################################################  initialization of epoch ################################################### '''
    batch_size          = Facnet_config['batch_size']
    num_iter            = Facnet_config['num_iter']
    max_batch           = Facnet_config['max_batch']
    alpha               = Facnet_config['alpha']
    
    arr = np.arange(anch_inds.shape[0])
    np.random.shuffle(arr)
    
    anch_inds, pos_inds, neg_inds, triplet_label      = anch_inds[arr], pos_inds[arr], neg_inds[arr], triplet_label[arr]
    
    nb_split = np.rint(anch_inds.shape[0]/batch_size).astype(int)
    if nb_split==0 or nb_split==1:
        nb_split = 2
    print(nb_split)
    skf = StratifiedKFold(n_splits=nb_split)
    
    train_acc, test_acc,loss1 = [],[],[]
    
    ''' #################################################  satrt epoch learning ################################################### '''
    m = 0
    for train_index, test_index in skf.split(triplet_label, triplet_label):
        if m==max_batch:
            break
        
        X_train, X_test = images_train[anch_inds[train_index]], images_train[anch_inds[test_index]]
        y_train, y_test = triplet_label[train_index], triplet_label[test_index]
        X_pos = images_train[pos_inds[test_index].astype(int)]
        X_neg = images_train[neg_inds[test_index].astype(int)]
        
        anchor1      = torch.from_numpy(X_test).to(device=device, dtype=torch.float)
        positive1    = torch.from_numpy(X_pos).to(device=device, dtype=torch.float)
        negative1    = torch.from_numpy(X_neg).to(device=device, dtype=torch.float)
        
        for k in range(num_iter):
            
            '''######################################### loss function ######################################################'''
            model.train()
            anchor,positive,negative     = model(anchor1), model(positive1), model(negative1)
 
            loss   = loss_function(alpha).forward(anchor, positive, negative).to(device)
            if loss.item()==0:
                model.eval()
                break;
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            
            '''############################################ Compute KNN #####################################################'''
            train_features = forward_all_images(model,device,images_train)
            test_features  = forward_all_images(model,device,images_test)
           
            _,train_acc_1  = test_model(train_features, labels_train,train_features,labels_train)
            _,test_acc_1   = test_model(train_features, labels_train,test_features,labels_test)
            
            train_acc.append(train_acc_1)
            test_acc.append(test_acc_1)
            
            loss1.append(loss.item())
            torch.save(model.state_dict(), r'models\model_'+str(epoch)+'_'+str(m)+'.pth')
            print("#### epoch : {} Fold number : {} iterataion number : {} train_acc: {:0.4f}  test_acc: {:0.4f} loss: {:0.8f} #####".format(epoch,m,k,train_acc[-1],test_acc[-1],loss.item()))
            #print("#### epoch : {} Fold number : {} iterataion number : {} #####".format(epoch,m,k))

        m = m+1

    return train_acc,test_acc,loss1
        