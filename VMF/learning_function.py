from forward_all_images import forward_all_images
from prediction import prediction
from loss_function import loss_function
from RandomSampler import RandomSampler
from torch.utils.data import DataLoader
from mean_dicrections import mean_dicrections

from cluster_training import cluster_training
import torch
from config import config

def learning_function(model,l_train,test):
    
    
    ''' #################################################  initialization  ################################################### '''
    
    optimizer     = torch.optim.Adam(model.parameters(),lr = config['learning_rate'])
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    train_acc, test_acc = [], []
    
    ''' #################################################  test the initial model  ################################################### '''
    model     = model.to(device=device, dtype=torch.float)
    model.eval()
    train_features,train_labels = forward_all_images(model,l_train)
    test_features,test_labels   = forward_all_images(model,test)
    
    mean_dir                    = mean_dicrections(train_features,train_labels)
    
    _,train_acc_1  = prediction(train_features,train_labels,mean_dir)
    _,test_acc_1   = prediction(test_features,test_labels,mean_dir)
    
    train_acc.append(train_acc_1)
    test_acc.append(test_acc_1)
    
    print("######################################################################################################################")
    print("   #####  Cosine: Train Epoch: {} train_acc: {:0.4f} test_acc: {:0.4f} #####".format(0,train_acc_1,test_acc_1))
    print("######################################################################################################################")
    
    ''' #################################################  Dtat loaders  ################################################### '''
    train_sampler=RandomSampler(len(l_train), config["iteration"] * config["batch_size"])
    l_loader = DataLoader(l_train, config["batch_size"],drop_last=True,sampler=train_sampler)
      
          
    ''' #################################################### block for learning  ########################################################## '''      
    best_acc  = 0
    iteration = 0
    for l_input, l_target in l_loader:
        iteration += 1
        
        l_input, l_target = l_input.to(device).float(), l_target.to(device).long()
        
        outputs = model(l_input)
        
        mean_dir1  = torch.from_numpy(mean_dir).to(device=device, dtype=torch.float)
        loss = loss_function(config["Con"]).forward(outputs, l_target, torch.t(mean_dir1),device)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        #print('##########################################################################################################')
        #print("   #####  Train iteration: {} train_loss: {:0.4f} ".format(iteration,loss.item()))
        #print('##########################################################################################################')
        
        if iteration == config["lr_decay_iter"]:
            optimizer.param_groups[0]["lr"] *= config["lr_decay_factor"]
        
        if iteration%config["mean_dir_cycle"]==0:
            model     = model.to(device=device, dtype=torch.float)
            train_features,train_labels = forward_all_images(model,l_train)
            mean_dir                    = mean_dicrections(train_features,train_labels)
            model.train()
            
        if iteration%config["test_model_cycel"]==0:
            model     = model.to(device=device, dtype=torch.float)
            model.eval()
            
            train_features,train_labels = forward_all_images(model,l_train)
            test_features,test_labels   = forward_all_images(model,test)
            
            mean_dir                    = mean_dicrections(train_features,train_labels)
            
            _,train_acc_1  = prediction(train_features,train_labels,mean_dir)
            _,test_acc_1   = prediction(test_features,test_labels,mean_dir)
            
            train_acc.append(train_acc_1)
            test_acc.append(test_acc_1)
            
            print("######################################################################################################################")
            print("   #####  Cosine: Train iteration: {} train_acc: {:0.4f} test_acc: {:0.4f} #####".format(iteration,train_acc_1,test_acc_1))
            print("######################################################################################################################")
            
            train_acc.append(train_acc_1)
            test_acc.append(test_acc_1)
            
            model.train()
            if test_acc_1>best_acc:
                best_acc = test_acc_1
                torch.save(model,'models/model.pth')
            
        
        
       
    return train_acc,test_acc