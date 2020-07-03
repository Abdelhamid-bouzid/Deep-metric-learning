import torch 
import numpy as np
from config import Facnet_config

def forward_all_images(model,device,all_img):
    batch_size          = Facnet_config['batch_size']
    if all_img.shape[0]==0:
        flag = 0
        N    = 0
    elif all_img.shape[0]<batch_size:
        flag = 1
        N = all_img.shape[0]
        step = 1
    else:
        flag = 1
        N = int(all_img.shape[0]/batch_size)
        step = batch_size
    
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_img     = torch.from_numpy(all_img).to(device=device, dtype=torch.float)

    all_features  = []
    for i in range(N):
        if i== N-1:
            s = i*step
            data = all_img[s:]
            features1   = model(data)
        else:
            s = i*step
            e = (i+1)*step
            data = all_img[s:e]
            features1   = model(data)
            
        all_features.append(features1.cpu().detach().numpy())
        
    if len(all_features)>0:
        all_features = np.concatenate(all_features,axis=0)
        
    all_img = all_img.cpu().detach().numpy()
    return all_features
