import torch 
import numpy as np
from config import Facnet_config

def forward_all_images(model,device,all_img):
    batch_size          = Facnet_config['batch_size']
    
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_img     = torch.from_numpy(all_img)

    all_features  = []
    for i in range(0,all_img.shape[0],batch_size):
        if i+batch_size == all_img.shape[0]-1 or i+batch_size > all_img.shape[0]-1:
            s         = i
            data      = all_img[s:].to(device=device, dtype=torch.float)
            features1 = model(data)
        else:
            s = i
            e = s + batch_size
            
            data      = all_img[s:e].to(device=device, dtype=torch.float)
            features1 = model(data)
            
        all_features.append(features1.cpu().detach().numpy())
    
    all_features = np.concatenate(all_features,axis=0)
        
    return all_features
