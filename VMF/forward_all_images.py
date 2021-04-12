import numpy as np
import torch 
from config import config
from torch.utils.data import DataLoader

def forward_all_images(S_model,data):
    
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader   = DataLoader(data, config["batch_size"],drop_last=False)
    
    S_model.eval()
    
    all_features  = []
    all_labels    = []
    for images, target in data_loader:
        images, target = images.to(device).float(), target
        
        outputs = S_model(images)
        all_features.append(outputs.cpu().detach().numpy())
        all_labels.append(target.numpy())
    all_features = np.concatenate(all_features,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)
    
        
    return all_features, all_labels
