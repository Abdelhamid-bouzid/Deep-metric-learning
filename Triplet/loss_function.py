import torch
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance

class loss_function(Function):
    def __init__(self, alpha):
        self.alpha  = alpha
        self.pdist  = PairwiseDistance(2)
    
    def forward(self, anchor, positive, negative):
        pos_dist   = self.pdist.forward(anchor, positive)
        neg_dist   = self.pdist.forward(anchor, negative)
        
        pos_mean   = pos_dist.mean(0)
        neg_mean   = neg_dist.mean(0)
        
        pos_var    = pos_dist.var(0)
        neg_var    = neg_dist.var(0)
        
        g_loss1    = pos_var + neg_var 
        g_loss2    = torch.clamp(0.01  + pos_mean - neg_mean, min = 0.0)
        g_loss     = g_loss1 + g_loss2
        
        hinge_dist = torch.clamp(self.alpha  + pos_dist - neg_dist, min = 0.0)
        loss       = torch.mean(hinge_dist) + g_loss
        return loss

        