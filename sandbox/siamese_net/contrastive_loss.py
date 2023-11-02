import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute the Euclidean distance between the two outputs
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Compute the loss
        loss = 0.5 * (label * euclidean_distance.pow(2) +
                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss.mean()



