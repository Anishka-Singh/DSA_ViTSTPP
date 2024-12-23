
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.fc = nn.Linear(256, 128)  

    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc1(x)


def info_nce_loss(features, labels, weight_matrix):
    
    scores = torch.matmul(features, weight_matrix)
    scores = torch.matmul(scores, labels.T)
    
    nce_loss = -torch.mean(torch.diag(scores) - torch.logsumexp(scores, dim=1))
    return nce_loss


feature_extractor = FeatureExtractor()
mlp_causal = MLP(128, 64)  
mlp_noncausal = MLP(128, 64)  
weight_matrix = torch.randn(64, 64)  


x = torch.randn(10, 256)  
y = torch.randn(10, 64)  


z = feature_extractor(x)


f_c = mlp_causal(z)
f_n = mlp_noncausal(z)


loss = info_nce_loss(f_c, y, weight_matrix)

loss.item()  




