import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        #BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Compute the probability of the ground truth class
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss

# Example
logits = torch.randn(5, 3)  # Assume batch size is 5 and there are 3 classes
logits = torch.tensor([[1., 0., 1.],
                       [0., 1., 0.],
                       [1., 1., 1.],
                       [0., 0., 1.],
                       [0., 1., 0.]])  # Multi-label ground truth
labels = torch.tensor([[1., 0., 1.],
                       [0., 1., 0.],
                       [1., 1., 1.],
                       [0., 0., 1.],
                       [0., 1., 0.]])  # Multi-label ground truth

criterion = FocalLoss()
loss = criterion(logits, labels)
print(loss)
