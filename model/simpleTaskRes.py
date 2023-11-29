from torch import nn
import torch

class SimpleTaskResLearner(nn.Module):
    def __init__(self, base_multimodal_features):
        super(SimpleTaskResLearner, self).__init__()

        self.multimodal_feature_residuals = nn.Parameter(torch.zeros_like(base_multimodal_features))

    def forward(self):
        return self.base_multimodal_features + self.alpha * self.multimodal_feature_residuals