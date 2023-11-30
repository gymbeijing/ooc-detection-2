from torch import nn
import torch

class SimpleTaskResLearner(nn.Module):
    def __init__(self, alpha, in_dim, out_dim=2):
        super(SimpleTaskResLearner, self).__init__()

        self.multimodal_feature_residuals = nn.Parameter(torch.zeros([in_dim]))
        self.fc = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.alpha = alpha

    def forward(self, base_multimodal_features):
        x = base_multimodal_features + self.alpha * self.multimodal_feature_residuals
        x = x.view(-1, self.in_dim)
        out = self.fc(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()