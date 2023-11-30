from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(LinearClassifier, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim

    def forward(self, x):
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