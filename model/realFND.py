import torch.nn as nn
import torch.nn.functional as F

class FakeNewsClassifier(nn.Module):
    def __init__(self):
        super(FakeNewsClassifier, self).__init__()
        self.cls = nn.Sequential()
        self.cls.add_module('fc', nn.Linear(768, 2))   # (fc): 768 -> 2
        self.cls.add_module('dropout', nn.Dropout(0.2))
        self.cls.add_module('tanh', nn.Tanh())
        self.cls.add_module('sigmoid', nn.Sigmoid())

    def forward(self, input):
        output = self.cls(input)
        return output
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class DomainClassifier(nn.Module):
    def __init__(self, event_num):
        super(DomainClassifier, self).__init__()
        self.cls = nn.Sequential()
        self.cls.add_module('fc', nn.Linear(768, event_num))   # (fc): 768 -> 2
        self.cls.add_module('dropout', nn.Dropout(0.2))
        self.cls.add_module('tanh', nn.Tanh())
        self.cls.add_module('sigmoid', nn.Sigmoid())

    def forward(self, input):
        output = self.cls(input)
        # print(output)
        return output
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    

class Agent(nn.Module):
    def __init__(self, action_num):
        super(Agent, self).__init__()
        self.affine1 = nn.Linear(768, 768)
        self.affine2 = nn.Linear(768, action_num)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
