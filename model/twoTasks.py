from torch import nn
from model.simpleTaskRes import SimpleTaskResLearner
from model.linearClassifier import LinearClassifier, normal_init


class TwoTasks(nn.Module):
    def __init__(self,  alpha, base_text_features, in_dim, out_dim=2):
        super().__init__()
        self.simple_taskres_learner = SimpleTaskResLearner(alpha, base_text_features)
        self.linear_classifier = LinearClassifier(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        class_similarity_scores = self.simple_taskres_learner() @ x.t()   # to be converted to logits
        out_scores = self.fc(x)
        return class_similarity_scores, out_scores

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
