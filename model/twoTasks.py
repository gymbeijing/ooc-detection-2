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
        # x.shape: [bs, 768]
        # print(f"self.simple_taskres_learner().shape: {self.simple_taskres_learner().shape}")  # [3, 768]
        # print(f"x.t().shape: {x.t().shape}")   # [768, bs]
        domain_similarity_scores = self.simple_taskres_learner() @ (x.t())   # to be converted to logits
        # print(domain_similarity_scores.shape)   # [3, 256]
        out_scores = self.linear_classifier(x)
        return domain_similarity_scores.t(), out_scores   # [256, 3], [256, 2]

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
