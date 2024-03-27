from torch.autograd import Variable, Function
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class SimpleGroupWeight(nn.Module):
    def __init__(self, n_groups):
        super(SimpleGroupWeight, self).__init__()
        self.group_count = n_groups
        self.weight = nn.Parameter(torch.FloatTensor(1, n_groups, 1), requires_grad=True)
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.group_count)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, expert_logits):
        # expert_logits [batch_size, group_count, num_class]
        mix_expert_logits = torch.sum(torch.sigmoid(self.weight) * expert_logits, dim=1)
        return mix_expert_logits
    

class MDAWS(nn.Module):
    def __init__(self, args):
        super(MDAWS, self).__init__()
        self.domain_adv = DomainClf(args)
        self.class_count = 2   # number of labels
        # self.source_groups_count = len(self.args.src_domain.split(",")) + 1
        self.source_groups_count = 4   # plus cross
        self.classifiers = nn.Linear(args.hidden_size, self.class_count * self.source_groups_count)
        self.hyper_lambda = 0.5
        self.hyper_beta = 0.5
        self.group_weight = SimpleGroupWeight(n_groups=self.source_groups_count)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
        self.args = args
        self.is_group_weight = True

    def adversarial_loss(self,features, domain_y):
        reverse_features = ReverseLayerF.apply(features, self.hyper_lambda)
        _, domain_loss = self.domain_adv(reverse_features, domain_y)

        return domain_loss
    
    def first_stage(self, src_feature=None, src_domain_y=None,
                               src_rumor_y=None, tgt_feature=None, tgt_domain_y=None):
        # (1) Compute domain classifier loss on both the source domain and the target domain
        src_reverse_features = ReverseLayerF.apply(src_feature, self.hyper_lambda)
        tgt_reverse_features = ReverseLayerF.apply(tgt_feature, self.hyper_lambda)
        # adversarial loss
        # src_domain_y is one-hot vector
        _, src_domain_loss = self.domain_adv(src_reverse_features, src_domain_y)
        _, tgt_domain_loss = self.domain_adv(tgt_reverse_features, tgt_domain_y)
        adv_loss = src_domain_loss + tgt_domain_loss

        # (2) Compute the fake news classifier loss on the source domain only
        # separately train the classifiers
        src_logits = self.classifiers(src_feature)
        # binary classification
        src_logits = src_logits.view(-1, self.source_groups_count, self.class_count)   # [bs, 4, 2]
        # convert the domain_y into one hot
        src_domain_onehot = src_feature.new_zeros(src_feature.shape[0], self.args.domain_class)   # [bs, 4]
        src_domain_onehot.zero_()
        if len(src_domain_y.shape) != 2:
            src_domain_y1 = src_domain_y.unsqueeze(1)
        else:
            src_domain_y1 = src_domain_y
        # mask other source classifier
        src_domain_onehot.scatter_(1, src_domain_y1, 1)
        src_logits = torch.sum(src_logits * src_domain_onehot.unsqueeze(-1), dim=1)
        clf_loss = self.cross_entropy_loss(src_logits, src_rumor_y)

        return clf_loss + adv_loss, (clf_loss, src_logits, src_feature)
    

    def second_stage(self, src_feature=None, src_domain_y=None,src_rumor_y=None,
                     tgt_feature=None, tgt_domain_y=None, tgt_rumor_y=None,
                     tgt_no_feature=None, tgt_no_domain_y=None, tgt_no_rumor_y=None):
        # (1) Train the domain and fake news classifier(s)
        if tgt_no_feature is not None:
            # utilize all the target data to train.
            loss_first, outputs = self.first_stage(src_feature, src_domain_y, src_rumor_y, tgt_no_feature, tgt_no_domain_y)
        else:
            loss_first, outputs = self.first_stage(src_feature, src_domain_y, src_rumor_y, tgt_feature, tgt_domain_y)

        # (2) Predict on the target feature
        expert_logits = self.classifiers(tgt_feature)
        expert_logits = expert_logits.view(-1, self.source_groups_count, self.class_count)   # [bs, 3, 2]
        tgt_domain_onehot = tgt_feature.new_ones(tgt_feature.shape[0], self.args.domain_class)
        if len(src_domain_y.shape) != 2:
            tgt_domain_y1 = tgt_domain_y.unsqueeze(1)
        else:
            tgt_domain_y1 = tgt_domain_y

        tgt_domain_onehot.scatter_(1, tgt_domain_y1, 0)
        expert_logits = tgt_domain_onehot.unsqueeze(-1) * expert_logits
        expert_weight = None
        # (3) Get mix logits
        if self.is_group_weight:
            mix_logits = self.group_weight(expert_logits)
        else:
            if self.args.is_weight_avg:
                mix_logits = torch.mean(expert_logits, dim=1)
            else:
                mix_logits, expert_weight = self.group_weight(tgt_feature, expert_logits, tgt_domain_onehot)


        loss = self.cross_entropy_loss(mix_logits, tgt_rumor_y)
        loss_first = self.hyper_beta * loss_first
        # lambda
        # loss_first
        # + loss_first
        if hasattr(self.args, "is_only_weak") and self.args.is_only_weak:
            loss = loss
        else:
            loss = loss + loss_first
        if expert_weight is None:
            expert_weight = torch.tensor([0.1, 0.2, 0.3])
        src_logits = outputs[1]
        return loss, src_logits, mix_logits, expert_weight
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    

class DomainClf(nn.Module):
    def __init__(self, args):
        super(DomainClf, self).__init__()
        self.args = args
        self.domain_clf = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.Linear(384, self.args.domain_class)
        )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cross_entropy_seperate = nn.CrossEntropyLoss(reduction="none")
        final_dim = (self.args.domain_class - 1) * 2
        self.domain_logit_clf = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.Linear(384, final_dim)
        )

    def set_source(self, target_class):
        self.target_class = target_class
        self.element = [0, 1, 2]
        self.element.pop(target_class)


    def forward(self, hidden, domain_class=None, **kwargs):
        predict_domain = self.domain_clf(hidden)
        output = (predict_domain,)
        if domain_class is not None:
            loss = self.cross_entropy(predict_domain, domain_class)
            output += (loss,)
        return output
