import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class MLLMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        classifier_dropout = (
            args.classifier_dropout if args.classifier_dropout is not None else args.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(args.hidden_size, args.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

# Main model
class ContrastiveModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_labels = args.num_labels
        self.args = args
        self.classifier = MLLMClassificationHead(args)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
    
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0-total1)**2).sum(2)
    
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def mmd(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss
    
    def forward(
        self,
        inputs_embeds=None,
        labels=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logits = self.classifier(inputs_embeds)

        loss = None
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        data = {"loss": loss, "logits": logits}
        return data
    
    def forward_ours(
        self,
        src_emb, tgt_emb, src_labels, tgt_labels,
        alpha=0.1,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        inputs_embeds = torch.cat([src_emb, tgt_emb], dim=0)
        labels = torch.cat([src_labels, tgt_labels], dim=0)
        logits = self.classifier(inputs_embeds)

        loss = None
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # sequence_output = sequence_output[:, 0, :]
        # if source_target_split is None:
        #     source_target_split = sequence_output.size(0) // 2
        
        # source_sequence_output = sequence_output[:source_target_split]
        # source_labels = labels[:source_target_split]
        # source_pos_output = source_sequence_output[(source_labels==1).nonzero()[:, 0]]
        # source_neg_output = source_sequence_output[(source_labels==0).nonzero()[:, 0]]

        # target_sequence_output = sequence_output[source_target_split:]
        # target_labels = labels[source_target_split:]
        # target_pos_output = target_sequence_output[(target_labels==1).nonzero()[:, 0]]
        # target_neg_output = target_sequence_output[(target_labels==0).nonzero()[:, 0]]
        
        # neg_output = sequence_output[(labels==0).nonzero()[:, 0]]
        # pos_output = sequence_output[(labels==1).nonzero()[:, 0]]
        src_neg_emb = src_neg_emb[(src_labels==1).nonzero()[:, 0]]
        src_pos_emb = src_pos_emb[(src_labels==1).nonzero()[:, 0]]

        tgt_neg_emb = tgt_neg_emb[(tgt_labels==1).nonzero()[:, 0]]
        tgt_pos_emb = tgt_pos_emb[(tgt_labels==1).nonzero()[:, 0]]

        pos_emb = inputs_embeds[(labels==0).nonzero()[:, 0]]
        neg_emb = inputs_embeds[(labels==1).nonzero()[:, 0]]

        if len(src_neg_emb) > 0 and len(tgt_neg_emb) > 0:
            loss += alpha * self.mmd(src_neg_emb, tgt_neg_emb)
        if len(src_pos_emb) > 0 and len(tgt_pos_emb) > 0:
            loss += alpha * self.mmd(src_pos_emb, tgt_pos_emb)
        if len(pos_emb) > 0 and len(neg_emb) > 0:
            loss -= alpha * self.mmd(pos_emb, neg_emb)
        
        data = {"loss": loss, "logits": logits}
