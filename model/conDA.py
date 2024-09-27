'''
Stores the core modules of conDA model
'''

from torch import nn
import torch
import torch.nn.functional as F
from collections import namedtuple
from model.mmd_code import MMD
from torch.nn import TripletMarginLoss


# (1) MLP in the figure
class ProjectionMLP(nn.Module):
    """
    Model to project [CLS] representation onto
    another space, where the contrastive loss will
    be calculated.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.args.in_dim, cfg.args.in_dim),   # 768 -> 768
            nn.ReLU(),
            nn.Linear(cfg.args.in_dim, cfg.args.proj_dim)   # 768 -> 500
        )

    def forward(self, input_features):
        # input_features: [bs, 768], previously from [:, 0, :]
        return self.layers(input_features)


# (2) Classifier in the figure
class MLLMClassificationHead(nn.Module):
    """
    A classifier following the MLLM embedding
    Reference: https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/roberta/modeling_roberta.py#L1426
    (RobertaClassificationHead, RobertaForSequenceClassification)
    """
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.args.hidden_size, cfg.args.hidden_size)
        classifier_dropout = (
            cfg.args.classifier_dropout if cfg.args.classifier_dropout is not None else cfg.args.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(cfg.args.hidden_size, cfg.args.num_labels)
        self.soft_max = nn.Softmax(dim=1)
        self.num_labels = cfg.args.num_labels

        ###### Add more dense layers to the original classifier ######
        self.ln1 = nn.Linear(768, 1024)
        self.ln2 = nn.Linear(1024, 4096)
        self.ln3 = nn.Linear(4096, 4096)
        self.ln4 = nn.Linear(4096, 1024)
        self.ln5 = nn.Linear(1024, 768)
        self.dense2 = nn.Linear(cfg.args.hidden_size, cfg.args.hidden_size)
        #############################
        
        ###### Add batch normalization ######
        self.bn = nn.BatchNorm1d(cfg.args.hidden_size)
        self.bn1 = nn.BatchNorm1d(cfg.args.hidden_size)
        #############################

    def forward(self, features):
        """
        Return the logits
        """
        # features: [bs, 768], previously from [:, 0, :]
        x = self.dropout(features)
        x = self.dense(x)   # 768 -> 768
        x = self.bn1(x)   # adding this bn boost performance in z->cls
        x = torch.tanh(x)
        ###### Add more dense layers to the original classifier ######
        # x = self.ln1(x)   # 768 -> 768
        # x = self.bn1(x)
        # x = torch.tanh(x)
        # x = self.ln2(x)   # 768 -> 768
        # x = self.bn2(x)
        # x = torch.tanh(x)
        # x = self.ln3(x)   # 768 -> 768
        # x = self.bn3(x)
        # x = torch.tanh(x)
        # x = self.ln4(x)   # 768 -> 768
        # x = self.bn4(x)
        # x = torch.tanh(x)
        # x = self.ln5(x)   # 768 -> 768
        # x = self.bn5(x)
        # x = torch.tanh(x)
        x = self.dense2(x)   # 768 -> 768
        x = self.bn(x)
        x = torch.tanh(x)
        #############################
        x = self.dropout(x)
        x = self.out_proj(x)   # 768 -> 2
        logits = x
        return logits

    # L_CE loss in the figure
    def compute_loss(self, logits, labels):
        # logits is the forward() output
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def compute_softmax_logits(self, logits):
        softmax_logits = self.soft_max(logits)
        return softmax_logits


# (3a) L_CTR in the figure
class SimCLRContrastiveLoss(nn.Module):
    """
    SimCLR style contrastive loss
    """
    def __init__(self, batch_size, temperature=0.5):   # Twitter-COMMs: 0.55   NewsCLIPpings: 0.5
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pair
        (z_i, z_j) as per SimCLR paper
        """
        # Normalize each embedding (no need for BLIP-2?)
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)   # [2*bs, 2*bs]

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)   # sim_ij = sim_ji?
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


# (3b) The overall model
class ContrastiveLearningModule(nn.Module):
    def __init__(self, model, mlp, loss_type, logger, device, lambda_w, lambda_mmd):
        # model, mlp is initialized outside
        super().__init__()
        self.model = model   # RobertaForContrastiveLearning in the original paper, here is MLLMClassificationHead
        self.mlp = mlp   # Projection mlp
        self.loss_type = loss_type   # "simclr"
        self.logger = logger
        self.device = device   # device
        self.lambda_w = lambda_w
        self.lambda_mmd = lambda_mmd

    def forward(self, src_emb, src_perturb_emb, tgt_emb, tgt_perturb_emb, src_labels, tgt_labels):

        src_batch_size = src_emb.shape[0]
        tgt_batch_size = tgt_emb.shape[0]

        # (1) Compute L_CE loss
        # source
        # ori
        src_logits = self.model(src_emb)
        src_LCE_real, src_logits_real = self.model.compute_loss(src_logits, src_labels), self.model.compute_softmax_logits(src_logits)

        # perturb
        src_perturb_logits = self.model(src_perturb_emb)
        src_LCE_perturb, src_logits_perturb = self.model.compute_loss(src_perturb_logits, src_labels), self.model.compute_softmax_logits(src_perturb_logits)

        # target
        # ori
        tgt_logits = self.model(tgt_emb)
        tgt_LCE_real, tgt_logits_real = self.model.compute_loss(tgt_logits, tgt_labels), self.model.compute_softmax_logits(tgt_logits)

        # perturb
        tgt_perturb_logits = self.model(tgt_perturb_emb)
        tgt_LCE_perturb, tgt_logits_perturb = self.model.compute_loss(tgt_perturb_logits, tgt_labels), self.model.compute_softmax_logits(tgt_perturb_logits)

        # (2) Compute Contrastive losses (simclr supported now)

        if self.loss_type == "simclr":
            if src_batch_size == tgt_batch_size:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = src_ctr_loss
            else:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = SimCLRContrastiveLoss(batch_size=tgt_batch_size)
                tgt_ctr_loss.to(self.device)

        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_emb)
            src_z_j = self.mlp(src_perturb_emb)
            src_lctr = src_ctr_loss(src_z_i, src_z_j)
            tgt_z_i = self.mlp(tgt_emb)
            tgt_z_j = self.mlp(tgt_perturb_emb)
            tgt_lctr = tgt_ctr_loss(tgt_z_i, tgt_z_j)

        # Full loss
        # (3) Compute MMD loss
        mmd = MMD(src_z_i, tgt_z_i, kernel='rbf')

        use_ce_perturb = True
        use_both_ce_losses = True
        lambda_mmd = 1.0

        if not use_both_ce_losses:
            loss = self.lambda_w * (src_lctr + tgt_lctr) / 2 + lambda_mmd * mmd
        else:
            if use_ce_perturb:
                loss = (1 - self.lambda_w) * (src_LCE_real + src_LCE_perturb) / 2 \
                       + self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd
            else:
                loss = (1 - self.lambda_w) * src_LCE_real \
                       + self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd

        data = {"total_loss": loss, "src_ctr_loss": src_lctr, "tgt_ctr_loss": tgt_lctr, "src_ce_loss_real": src_LCE_real,
                "src_ce_loss_perturb": src_LCE_perturb, "mmd": mmd, "src_logits": src_logits_real,
                "tgt_logits":tgt_logits_real}

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
            data = data_named_tuple(**data)
        elif isinstance(data, list):
            data = tuple(data)

        return data
    


# (3b)' The overall model + our proposed triplet loss
class ContrastiveLearningAndTripletLossModule(nn.Module):
    def __init__(self, model, mlp, loss_type, logger, device, lambda_w):
        # model, mlp is initialized outside
        super().__init__()
        self.model = model   # RobertaForContrastiveLearning in the original paper, here is MLLMClassificationHead
        self.mlp = mlp   # Projection mlp
        self.loss_type = loss_type   # "simclr"
        self.logger = logger
        self.device = device   # device
        self.lambda_w = lambda_w

    def forward(self, src_emb, src_perturb_emb, src_negative_emb, tgt_emb, tgt_perturb_emb, src_labels, tgt_labels):

        src_batch_size = src_emb.shape[0]
        tgt_batch_size = tgt_emb.shape[0]

        # (1) Compute L_CE loss
        # source
        # original
        src_logits = self.model(src_emb)
        src_LCE_real, src_logits_real = self.model.compute_loss(src_logits, src_labels), self.model.compute_softmax_logits(src_logits)

        # perturb
        src_perturb_logits = self.model(src_perturb_emb)
        src_LCE_perturb, src_logits_perturb = self.model.compute_loss(src_perturb_logits, src_labels), self.model.compute_softmax_logits(src_perturb_logits)

        ######### negative #########
        src_negative_logits = self.model(src_negative_emb)
        src_LCE_negative, src_negative_perturb = self.model.compute_loss(src_negative_logits, 1-src_labels), self.model.compute_softmax_logits(src_negative_logits)
        ############################

        # target
        # original
        tgt_logits = self.model(tgt_emb)
        tgt_LCE_real, tgt_logits_real = self.model.compute_loss(tgt_logits, tgt_labels), self.model.compute_softmax_logits(tgt_logits)

        # perturb
        tgt_perturb_logits = self.model(tgt_perturb_emb)
        tgt_LCE_perturb, tgt_logits_perturb = self.model.compute_loss(tgt_perturb_logits, tgt_labels), self.model.compute_softmax_logits(tgt_perturb_logits)

        # (2) Compute Contrastive losses (simclr supported now)

        if self.loss_type == "simclr":
            if src_batch_size == tgt_batch_size:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = src_ctr_loss
            else:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = SimCLRContrastiveLoss(batch_size=tgt_batch_size)
                tgt_ctr_loss.to(self.device)
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_emb)   # original
            src_z_j = self.mlp(src_perturb_emb)   # positive
            src_z_k = self.mlp(src_negative_emb)   # negative
            src_lctr = src_ctr_loss(src_z_i, src_z_j)
            src_ltriplet = triplet_loss(src_z_i, src_z_j, src_z_k)
            tgt_z_i = self.mlp(tgt_emb)   # original
            tgt_z_j = self.mlp(tgt_perturb_emb)   # positive
            tgt_lctr = tgt_ctr_loss(tgt_z_i, tgt_z_j)

        # Full loss
        # (3) Compute MMD loss
        mmd = MMD(src_z_i, tgt_z_i, kernel='rbf')

        use_ce_perturb = True
        use_both_ce_losses = True
        lambda_mmd = 1.0

        if not use_both_ce_losses:
            loss = self.lambda_w * (src_lctr + tgt_lctr) / 2 + lambda_mmd * mmd
        else:
            if use_ce_perturb:   # lambda_w: 0.5, lambda_mmd: 1.0
                loss = (1 - self.lambda_w) * (src_LCE_real + src_LCE_perturb + src_LCE_negative) / 2 \
                       + self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd \
                       + src_ltriplet   # triplet loss
            else:
                loss = (1 - self.lambda_w) * (src_LCE_real + src_LCE_negative) \
                       + self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd \
                       + src_ltriplet  # triplet loss

        data = {"total_loss": loss, "src_ctr_loss": src_lctr, "src_triplet_loss": src_ltriplet, "tgt_ctr_loss": tgt_lctr, "src_ce_loss_real": src_LCE_real,
                "src_ce_loss_perturb": src_LCE_perturb, "src_ce_loss_negative": src_LCE_negative, "mmd": mmd, "src_logits": src_logits_real,
                "tgt_logits":tgt_logits_real}

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
            data = data_named_tuple(**data)
        elif isinstance(data, list):
            data = tuple(data)

        return data
    

# (3b)'' The overall model + our proposed triplet loss + input z to the classifier instead of h
class ContrastiveLearningAndTripletLossZModule(nn.Module):
    def __init__(self, model, mlp, loss_type, logger, device, lambda_w, lambda_mmd):
        # model, mlp is initialized outside
        super().__init__()
        self.model = model   # RobertaForContrastiveLearning in the original paper, here is MLLMClassificationHead
        self.mlp = mlp   # Projection mlp
        self.loss_type = loss_type   # "simclr"
        self.logger = logger
        self.device = device   # device
        self.lambda_w = lambda_w
        self.lambda_mmd = lambda_mmd

    def forward(self, src_emb, src_perturb_emb, src_negative_emb, tgt_emb, tgt_perturb_emb, src_labels, tgt_labels):

        src_batch_size = src_emb.shape[0]
        tgt_batch_size = tgt_emb.shape[0]

        # (2) Compute Contrastive losses (simclr supported now)

        if self.loss_type == "simclr":
            if src_batch_size == tgt_batch_size:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = src_ctr_loss
            else:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = SimCLRContrastiveLoss(batch_size=tgt_batch_size)
                tgt_ctr_loss.to(self.device)
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_emb)   # original
            src_z_j = self.mlp(src_perturb_emb)   # positive
            src_z_k = self.mlp(src_negative_emb)   # negative
            src_lctr = src_ctr_loss(src_z_i, src_z_j)
            src_ltriplet = triplet_loss(src_z_i, src_z_j, src_z_k)
            tgt_z_i = self.mlp(tgt_emb)   # original
            tgt_z_j = self.mlp(tgt_perturb_emb)   # positive
            tgt_lctr = tgt_ctr_loss(tgt_z_i, tgt_z_j)


        # (1) Compute L_CE loss
        # source
        # original
        src_logits = self.model(src_z_i)
        src_LCE_real, src_logits_real = self.model.compute_loss(src_logits, src_labels), self.model.compute_softmax_logits(src_logits)

        # perturb
        src_perturb_logits = self.model(src_z_j)
        src_LCE_perturb, src_logits_perturb = self.model.compute_loss(src_perturb_logits, src_labels), self.model.compute_softmax_logits(src_perturb_logits)

        ######### negative #########
        src_negative_logits = self.model(src_z_k)
        src_LCE_negative, src_negative_perturb = self.model.compute_loss(src_negative_logits, 1-src_labels), self.model.compute_softmax_logits(src_negative_logits)
        ############################

        # target
        # original
        tgt_logits = self.model(tgt_z_i)
        tgt_LCE_real, tgt_logits_real = self.model.compute_loss(tgt_logits, tgt_labels), self.model.compute_softmax_logits(tgt_logits)

        # perturb
        tgt_perturb_logits = self.model(tgt_z_j)
        tgt_LCE_perturb, tgt_logits_perturb = self.model.compute_loss(tgt_perturb_logits, tgt_labels), self.model.compute_softmax_logits(tgt_perturb_logits)

        # Full loss
        # (3) Compute MMD loss
        mmd = MMD(src_z_i, tgt_z_i, kernel='rbf')

        use_ce_perturb = True
        use_both_ce_losses = True
        # lambda_mmd = 1.0   # original: 1.0
        lambda_mmd = self.lambda_mmd

        if not use_both_ce_losses:
            loss = self.lambda_w * (src_lctr + tgt_lctr) / 2 + lambda_mmd * mmd
        else:
            if use_ce_perturb:   # lambda_w: 0.5, lambda_mmd: 1.0
                loss = (1 - self.lambda_w) * (src_LCE_real + src_LCE_perturb + src_LCE_negative) / 2 \
                       + 1 * self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd \
                       + src_ltriplet   # triplet loss  # commented out for newsclipping experiments
            else:
                loss = (1 - self.lambda_w) * (src_LCE_real + src_LCE_negative) \
                       + 1 * self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd \
                       + src_ltriplet  # triplet loss

        data = {"total_loss": loss, "src_ctr_loss": src_lctr, "src_triplet_loss": src_ltriplet, "tgt_ctr_loss": tgt_lctr, "src_ce_loss_real": src_LCE_real,
                "src_ce_loss_perturb": src_LCE_perturb, "src_ce_loss_negative": src_LCE_negative, "mmd": mmd, "src_logits": src_logits_real,
                "tgt_logits":tgt_logits_real}

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
            data = data_named_tuple(**data)
        elif isinstance(data, list):
            data = tuple(data)

        return data


# (3b)''' The overall model + our proposed triplet loss + input z to the classifier instead of h - lce_neg - ltriplet
class ContrastiveLearningLossZModule(nn.Module):
    def __init__(self, model, mlp, loss_type, logger, device, lambda_w, lambda_mmd):
        # model, mlp is initialized outside
        super().__init__()
        self.model = model   # RobertaForContrastiveLearning in the original paper, here is MLLMClassificationHead
        self.mlp = mlp   # Projection mlp
        self.loss_type = loss_type   # "simclr"
        self.logger = logger
        self.device = device   # device
        self.lambda_w = lambda_w
        self.lambda_mmd = lambda_mmd

    def forward(self, src_emb, src_perturb_emb, src_negative_emb, tgt_emb, tgt_perturb_emb, src_labels, tgt_labels):

        src_batch_size = src_emb.shape[0]
        tgt_batch_size = tgt_emb.shape[0]

        # (2) Compute Contrastive losses (simclr supported now)

        if self.loss_type == "simclr":
            if src_batch_size == tgt_batch_size:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = src_ctr_loss
            else:
                src_ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                src_ctr_loss.to(self.device)
                tgt_ctr_loss = SimCLRContrastiveLoss(batch_size=tgt_batch_size)
                tgt_ctr_loss.to(self.device)
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_emb)   # original
            src_z_j = self.mlp(src_perturb_emb)   # positive
            src_lctr = src_ctr_loss(src_z_i, src_z_j)
            tgt_z_i = self.mlp(tgt_emb)   # original
            tgt_z_j = self.mlp(tgt_perturb_emb)   # positive
            tgt_lctr = tgt_ctr_loss(tgt_z_i, tgt_z_j)


        # (1) Compute L_CE loss
        # source
        # original
        src_logits = self.model(src_z_i)
        src_LCE_real, src_logits_real = self.model.compute_loss(src_logits, src_labels), self.model.compute_softmax_logits(src_logits)

        # perturb
        src_perturb_logits = self.model(src_z_j)
        src_LCE_perturb, src_logits_perturb = self.model.compute_loss(src_perturb_logits, src_labels), self.model.compute_softmax_logits(src_perturb_logits)

        # ######### negative #########
        # src_negative_logits = self.model(src_z_k)
        # src_LCE_negative, src_negative_perturb = self.model.compute_loss(src_negative_logits, 1-src_labels), self.model.compute_softmax_logits(src_negative_logits)
        # ############################

        # target
        # original
        tgt_logits = self.model(tgt_z_i)
        tgt_LCE_real, tgt_logits_real = self.model.compute_loss(tgt_logits, tgt_labels), self.model.compute_softmax_logits(tgt_logits)

        # perturb
        tgt_perturb_logits = self.model(tgt_z_j)
        tgt_LCE_perturb, tgt_logits_perturb = self.model.compute_loss(tgt_perturb_logits, tgt_labels), self.model.compute_softmax_logits(tgt_perturb_logits)

        # Full loss
        # (3) Compute MMD loss
        mmd = MMD(src_z_i, tgt_z_i, kernel='rbf')

        use_ce_perturb = True
        use_both_ce_losses = True
        # lambda_mmd = 1.0   # original: 1.0
        lambda_mmd = self.lambda_mmd

        if not use_both_ce_losses:
            loss = self.lambda_w * (src_lctr + tgt_lctr) / 2 + lambda_mmd * mmd
        else:
            if use_ce_perturb:   # lambda_w: 0.5, lambda_mmd: 1.0
                loss = (1 - self.lambda_w) * (src_LCE_real + src_LCE_perturb) / 2 \
                       + 75 * self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd \
                    #    + src_ltriplet   # triplet loss  # commented out for newsclipping experiments
            else:
                loss = (1 - self.lambda_w) * (src_LCE_real) \
                       + 75 * self.lambda_w * (src_lctr + tgt_lctr) / 2 \
                       + lambda_mmd * mmd \
                    #    + src_ltriplet  # triplet loss

        data = {"total_loss": loss, "src_ctr_loss": src_lctr, "tgt_ctr_loss": tgt_lctr, "src_ce_loss_real": src_LCE_real,
                "src_ce_loss_perturb": src_LCE_perturb, "mmd": mmd, "src_logits": src_logits_real,
                "tgt_logits":tgt_logits_real}

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
            data = data_named_tuple(**data)
        elif isinstance(data, list):
            data = tuple(data)

        return data

