from torch import nn
import torch
import torch.nn.functional as F
from collections import namedtuple
from mmd_code import MMD


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
            nn.Linear(cfg.in_dim, cfg.in_dim),
            nn.ReLU(),
            nn.Linear(cfg.in_dim, cfg.proj_dim)
        )

    def forward(self, input_features):
        # input_features: [bs, 768], previously from [:, 0, :]
        # x = input_features[:, 0, :]
        return self.layers(input_features)


# (2) Classifier in the figure
class MLLMClassificationHead(nn.Module):
    """
    A classifier following the MLLM embedding
    """
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        classifier_dropout = (
            cfg.classifier_dropout if cfg.classifier_dropout is not None else cfg.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, features):
        """
        Return the logits
        """
        # features: [bs, 768], previously from [:, 0, :]
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
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
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negative_mask", (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float())

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
    def __init__(self, model, mlp, loss_type, logger, device, lambda_w):
        # model, mlp is initialized outside
        super().__init__()
        self.model = model   # RobertaForContrastiveLearning in the original paper, here is MLLMClassificationHead
        self.mlp = mlp   # Projection mlp
        self.loss_type = loss_type   # "simclr"
        self.logger = logger
        self.device = device   # device
        self.lambda_w = lambda_w

    def forward(self, src_emb, src_perturb_emb, tgt_emb, tgt_perturb_emb, src_labels, tgt_labels):

        batch_size = src_emb.shape[0]

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
            ctr_loss = SimCLRContrastiveLoss(batch_size=batch_size)
            ctr_loss.to(self.device)

        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_emb)
            src_z_j = self.mlp(src_perturb_emb)
            src_lctr = ctr_loss(src_z_i, src_z_j)
            tgt_z_i = self.mlp(tgt_emb)
            tgt_z_j = self.mlp(tgt_perturb_emb)
            tgt_lctr = ctr_loss(tgt_z_i, tgt_z_j)

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




