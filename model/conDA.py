from torch import nn
import torch
import torch.nn.functional as F
from collections import namedtuple

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
        x = input_features[:, 0, :]
        return self.layers(x)


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


class ContrastiveLearningModule(nn.Module):
    def __init__(self, model, mlp, loss_type, logger, device, lambda_w):
        super().__init__()
        self.model = model
        self.mlp = mlp
        self.loss_type = loss_type
        self.logger = logger
        self.device = device
        self.lambda_w = lambda_w

    def forward(self, src_texts, src_masks, src_texts_perturb, src_masks_perturb,
                tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb,
                src_labels, tgt_labels):

        batch_size = src_texts.shape[0]

        # source
        # ori
        src_output_dic = self.model(src_texts, attention_mask=src_masks, labels=src_labels)
        src_LCE_real, src_logits_real = src_output_dic["loss"], src_output_dic["logits"]

        # perturb
        src_output_dic_perturbed = self.model(src_texts_perturb, attention_mask=src_masks_perturb, labels=src_labels)
        src_LCE_perturb, src_logits_perturb = src_output_dic_perturbed["loss"], src_output_dic_perturbed["logits"]

        # target
        # ori
        tgt_output_dic = self.model(tgt_texts, attention_mask=tgt_masks, labels=tgt_labels)
        tgt_LCE_real, tgt_logits_real = tgt_output_dic["loss"], tgt_output_dic["logits"]

        # perturb
        tgt_output_dic_perturbed = self.model(tgt_texts_perturb, attention_mask=tgt_masks_perturb, labels=tgt_labels)
        tgt_LCE_perturb, tgt_logits_perturb = tgt_output_dic_perturbed["loss"], tgt_output_dic_perturbed["logits"]

        # Contrastive losses (simclr supported now)

        if self.loss_type == "simclr":
            ctr_loss = SimCLRContrastiveLoss(batch_size=batch_size)
            ctr_loss.to(self.device)

        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_output_dic["last_hidden_state"])
            src_z_j = self.mlp(src_output_dic_perturbed["last_hidden_state"])
            src_lctr = ctr_loss(src_z_i, src_z_j)
            tgt_z_i = self.mlp(tgt_output_dic["last_hidden_state"])
            tgt_z_j = self.mlp(tgt_output_dic_perturbed["last_hidden_state"])
            tgt_lctr = ctr_loss(tgt_z_i, tgt_z_j)

        # Full loss

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




