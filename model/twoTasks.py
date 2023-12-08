from torch import nn
from model.simpleTaskRes import SimpleTaskResLearner, _get_base_text_features
from model.linearClassifier import LinearClassifier, normal_init
import pytorch_lightning as pl   # 2.1.2
from torch import optim
from torch.nn import functional as F
from torchmetrics.classification import BinaryAccuracy
import torch


class TwoTasksLightning(pl.LightningModule):
    def __init__(self, cfg):   # cfg: alpha, in_dim
        super().__init__()
        self.classnames = ["climate", "covid", "military"]
        base_text_features = _get_base_text_features(self.classnames)
        self.simple_taskres_learner = SimpleTaskResLearner(cfg, base_text_features)
        self.linear_classifier = LinearClassifier(cfg.in_dim, cfg.out_dim)
        self.compute_accuracy = BinaryAccuracy(threshold=cfg.args.threshold)
        self.validation_num_correct = {"covid": 0, "climate": 0, "military": 0}
        self.validation_num_total = {"covid": 0, "climate": 0, "military": 0}
        self.threshold = cfg.args.threshold

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-5)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        input = train_batch["multimodal_emb"]
        label = train_batch["label"]
        domain_id = train_batch["domain_id"]   # 0, 1, 2
        domain_name = train_batch["domain_name"]

        domain_similarity_scores, y_pred = self.forward(input)
        domain_loss = F.cross_entropy(domain_similarity_scores, domain_id)
        classification_loss = F.cross_entropy(y_pred, label)  # cross-entropy loss
        loss = domain_loss + classification_loss
        self.log('train_loss', loss, on_epoch=True)

        acc = self.compute_accuracy(y_pred[:, 1], label)
        self.log('training_accuracy', acc, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input = val_batch["multimodal_emb"]
        label = val_batch["label"]
        domain_id = val_batch["domain_id"]
        domain_name = val_batch["domain_name"]   # list

        domain_similarity_scores, y_pred = self.forward(input)
        domain_loss = F.cross_entropy(domain_similarity_scores, domain_id)
        classification_loss = F.cross_entropy(y_pred, label)  # cross-entropy loss
        loss = domain_loss + classification_loss
        self.log('val_loss', loss, on_epoch=True)

        acc = self.compute_accuracy(y_pred[:, 1], label)
        self.log('validation_accuracy', acc, on_epoch=True)

        # {"climate": 0, "covid": 1, "military": 2}
        indice_climate = (torch.Tensor(domain_id) == 0).nonzero().squeeze()   # E.g. tensor([1, 2])
        indice_covid = (torch.Tensor(domain_id) == 1).nonzero().squeeze()
        indice_military = (torch.Tensor(domain_id) == 2).nonzero().squeeze()

        self.validation_num_total["climate"] += len(indice_climate)
        self.validation_num_total["covid"] += len(indice_covid)
        self.validation_num_total["military"] += len(indice_military)

        # print(y_pred[indice_climate, 1] >= self.threshold)
        # print(label[indice_climate])
        self.validation_num_correct["climate"] += sum((y_pred[indice_climate, 1] >= self.threshold)==(label[indice_climate]))
        self.validation_num_correct["covid"] += sum((y_pred[indice_covid, 1] >= self.threshold)==(label[indice_covid]))
        self.validation_num_correct["military"] += sum((y_pred[indice_military, 1] >= self.threshold)==(label[indice_military]))

        return loss

    def on_validation_epoch_end(self):
        print(f"acc@covid: {float(self.validation_num_correct['covid'] / self.validation_num_total['covid'])}")
        print(f"acc@climate: {float(self.validation_num_correct['climate'] / self.validation_num_total['climate'])}")
        print(f"acc@military: {float(self.validation_num_correct['military'] / self.validation_num_total['military'])}")


class TwoTasks(nn.Module):
    def __init__(self, cfg):   # cfg: alpha, in_dim
        super().__init__()
        self.classnames = ["climate", "covid", "military"]
        base_text_features = _get_base_text_features(self.classnames)
        self.simple_taskres_learner = SimpleTaskResLearner(cfg, base_text_features)
        self.linear_classifier = LinearClassifier(cfg.in_dim, cfg.out_dim)
        self.compute_accuracy = BinaryAccuracy(threshold=cfg.args.threshold)
        self.validation_num_correct = {"covid": 0, "climate": 0, "military": 0}
        self.validation_num_total = {"covid": 0, "climate": 0, "military": 0}
        self.threshold = cfg.args.threshold

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

