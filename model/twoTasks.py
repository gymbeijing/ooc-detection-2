from torch import nn
from model.simpleTaskRes import SimpleTaskResLearner, _get_base_text_features
from model.linearClassifier import LinearClassifier, normal_init
import pytorch_lightning as pl   # 2.1.2
from torch import optim
from torch.nn import functional as F
from torchmetrics.classification import BinaryAccuracy


class TwoTasks(pl.LightningModule):
    def __init__(self, cfg):   # cfg: alpha, in_dim
        super().__init__()
        self.classnames = ["climate", "covid", "military"]
        base_text_features = _get_base_text_features(self.classnames)
        self.simple_taskres_learner = SimpleTaskResLearner(cfg, base_text_features)
        self.linear_classifier = LinearClassifier(cfg.in_dim, cfg.out_dim)
        self.compute_accuracy = BinaryAccuracy()
        self.validation_num_correct = {"covid": 0, "climate": 0, "military": 0}
        self.validation_num_total = {"covid": 0, "climate": 0, "military": 0}

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
        domain = train_batch["domain"]   # 0, 1, 2

        domain_similarity_scores, y_pred = self.forward(input)
        domain_loss = F.cross_entropy(domain_similarity_scores, domain)
        classification_loss = F.cross_entropy(y_pred, label)  # cross-entropy loss
        loss = domain_loss + classification_loss
        self.log('train_loss', loss, on_epoch=True)

        acc = self.compute_accuracy(y_pred, label)
        self.log('training accuracy', acc, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input = val_batch["multimodal_emb"]
        label = val_batch["label"]
        domain = val_batch["domain"]
        topic = val_batch["topic"]

        domain_similarity_scores, y_pred = self.forward(input)
        domain_loss = F.cross_entropy(domain_similarity_scores, domain)
        classification_loss = F.cross_entropy(y_pred, label)  # cross-entropy loss
        loss = domain_loss + classification_loss
        self.log('val loss', loss)

        acc = self.compute_accuracy(y_pred, label)
        self.log('validation accuracy', acc, on_epoch=True)

        indice_covid = (topic == "covid").nonzero().squeeze()
        indice_climate = (topic == "climate").nonzero().squeeze()
        indice_military = (topic == "military").nonzero().squeeze()

        self.validation_num_total["covid"] += len(indice_covid)
        self.validation_num_total["climate"] += len(indice_climate)
        self.validation_num_total["military"] += len(indice_military)

        self.validation_num_correct["covid"] += len(y_pred[indice_covid].eq(label[indice_covid]))
        self.validation_num_correct["climate"] += len(y_pred[indice_climate].eq(label[indice_climate]))
        self.validation_num_correct["military"] += len(y_pred[indice_military].eq(label[indice_military]))

        return loss

    def on_validation_epoch_end(self):
        return {"acc @ covid": float(self.validation_num_correct["covid"] / self.validation_num_total["covid"]),
                "acc @ climate": float(self.validation_num_correct["climate"] / self.validation_num_total["climate"]),
                "acc @ military": float(self.validation_num_correct["military"] / self.validation_num_total["military"])}

