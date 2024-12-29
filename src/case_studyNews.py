import os
import sys

from itertools import count
from multiprocessing import Process
from model.conDA import ProjectionMLP, MLLMClassificationHead, ContrastiveLearningLossZModule

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
# from transformers import *
from itertools import cycle
from functools import reduce

from dataset.newsCLIPpingsDatasetConDATriplet import get_dataloader
from configs.configConDANews import ConfigConDANews

from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch.nn.functional as F
from utils.helper import compute_auc
from model.linearClassifier import LinearClassifier
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(int(1001))

DISTRIBUTED_FLAG = False


def return_classification(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:   # logits: [bs, 2], labels: [bs, ]
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:   # logits: [bs,]
        classification = (logits > 0).long().flatten()   # ?
    assert classification.shape == labels.shape
    return classification.cpu(), labels.cpu()


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:   # logits: [bs, 2], labels: [bs, ]
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:   # logits: [bs,]
        classification = (logits > 0).long().flatten()   # ?
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}')]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc) as loop, torch.no_grad():
        targets = []
        outputs = []
        domain_labels_list = []
        output_logits = []
        for example in loop:
            losses = []
            logit_votes = []
            # print(example)
            for data in example:
                # print(data)
                emb, labels, domain_labels = data["original_multimodal_emb"], data["original_label"], data["domain_label"]
                emb, labels = emb.to(device), labels.to(device)
                batch_size = emb.shape[0]

                ###### For the z instead of h input to the model ######
                z = model.mlp(emb)
                ###############

                # logits = model(emb)   # What is the model here? it's the mllm_cls_head
                logits = model.model(z)   # What is the model here? it's the entire ConDA, compatible with ContrastiveLearningAndTripletLossZModule
                # loss, softmax_logits = model.compute_loss(logits, labels=labels), model.compute_softmax_logits(logits)
                loss, softmax_logits = model.model.compute_loss(logits, labels=labels), model.model.compute_softmax_logits(logits)
                losses.append(loss)
                logit_votes.append(softmax_logits)

                classifications, labels = return_classification(logits, labels)
                targets.append(labels)
                outputs.append(classifications)

        
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        cls_report = classification_report(targets, outputs, digits=4, zero_division=0)
        print(cls_report)

    return outputs, targets


def test(net, iterator, device):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    net.eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():

        total_loss = 0
        num_correct = dict()
        num_total = dict()
        targets = []
        outputs = []
        domain_labels_list = []

        y_pred_list = []
        y_true_list = []
        for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
            inputs = batch["original_multimodal_emb"].to(device)
            labels = batch["original_label"].to(device)
            domain_labels = batch["domain_label"]
            domain_labels_list += list(domain_labels)
            inputs, labels = Variable(inputs), Variable(labels)

            # Get the output predictions
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

            # Compute total loss of the current epoch
            total_loss += loss.item()

            # Compute the number of correct predictions
            top_pred = torch.zeros_like(labels)
            y_preds = softmax(y_preds)
            top_pred[y_preds[:, 1] >= 0.5] = 1
            y = labels.cpu()
            cur_batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(cur_batch_size)

            y_pred_list.append(top_pred)  # [bs, 2]?
            y_true_list.append(y.cpu())  # [bs, 2]?

            # topic_labels = batch["topic"]

    return torch.cat(y_pred_list, dim=0).numpy(), torch.cat(y_true_list, dim=0).numpy()


if __name__ == "__main__":
    sys.argv = ["notebook", "--batch_size", "256", "--max_epochs", "1", "--target_domain", "bbc,guardian", "--base_model", "blip-2", "--loss_type", "simclr"]
    cfg = ConfigConDANews()
    batch_size = cfg.args.batch_size
    loss_type = cfg.args.loss_type
    max_epochs = cfg.args.max_epochs
    learning_rate = cfg.args.learning_rate
    weight_decay = 0
    lambda_w = cfg.args.lambda_w
    lambda_mmd = cfg.args.lambda_mmd
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    writer = None

    # (1) classification MLP
    mllm_cls_head = MLLMClassificationHead(cfg).to(device)

    # (2) projection MLP
    mlp = ProjectionMLP(cfg).to(device)

    # (3) the entire contrastive learning framework
    model = ContrastiveLearningLossZModule(model=mllm_cls_head, mlp=mlp, loss_type=loss_type, logger=writer, device=device,
                                      lambda_w=lambda_w, lambda_mmd=lambda_mmd)
    model.load_state_dict(torch.load('./saved_model/ConDANews_U+W.pt')["model_state_dict"])

    tgt_excluded_topic = ['bbc', 'guardian', 'usa_today']

    val_iterator, val_size = get_dataloader(cfg, target_domain=tgt_excluded_topic, shuffle=False, phase="test")

    outputs, targets = validate(model, device, val_iterator)

    print(outputs.shape)

    net = LinearClassifier(768)
    net.load_state_dict(torch.load('./saved_model/Blip2_U+W.pt'))
    net.to(device)

    outputs_base, targets_base = test(net, val_iterator, device)
    print(outputs_base.shape)

    # Find indices where elements are unequal
    unequal_indices = np.where((outputs != outputs_base) & (outputs == targets))[0]

    print("Indices of unequal elements:", unequal_indices[-100:])

