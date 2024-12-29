import os
import sys

from itertools import count
from multiprocessing import Process
from model.conDA import ProjectionMLP, MLLMClassificationHead, ContrastiveLearningAndTripletLossZModule

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

from dataset.twitterCOMMsDatasetConDATriplet import get_dataloader
from configs.configConDA import ConfigConDA

from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch.nn.functional as F
from utils.helper import compute_auc
from model.linearClassifier import LinearClassifier
from torch.autograd import Variable

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

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    validation_f1 = 0

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}')]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc) as loop, torch.no_grad():
        targets = []
        outputs = []
        for example in loop:
            losses = []
            logit_votes = []
            # print(example)
            for data in example:
                emb, labels = data["original_multimodal_emb"], data["original_label"]
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

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size
            
            classifications, labels = return_classification(logits, labels)
            targets.append(labels)
            outputs.append(classifications)

            loop.set_postfix(loss=loss.item(), acc="{:.4f}".format(validation_accuracy / validation_epoch_size))
        
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        auc_score = compute_auc(targets, outputs)
        print(f"AUC score: {auc_score}")
        validation_f1 = f1_score(targets, outputs, average='macro')
        print(f"f1: {validation_f1}")
        cls_report = classification_report(targets, outputs, digits=4, zero_division=0)
        print(cls_report)

    return outputs, targets


def test(net, iterator, device):
    net.eval()
    softmax = nn.Softmax(dim=1)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    with torch.no_grad():
        total_loss = 0
        num_correct = dict()
        num_total = dict()
        f1 = dict()
        cls_report = dict()
        auc_score = dict()
        num_correct["all"] = 0
        num_total["all"] = 0
        num_correct["climate"] = 0
        num_total["climate"] = 0
        num_correct["covid"] = 0
        num_total["covid"] = 0
        num_correct["military"] = 0
        num_total["military"] = 0

        y_pred_list = []
        y_true_list = []
        f1["climate"] = 0
        f1["covid"] = 0
        f1["military"] = 0
        topic_label_list = []
        for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
            inputs = batch["original_multimodal_emb"].to(device)
            labels = batch["original_label"].to(device)
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

            # Compute overall performance
            num_correct["all"] += sum(top_pred == y).item()
            num_total["all"] += cur_batch_size

            # Compute topic-wise performance
            topic_labels = batch["topic"]
            topic_label_list += topic_labels
            topic_list = ["climate", "covid", "military"]

        cls_report = classification_report(torch.cat(y_true_list, dim=0), torch.cat(y_pred_list, dim=0), digits=4, zero_division=0)
        print(cls_report)

    return torch.cat(y_pred_list, dim=0).numpy(), torch.cat(y_true_list, dim=0).numpy()


if __name__ == "__main__":
    sys.argv = ["notebook", "--batch_size", "256", "--max_epochs", "1", "--tgt_topic", "covid", "--base_model", "blip-2", "--loss_type", "simclr"]
    cfg = ConfigConDA()
    tgt_validation_loader, tgt_validation_dataset_size = get_dataloader(cfg, few_shot_topic=[], shuffle=False, phase="val")
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # (1) classification MLP
    mllm_cls_head = MLLMClassificationHead(cfg).to(device)

    # (2) projection MLP
    mlp = ProjectionMLP(cfg).to(device)

    # (3) the entire contrastive learning framework
    model = ContrastiveLearningAndTripletLossZModule(model=mllm_cls_head, mlp=mlp, loss_type="simclr", logger=None, device=device,
                                            lambda_w=0.5, lambda_mmd=1.0)
    model.load_state_dict(torch.load('./saved_model/ConDA_Cv.pt')["model_state_dict"])

    tgt_excluded_topic = ['climate', 'covid', 'military']
    tgt_excluded_topic.remove(cfg.args.tgt_topic)

    val_iterator, val_size = get_dataloader(cfg, few_shot_topic=tgt_excluded_topic, shuffle=False, phase="val")

    outputs, targets = validate(model, device, val_iterator)

    print(outputs.shape)

    net = LinearClassifier(768)
    net.load_state_dict(torch.load('./saved_model/Blip2_Cv.pt'))
    net.to(device)

    outputs_base, targets_base = test(net, val_iterator, device)
    print(outputs_base.shape)

    # Find indices where elements are unequal
    unequal_indices = np.where((outputs != outputs_base) & (outputs == targets))[0]

    print("Indices of unequal elements:", unequal_indices[:50])

