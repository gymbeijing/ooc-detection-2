#!/usr/bin/env python
# coding: utf-8

"""
python -m trainers.train_cadaNews --batch_size 256 --event_num 2 --max_epochs 10 --hidden_dim 768 --base_model blip-2 --threshold 0.5 --target_agency bbc,guardian
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.helper import save_tensor, load_tensor, load_json
from dataset.newsCLIPpingsDataset import get_dataloader_2
from model.linearClassifier import LinearClassifier

from sklearn.metrics import f1_score
import numpy as np

import logging
import argparse
from model.cada import CADA
from configs.configEANNNews import ConfigEANNNews
from sklearn.metrics import f1_score, classification_report
from utils.helper import accuracy_at_eer, compute_auc

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

softmax = nn.Softmax(dim=1)


def train(train_iterator, val_iterator, device):

    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        num_correct = 0
        num_total = 0
        for i, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
            # Extract batch image and batch text inputs
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            domain_labels = batch["domain_id"].to(device)
            inputs, labels, domain_labels = Variable(inputs), Variable(labels), Variable(domain_labels)

            # Get the output predictions
            net.zero_grad()
            y_preds, rumor_domain_preds, non_domain_preds = net(inputs, labels)
            # print(rumor_domain_preds.shape)
            # print(domain_labels[labels==True])
            loss = criterion(y_preds, labels) + criterion(rumor_domain_preds, domain_labels[labels==True]) + criterion(non_domain_preds, domain_labels[labels==False])

            # Back-propagate and update the parameters
            loss.backward()
            optimizer.step()

            # Compute total loss of the current epoch
            total_loss += loss.item()

            # Compute the number of correct predictions
            # Implementation (1) Select the class with a higher predicted score, equiv. to threshold=0.5
            # _, top_pred = y_preds.topk(1, 1)
            # y = labels.cpu()
            # batch_size = y.shape[0]
            # top_pred = top_pred.cpu().view(batch_size)

            # Implementation (2) If the predicted score (col=1) is higher than the threshold
            top_pred = torch.zeros_like(labels)
            y_preds = softmax(y_preds)
            top_pred[y_preds[:, 1] >= threshold] = 1
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)

            num_correct += sum(top_pred == y).item()
            num_total += batch_size

            if i % 1000 == 0:
                logger.info("Epoch [%d/%d] %d-th batch: training accuracy: %.3f, loss: %.3f" % (
                    epoch + 1, EPOCHS, i, num_correct / num_total, total_loss / num_total))

        logger.info("Epoch [%d/%d]: training accuracy: %.3f, loss: %.3f" % (
            epoch + 1, EPOCHS, num_correct / num_total, total_loss / num_total))

        test_pred, test_true = test(net, val_iterator, criterion, device)
        assert test_pred.shape[0] == test_len, "test_pred.shape[0] is not equal to the length of val data"
        assert test_true.shape[0] == test_len, "test_true.shape[0] is not equal to the length of val data"

    return net


def test(net, iterator, criterion, device):
    net.eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        total_loss = 0
        num_correct = dict()
        num_total = dict()
        f1 = dict()
        cls_report = dict()
        auc_score = dict()
        num_correct["all"] = 0
        num_total["all"] = 0
        num_correct["bbc"] = 0
        num_total["bbc"] = 0
        num_correct["guardian"] = 0
        num_total["guardian"] = 0
        num_correct["usa_today"] = 0
        num_total["usa_today"] = 0
        num_correct["washington_post"] = 0
        num_total["washington_post"] = 0

        y_pred_list = []
        y_true_list = []
        f1["bbc"] = 0
        f1["guardian"] = 0
        f1["usa_today"] = 0
        f1["washington_post"] = 0
        topic_label_list = []
        topic_list = ["bbc", "guardian", "usa_today", "washington_post"]
        for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            domain_labels = batch["domain_id"].to(device)
            inputs, labels, domain_labels = Variable(inputs), Variable(labels), Variable(domain_labels)

            # Get the output predictions
            y_preds, rumor_domain_preds, non_domain_preds = net(inputs, labels)
            loss = criterion(y_preds, labels) + criterion(rumor_domain_preds, domain_labels[labels==True]) + criterion(non_domain_preds, domain_labels[labels==False])

            # Compute total loss of the current epoch
            total_loss += loss.item()

            # Compute the number of correct predictions
            top_pred = torch.zeros_like(labels)
            y_preds = softmax(y_preds)
            top_pred[y_preds[:, 1] >= threshold] = 1
            y = labels.cpu()
            cur_batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(cur_batch_size)

            y_pred_list.append(top_pred)  # [bs, 2]?
            y_true_list.append(y.cpu())  # [bs, 2]?

            # Compute overall performance
            num_correct["all"] += sum(top_pred == y).item()
            num_total["all"] += cur_batch_size

            # Compute topic-wise performance
            topic_labels = batch["news_source"]
            topic_label_list += topic_labels

            for topic in topic_list:
                inds = []
                for ind, topic_label in enumerate(topic_labels):
                    if topic in topic_label:
                        inds.append(ind)
                num_total[topic] += len(inds)
                inds = np.array(inds)
                num_correct[topic] += sum(top_pred[inds] == y[inds])

            if i % 1000 == 0:
                logger.info("%d-th batch: Testing accuracy %.4f, loss: %.4f" % (
                    i, num_correct["all"] / num_total["all"], total_loss / num_total["all"]))
                
        for topic in topic_list:
            print(topic)
            inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if topic in topic_fullname]
            f1[topic] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')
            print(f"f1: {f1[topic]}")
            cls_report[topic] = classification_report(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], digits=4, zero_division=0)
            print(f"classification report: {cls_report[topic]}")
            auc_score[topic] = compute_auc(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds])
            print(f"auc score: {auc_score[topic]}")

        logger.info("Overall testing accuracy %.4f, bbc testing accuracy %.4f, guardian testing accuracy %.4f, "
                    "usa_today testing accuracy %.4f, washington_post testing accuracy %.4f, loss: %.4f" % (num_correct["all"] / num_total["all"],
                                                                    num_correct["bbc"] / num_total["bbc"],
                                                                    num_correct["guardian"] / num_total["guardian"],
                                                                    num_correct["usa_today"] / num_total["usa_today"],
                                                                    num_correct["washington_post"] / num_total["washington_post"],
                                                                    total_loss / num_total["all"]))
        print(f"f1: {f1}")

    return torch.cat(y_pred_list, dim=0), torch.cat(y_true_list, dim=0)



if __name__ == '__main__':

    cfg = ConfigEANNNews()
    BATCH_SIZE = cfg.args.batch_size
    EPOCHS = cfg.args.max_epochs
    threshold = cfg.args.threshold
    target_agency = cfg.args.target_agency


    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    train_iterator, train_len = get_dataloader_2(target_agency=target_agency, shuffle=True, batch_size=BATCH_SIZE, phase='train')
    # train_iterator, train_len = get_dataloader_2(target_agency=target_domain, shuffle=True, batch_size=BATCH_SIZE, phase='train')
    logger.info(f"Found {train_len} items in the training dataset")

    logger.info("Loading testing data")
    test_iterator, test_len = get_dataloader_2(target_agency=target_agency, shuffle=False, batch_size=BATCH_SIZE, phase='test')
    logger.info(f"Found {test_len} items in valid data")

    logger.info("Start training the model")

    net = CADA(cfg.args)   # blip-2 multimodal: 768, blip-2 unimodal: 512
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)

    net = train(train_iterator, test_iterator, device)
