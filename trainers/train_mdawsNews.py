#!/usr/bin/env python
# coding: utf-8

"""
python -m trainers.train_mdawsNews --target_agency bbc
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
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from tqdm.auto import tqdm

from utils.helper import save_tensor, load_tensor, load_json

from sklearn.metrics import f1_score
import numpy as np

import logging
import argparse
from model.mdaws import MDAWS
from configs.configMDAWS import ConfigMDAWS
from itertools import chain
from dataset.newsCLIPpingsDataset import get_dataloader_2
from sklearn.metrics import f1_score, classification_report
from utils.helper import accuracy_at_eer, compute_auc

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

softmax = nn.Softmax(dim=1)


def train(train_iterator, tgt_train_iterator, val_iterator, device):

    lr = 1e-3   # 1e-4?
    main_optimizer = \
            torch.optim.Adam(filter(lambda p: p.requires_grad, chain(net.domain_adv.parameters(), net.classifiers.parameters())), lr=1e-3)
    group_optimizer = torch.optim.Adam(
            [{"params": filter(lambda p: p.requires_grad, chain(net.domain_adv.parameters(), net.classifiers.parameters())),
              "lr":1e-3, "weight_decay": 0.01
              },
             {"params": filter(lambda p: p.requires_grad, net.group_weight.parameters())}
             ],

            lr=1e-3
        )

    # cross_entropy_loss = nn.CrossEntropyLoss()
    # nll_loss = nn.NLLLoss()
    # cross_entropy_loss.to(device)
    # nll_loss.to(device)

    for batch in tqdm(tgt_train_iterator, desc="iterations"):
        tgt_train_batch = batch

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

            tgt_inputs = tgt_train_batch["multimodal_emb"].to(device)
            tgt_labels = tgt_train_batch["label"].to(device)
            tgt_domain_labels = tgt_train_batch["domain_id"].to(device)
            tgt_inputs, tgt_labels, tgt_domain_labels = Variable(tgt_inputs), Variable(tgt_labels), Variable(tgt_domain_labels)

            # Get the output predictions
            net.zero_grad()
            if epoch < cfg.args.pre_train_epochs:
                loss, outputs = net.first_stage(src_feature=inputs, src_domain_y=domain_labels, src_rumor_y=labels, tgt_feature=tgt_inputs, tgt_domain_y=tgt_domain_labels)
                y_preds = outputs[1]

                # Back-propagate and update the parameters
                loss.backward()
                main_optimizer.step()   # stage 1
            else:
                loss, src_logits, mix_logits, expert_weight = net.second_stage(src_feature=inputs, src_domain_y=domain_labels, src_rumor_y=labels, 
                                                                  tgt_feature=tgt_inputs, tgt_domain_y=tgt_domain_labels, tgt_rumor_y=tgt_labels)
                y_preds = src_logits

                # Back-propagate and update the parameters
                loss.backward()
                group_optimizer.step()   # stage 2

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

        test_pred, test_true = test(net, val_iterator, device)
        assert test_pred.shape[0] == test_len, "test_pred.shape[0] is not equal to the length of val data"
        assert test_true.shape[0] == test_len, "test_true.shape[0] is not equal to the length of val data"

    return net


def test(net, iterator, device):
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
            loss, src_logits, mix_logits, expert_weight = net.second_stage(src_feature=inputs, src_domain_y=domain_labels, src_rumor_y=labels, tgt_feature=inputs, tgt_domain_y=domain_labels, tgt_rumor_y=labels)
            y_preds = mix_logits

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
                
        # for topic in topic_list:
        #     inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if topic in topic_fullname]
        #     f1[topic] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')
        inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if target_agency in topic_fullname]
        f1[target_agency] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')
        cls_report[target_agency] = classification_report(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], digits=4, zero_division=0)
        auc_score[target_agency] = compute_auc(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds])
        print(f"classification report: {cls_report[target_agency]}")
        print(f"auc score: {auc_score[target_agency]}")
        # print(num_total)
        # print(num_correct)

        logger.info(f"Overall testing accuracy %.4f, {target_agency} testing accuracy %.4f, loss: %.4f" % (num_correct["all"] / num_total["all"],
                                                                    num_correct[target_agency] / num_total[target_agency],
                                                                    total_loss / num_total["all"]))
        print(f"f1: {f1}")

    return torch.cat(y_pred_list, dim=0), torch.cat(y_true_list, dim=0)



if __name__ == '__main__':

    cfg = ConfigMDAWS()
    BATCH_SIZE = cfg.args.batch_size
    EPOCHS = cfg.args.max_epochs
    threshold = cfg.args.threshold
    target_agency = cfg.args.target_agency


    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)
    source_list = ['bbc', 'guardian', 'usa_today', 'washington_post']
    target_list = [target_agency]
    source_list.remove(target_agency)
    if target_agency == 'bbc':
        source_list.remove("guardian")
        target_list.append("guardian")
    if target_agency == 'guardian':
        source_list.remove("bbc")
        target_list.append("bbc")
    if target_agency == 'usa_today':
        source_list.remove("washington_post")
        target_list.append("washington_post")
    if target_agency == 'washington_post':
        source_list.remove("usa_today")
        target_list.append("usa_today")
    src_agency = ",".join(source_list)
    tgt_agency = ",".join(target_list)

    logger.info(f"train source agency: {src_agency}, train target agency: {tgt_agency}")

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    _, train_iterator, train_len = get_dataloader_2(target_agency=tgt_agency, shuffle=True, batch_size=BATCH_SIZE, phase='train')
    logger.info(f"Found {train_len} items in training data")
    tgt_train_data, _, _ = get_dataloader_2(target_agency=src_agency, shuffle=True, batch_size=BATCH_SIZE, phase='train')
    sample_size = 50
    indices = torch.randperm(len(tgt_train_data)).tolist()[:sample_size]  # Randomly select indices
    sample_sampler = SubsetRandomSampler(indices)
    tgt_train_iterator = DataLoader(tgt_train_data, batch_size=sample_size, sampler=sample_sampler)


    logger.info("Loading valid data")
    target_list = ['bbc', 'guardian', 'usa_today', 'washington_post']
    target_list.remove(target_agency)
    tgt_agency = ",".join(target_list)
    logger.info(f"test target agency: {tgt_agency}")
    test_iterator, test_len = get_dataloader_2(target_agency=tgt_agency, shuffle=False, batch_size=BATCH_SIZE, phase='test')
    logger.info(f"Found {test_len} items in valid data")

    logger.info("Start training the model")

    net = MDAWS(cfg.args)   # blip-2 multimodal: 768, blip-2 unimodal: 512
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)

    net = train(train_iterator, tgt_train_iterator, test_iterator, device)
