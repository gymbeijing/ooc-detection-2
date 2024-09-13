#!/usr/bin/env python
# coding: utf-8

"""
python -m trainers.train_eann --batch_size 256 --event_num 4 --max_epochs 10 --hidden_dim 768 --base_model blip-2 --threshold 0.3 --few_shot_topic military
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
from dataset.twitterCOMMsDataset import TwitterCOMMsDataset
from model.linearClassifier import LinearClassifier

from sklearn.metrics import f1_score
import numpy as np

import logging
import argparse
from model.eann import EANN
from configs.configEANN import ConfigEANN
from sklearn.metrics import f1_score, classification_report
from utils.helper import accuracy_at_eer, compute_auc

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

softmax = nn.Softmax(dim=1)

# def train(train_iterator, val_iterator, device):

#     lr = 0.0001
#     optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

#     criterion = nn.CrossEntropyLoss()
#     criterion.to(device)

#     for epoch in range(EPOCHS):
#         net.train()
#         total_loss = 0
#         num_correct = 0
#         num_total = 0
#         for i, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
#             text_inputs = batch["text_emb"].to(device)
#             image_inputs = batch["image_emb"].to(device)
#             labels = batch["label"].to(device)
#             domain_labels = batch["domain_id"].to(device)
#             text_inputs, image_inputs, labels, domain_labels = Variable(text_inputs), Variable(image_inputs), Variable(labels), Variable(domain_labels)

#             net.zero_grad()
#             y_preds, domain_preds = net(text_inputs, image_inputs)
#             loss = criterion(y_preds, labels) + criterion(domain_preds, domain_labels)

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             _, top_pred = y_preds.topk(1, 1)
#             y = labels.cpu()
#             batch_size = y.shape[0]
#             top_pred = top_pred.cpu().view(batch_size)

#             num_correct += sum(top_pred == y).item()
#             num_total += batch_size

#             if i % 1000 == 0:
#                 logger.info("Epoch [%d/%d] %d-th batch: Training accuracy: %.3f, loss: %.3f" % (
#                 epoch + 1, EPOCHS, i, num_correct / num_total, total_loss / num_total))

#         logger.info("Epoch [%d/%d]: Training accuracy: %.3f, loss: %.3f" % (
#         epoch + 1, EPOCHS, num_correct / num_total, total_loss / num_total))

#         test(net, val_iterator, criterion, device)

#     return net


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
            y_preds, domain_preds = net(inputs)
            loss = criterion(y_preds, labels) + criterion(domain_preds, domain_labels)

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
        assert test_pred.shape[0] == len(val_data), "test_pred.shape[0] is not equal to the length of val data"
        assert test_true.shape[0] == len(val_data), "test_true.shape[0] is not equal to the length of val data"

    return net


# def test(net, iterator, criterion, device):
#     net.eval()

#     with torch.no_grad():
#         total_loss = 0
#         num_correct = 0
#         num_total = 0
#         for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
#             text_inputs = batch["text_emb"].to(device)
#             image_inputs = batch["image_emb"].to(device)
#             labels = batch["label"].to(device)
#             domain_labels = batch["domain_id"].to(device)
#             text_inputs, image_inputs, labels, domain_labels = Variable(text_inputs), Variable(image_inputs), Variable(labels), Variable(domain_labels)

#             y_preds, domain_preds = net(text_inputs, image_inputs)
#             loss = criterion(y_preds, labels) + criterion(domain_preds, domain_labels)

#             total_loss += loss.item()

#             _, top_pred = y_preds.topk(1, 1)
#             y = labels.cpu()
#             batch_size = y.shape[0]
#             top_pred = top_pred.cpu().view(batch_size)

#             num_correct += sum(top_pred == y).item()
#             num_total += batch_size

#             if i % 1000 == 0:
#                 logger.info("%d-th batch: Testing accuracy %.3f, loss: %.3f" % (
#                 i, num_correct / num_total, total_loss / num_total))

#         logger.info("Testing accuracy %.3f, loss: %.3f" % (num_correct / num_total, total_loss / num_total))

#     return


def test(net, iterator, criterion, device):
    net.eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        total_loss = 0
        num_correct = dict()
        num_total = dict()
        cls_report = dict()
        auc_score = dict()
        f1 = dict()
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
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            domain_labels = batch["domain_id"].to(device)
            inputs, labels, domain_labels = Variable(inputs), Variable(labels), Variable(domain_labels)

            # Get the output predictions
            y_preds, domain_preds = net(inputs)
            loss = criterion(y_preds, labels) + criterion(domain_preds, domain_labels)

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
            topic_labels = batch["topic"]
            topic_label_list += topic_labels
            topic_list = ["climate", "covid", "military"]

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

        logger.info("Overall testing accuracy %.4f, climate testing accuracy %.4f, covid testing accuracy %.4f, "
                    "military testing accuracy %.4f, loss: %.4f" % (num_correct["all"] / num_total["all"],
                                                                    num_correct["climate"] / num_total["climate"],
                                                                    num_correct["covid"] / num_total["covid"],
                                                                    num_correct["military"] / num_total["military"],
                                                                    total_loss / num_total["all"]))
        print(f"f1: {f1}")

    return torch.cat(y_pred_list, dim=0), torch.cat(y_true_list, dim=0)



if __name__ == '__main__':

    cfg = ConfigEANN()
    BATCH_SIZE = cfg.args.batch_size
    EPOCHS = cfg.args.max_epochs
    threshold = cfg.args.threshold
    few_shot_topic = cfg.args.few_shot_topic


    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    train_data = TwitterCOMMsDataset(feather_path='./raw_data/train_completed_exist.feather',
                                     img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
                                     multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_train.pt',
                                     metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_idx_to_image_path_train.json',
                                     few_shot_topic=few_shot_topic)  # took ~one hour to construct the dataset
    logger.info(f"Found {train_data.__len__()} items in training data")

    logger.info("Loading valid data")
    val_data = TwitterCOMMsDataset(feather_path='./raw_data/val_completed_exist.feather',
                                   img_dir=root_dir+'twitter-comms/images/val_images/val_tweet_image_ids',
                                   multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_valid.pt',
                                   metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_multimodal_idx_to_image_path_valid.json',
                                   )
    logger.info(f"Found {val_data.__len__()} items in valid data")

    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE)
    val_iterator = data.DataLoader(val_data,
                                   shuffle=False,
                                   batch_size=BATCH_SIZE)

    logger.info("Start training the model")

    net = EANN(cfg.args)   # blip-2 multimodal: 768, blip-2 unimodal: 512
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)

    net = train(train_iterator, val_iterator, device)
