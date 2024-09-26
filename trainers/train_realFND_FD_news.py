#!/usr/bin/env python
# coding: utf-8

"""
python -m trainers.train_realFND_FD_news --batch_size 256 --event_num 4 --max_epochs 5 --hidden_dim 768 --base_model blip-2 --threshold 0.5 --few_shot_topic bbc,guardian
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
from tqdm.auto import tqdm

from sklearn.metrics import f1_score
import numpy as np

import logging
import argparse
from model.realFND import FakeNewsClassifier, DomainClassifier
from configs.configRealFND import ConfigRealFND
from dataset.newsCLIPpingsDataset import get_dataloader_2

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

softmax = nn.Softmax(dim=1)


def train(name, net, train_iterator, val_iterator, device):

    lr = 1e-4
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for epoch in range(cfg.args.max_epochs):
        net.train()
        total_loss = 0
        num_correct = 0
        num_total = 0
        for i, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
            # Extract batch image and batch text inputs
            inputs = batch["multimodal_emb"].to(device)
            if name == "f":
                labels = batch["label"].to(device)
            else:   # name == "d"
                labels = batch["domain_id"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            # Get the output predictions
            net.zero_grad()
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

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
            top_pred[y_preds[:, 1] >= cfg.args.threshold] = 1
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)

            num_correct += sum(top_pred == y).item()
            num_total += batch_size

            if i % 1000 == 0:
                logger.info("Epoch [%d/%d] %d-th batch: training accuracy: %.3f, loss: %.3f" % (
                    epoch + 1, cfg.args.max_epochs, i, num_correct / num_total, total_loss / num_total))

        logger.info("Epoch [%d/%d]: training accuracy: %.3f, loss: %.3f" % (
            epoch + 1, cfg.args.max_epochs, num_correct / num_total, total_loss / num_total))

        test_pred, test_true = test(name, net, val_iterator, criterion, device)
        assert test_pred.shape[0] == test_len, "test_pred.shape[0] is not equal to the length of val data"
        assert test_true.shape[0] == test_len, "test_true.shape[0] is not equal to the length of val data"

    return net


def test(name, net, iterator, criterion, device):
    net.eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        total_loss = 0
        num_correct = dict()
        num_total = dict()
        f1 = dict()
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
            if name == "f":
                labels = batch["label"].to(device)
            else:   # name == "d"
                labels = batch["domain_id"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            # Get the output predictions
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

            # Compute total loss of the current epoch
            total_loss += loss.item()

            # Compute the number of correct predictions
            # top_pred = torch.zeros_like(labels)
            # y_preds = softmax(y_preds)
            # top_pred[y_preds[:, 1] >= threshold] = 1
            y = labels.cpu()
            cur_batch_size = y.shape[0]
            # top_pred = top_pred.cpu().view(cur_batch_size)
            
            top_pred = y_preds.argmax(dim=1).cpu()

            y_pred_list.append(top_pred.cpu())  # [bs, 2]?
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
            inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if topic in topic_fullname]
            f1[topic] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')

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

    cfg = ConfigRealFND()
    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    train_dataloader, train_len = get_dataloader_2(target_agency=cfg.args.few_shot_topic, shuffle=True, batch_size=cfg.args.batch_size, phase='train')
    logger.info(f"Found {train_len} items in training data")

    logger.info("Loading valid data")
    val_dataloader, test_len = get_dataloader_2(target_agency=cfg.args.few_shot_topic, shuffle=False, batch_size=cfg.args.batch_size, phase='test')
    logger.info(f"Found {test_len} items in valid data")

    logger.info("Start training the model")

    fake_news_classifier = FakeNewsClassifier()
    fake_news_classifier.cuda()
    fake_news_classifier.train()
    fake_news_classifier.weight_init(mean=0, std=0.02)

    domain_classifier = DomainClassifier(cfg.args.event_num)
    domain_classifier.cuda()
    domain_classifier.train()
    domain_classifier.weight_init(mean=0, std=0.02)

    fake_news_classifier = train("f", fake_news_classifier, train_dataloader, val_dataloader, device)
    torch.save(fake_news_classifier.state_dict(), os.path.join('real_fnd_output', f'fake_news_classifier_{cfg.args.few_shot_topic}.ckpt'))
    domain_classifier = train("d", domain_classifier, val_dataloader, val_dataloader, device)
    torch.save(domain_classifier.state_dict(), os.path.join('real_fnd_output', f'domain_classifier_{cfg.args.few_shot_topic}.ckpt'))