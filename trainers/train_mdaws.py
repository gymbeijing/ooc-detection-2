#!/usr/bin/env python
# coding: utf-8

"""
python -m trainers.train_mdaws --few_shot_topic military
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
from dataset.twitterCOMMsDataset import TwitterCOMMsDataset

from sklearn.metrics import f1_score
import numpy as np

import logging
import argparse
from model.mdaws import MDAWS
from configs.configMDAWS import ConfigMDAWS
from itertools import chain
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
        assert test_pred.shape[0] == len(val_data), "test_pred.shape[0] is not equal to the length of val data"
        assert test_true.shape[0] == len(val_data), "test_true.shape[0] is not equal to the length of val data"

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
                
        # for topic in topic_list:
        #     inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if topic in topic_fullname]
        #     f1[topic] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')
        inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if few_shot_topic in topic_fullname]
        f1[few_shot_topic] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')
        cls_report[few_shot_topic] = classification_report(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], digits=4, zero_division=0)
        print(f"classification report: {cls_report[few_shot_topic]}")
        auc_score[few_shot_topic] = compute_auc(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds])
        print(f"auc score: {auc_score[few_shot_topic]}")
        # print(num_total)
        # print(num_correct)

        logger.info(f"Overall testing accuracy %.4f, {few_shot_topic} testing accuracy %.4f, loss: %.4f" % (num_correct["all"] / num_total["all"],
                                                                    num_correct[few_shot_topic] / num_total[few_shot_topic],
                                                                    total_loss / num_total["all"]))
        print(f"f1: {f1}")

    return torch.cat(y_pred_list, dim=0), torch.cat(y_true_list, dim=0)



if __name__ == '__main__':

    cfg = ConfigMDAWS()
    BATCH_SIZE = cfg.args.batch_size
    EPOCHS = cfg.args.max_epochs
    threshold = cfg.args.threshold
    few_shot_topic = cfg.args.few_shot_topic


    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)
    topic_list = ['covid', 'climate', 'military']
    topic_list.remove(few_shot_topic)
    src_topic_list = topic_list

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    train_data = TwitterCOMMsDataset(feather_path='./raw_data/train_completed_exist.feather',
                                     img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
                                     multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_train.pt',
                                     metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_idx_to_image_path_train.json',
                                     few_shot_topic=few_shot_topic)  # took ~one hour to construct the dataset
    logger.info(f"Found {train_data.__len__()} items in training data")
    tgt_train_data = TwitterCOMMsDataset(feather_path='./raw_data/train_completed_exist.feather',
                                     img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
                                     multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_train.pt',
                                     metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_idx_to_image_path_train.json',
                                     few_shot_topic=src_topic_list)  # took ~one hour to construct the dataset
    sample_size = 50
    indices = torch.randperm(len(tgt_train_data)).tolist()[:sample_size]  # Randomly select indices
    sample_sampler = SubsetRandomSampler(indices)
    tgt_train_iterator = DataLoader(tgt_train_data, batch_size=sample_size, sampler=sample_sampler)


    logger.info("Loading valid data")
    val_data = TwitterCOMMsDataset(feather_path='./raw_data/val_completed_exist.feather',
                                   img_dir=root_dir+'twitter-comms/images/val_images/val_tweet_image_ids',
                                   multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_valid.pt',
                                   metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_multimodal_idx_to_image_path_valid.json',
                                   few_shot_topic=src_topic_list
                                   )
    logger.info(f"Found {val_data.__len__()} items in valid data")

    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE)
    val_iterator = data.DataLoader(val_data,
                                   shuffle=False,
                                   batch_size=BATCH_SIZE)

    logger.info("Start training the model")

    net = MDAWS(cfg.args)   # blip-2 multimodal: 768, blip-2 unimodal: 512
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)

    net = train(train_iterator, tgt_train_iterator, val_iterator, device)
