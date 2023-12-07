#!/usr/bin/env python
# coding: utf-8

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
from torch.nn import functional as F
from tqdm.auto import tqdm

from utils.helper import save_tensor, load_tensor, load_json
from dataset.twitter_comms_dataset import TwitterCOMMsDataset, get_dataloader
from model.twoTasks import TwoTasks
from configs import ConfigTwoTasks
from model.simpleTaskRes import _get_base_text_features
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


CUSTOM_TEMPLATES = {
    "Twitter-COMMs": "a piece of news in {}."   # {domain}
}

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

# Define the Dataset class
# Use import instead


# def train(train_iterator, val_iterator, device):
#     classnames = ["climate", "covid", "military"]
#     base_text_features = _get_base_text_features(classnames)
#     net = TwoTasks(alpha=alpha, in_dim=768, base_text_features=base_text_features)
#     net.cuda()
#     net.train()
#     net.weight_init(mean=0, std=0.02)
#     softmax = nn.Softmax(dim=1)
#
#     lr = 0.0001
#     optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
#
#     criterion = nn.CrossEntropyLoss()
#     criterion.to(device)
#
#     for epoch in range(EPOCHS):
#         net.train()
#         total_loss = 0
#         num_correct = 0
#         num_total = 0
#         for i, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
#             # Extract batch image and batch text inputs
#             inputs = batch["multimodal_emb"].to(device)
#             labels = batch["label"].to(device)
#             domain = batch["domain"].to(device)
#
#             inputs, labels = Variable(inputs), Variable(labels)
#             domain = Variable(domain)
#
#             # Get the output predictions
#             net.zero_grad()
#             domain_similarity_scores, y_preds = net(inputs)
#
#             # print(domain_similarity_scores.shape)
#             domain_loss = criterion(domain_similarity_scores, domain)
#             classification_loss = criterion(y_preds, labels)  # cross-entropy loss
#             loss = domain_loss + classification_loss
#
#             # Back-propagate and update the parameters
#             loss.backward()
#             optimizer.step()
#
#             # Compute total loss of the current epoch
#             total_loss += loss.item()
#
#             # Compute the number of correct predictions
#             # Implementation (1) Select the class with a higher predicted score, equiv. to threshold=0.5
#             # _, top_pred = y_preds.topk(1, 1)
#             # y = labels.cpu()
#             # batch_size = y.shape[0]
#             # top_pred = top_pred.cpu().view(batch_size)
#
#             # Implementation (2) If the predicted score (col=1) is higher than the threshold
#             top_pred = torch.zeros_like(labels)
#             y_preds = softmax(y_preds)
#             top_pred[y_preds[:, 1] >= threshold] = 1
#             y = labels.cpu()
#             batch_size = y.shape[0]
#             top_pred = top_pred.cpu().view(batch_size)
#
#             num_correct += sum(top_pred == y).item()
#             num_total += batch_size
#
#             if i % 1000 == 0:
#                 logger.info("Epoch [%d/%d] %d-th batch: training accuracy: %.3f, loss: %.6f" % (
#                 epoch + 1, EPOCHS, i, num_correct / num_total, total_loss / num_total))
#
#         logger.info("Epoch [%d/%d]: training accuracy: %.3f, loss: %.6f" % (
#         epoch + 1, EPOCHS, num_correct / num_total, total_loss / num_total))
#
#         test_pred, test_true = test(net, val_iterator, criterion, device)
#         assert test_pred.shape[0] == len(val_data), "test_pred.shape[0] is not equal to the length of val data"
#         assert test_true.shape[0] == len(val_data), "test_true.shape[0] is not equal to the length of val data"
#
#     return net
#
#
# def test(net, iterator, criterion, device):
#     net.eval()
#     softmax = nn.Softmax(dim=1)
#
#     with torch.no_grad():
#         total_loss = 0
#         num_correct = dict()
#         num_total = dict()
#         num_correct["all"] = 0
#         num_total["all"] = 0
#         num_correct["climate"] = 0
#         num_total["climate"] = 0
#         num_correct["covid"] = 0
#         num_total["covid"] = 0
#         num_correct["military"] = 0
#         num_total["military"] = 0
#
#         y_pred_list = []
#         y_true_list = []
#         for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
#             inputs = batch["multimodal_emb"].to(device)
#             labels = batch["label"].to(device)
#             domain = batch["domain"].to(device)
#             inputs, labels = Variable(inputs), Variable(labels)
#             domain = Variable(domain)
#
#             # Get the output predictions
#             domain_similarity_scores, y_preds = net(inputs)
#
#             domain_loss = criterion(domain_similarity_scores, domain)
#             classification_loss = criterion(y_preds, labels)
#             loss = domain_loss + classification_loss
#
#             # Compute total loss of the current epoch
#             total_loss += loss.item()
#
#             # Compute the number of correct predictions
#             top_pred = torch.zeros_like(labels)
#             y_preds = softmax(y_preds)
#             top_pred[y_preds[:, 1] >= threshold] = 1
#             y = labels.cpu()
#             cur_batch_size = y.shape[0]
#             top_pred = top_pred.cpu().view(cur_batch_size)
#
#             y_pred_list.append(y_preds)   # [bs, 2]?
#             y_true_list.append(y)   # [bs, 2]?
#
#             # Compute overall performance
#             num_correct["all"] += sum(top_pred == y).item()
#             num_total["all"] += cur_batch_size
#
#             # Compute topic-wise performance
#             topic_labels = batch["topic"]
#             topic_list = ["climate", "covid", "military"]
#
#             for topic in topic_list:
#                 inds = []
#                 for ind, topic_label in enumerate(topic_labels):
#                     if topic in topic_label:
#                         inds.append(ind)
#                 num_total[topic] += len(inds)
#                 inds = np.array(inds)
#                 num_correct[topic] += sum(top_pred[inds] == y[inds])
#
#             if i % 1000 == 0:
#                 logger.info("%d-th batch: Testing accuracy %.3f, loss: %.3f" % (
#                     i, num_correct["all"] / num_total["all"], total_loss / num_total["all"]))
#         logger.info("Overall testing accuracy %.3f, climate testing accuracy %.3f, covid testing accuracy %.3f, "
#                     "military testing accuracy %.3f, loss: %.6f" % (num_correct["all"] / num_total["all"],
#                                                                     num_correct["climate"] / num_total["climate"],
#                                                                     num_correct["covid"] / num_total["covid"],
#                                                                     num_correct["military"] / num_total["military"],
#                                                                     total_loss / num_total["all"]))
#
#     return torch.cat(y_pred_list, dim=0), torch.cat(y_true_list, dim=0)


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--bs", type=int, required=True, help="batch size")
#     p.add_argument("--epochs", type=int, required=True, help="number of training epochs")
#     p.add_argument("--few_shot_topic", type=str, required=False,
#                    help="topic that will not be included in the training")
#     p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
#     p.add_argument("--threshold", type=float, required=False, default=0.5,
#                    help="threshold value for making the class prediction")
#     p.add_argument("--alpha", type=float, required=False, default=0.5,
#                    help="weight assigned to the residual part")
#
#     args = p.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#
#     # Parse arguments
#     args = parse_args()
#     BATCH_SIZE = args.bs
#     EPOCHS = args.epochs
#     few_shot_topic = args.few_shot_topic
#     base_model = args.base_model
#     threshold = args.threshold
#     alpha = args.alpha
#     logger.info(f"base model: {base_model}")
#     logger.info(f"few shot topic: {few_shot_topic}")
#
#     # Set up device to use
#     device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#     logger.info(device)
#
#     root_dir = '/import/network-temp/yimengg/data/'
#
#     logger.info("Loading training data")
#     # train_data = TwitterCOMMsDataset(feather_path='../raw_data/train_completed_exist.feather',
#     #                                  img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
#     #                                  multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{base_model}_multimodal_embeds_train.pt',
#     #                                  metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{base_model}_idx_to_image_path_train.json',
#     #                                  few_shot_topic=[few_shot_topic]
#     #                                  )  # took ~one hour to construct the dataset
#     train_data = TwitterCOMMsDataset(feather_path='./raw_data/val_completed_exist.feather',
#                                      img_dir=root_dir + 'twitter-comms/images/val_images/val_tweet_image_ids',
#                                      multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{base_model}_multimodal_embeds_valid.pt',
#                                      metadata_path=root_dir + f'twitter-comms/processed_data/metadata/{base_model}_multimodal_idx_to_image_path_valid.json',
#                                      few_shot_topic=few_shot_topic
#                                    )   # small sample
#     logger.info(f"Found {train_data.__len__()} items in training data")
#
#     logger.info("Loading valid data")
#     val_data = TwitterCOMMsDataset(feather_path='./raw_data/val_completed_exist.feather',
#                                    img_dir=root_dir+'twitter-comms/images/val_images/val_tweet_image_ids',
#                                    multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{base_model}_multimodal_embeds_valid.pt',
#                                    metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{base_model}_multimodal_idx_to_image_path_valid.json'
#                                    )
#     logger.info(f"Found {val_data.__len__()} items in valid data")
#
#     train_iterator = data.DataLoader(train_data,
#                                      shuffle=True,
#                                      batch_size=BATCH_SIZE)
#     val_iterator = data.DataLoader(val_data,
#                                    shuffle=False,
#                                    batch_size=BATCH_SIZE)
#
#     logger.info("Start training the model")
#
#     net = train(train_iterator, val_iterator, device)


def main():
    cfg = ConfigTwoTasks()
    model = TwoTasks(cfg)
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=cfg.args.max_epochs,
                         logger=logger)
    logger.info("Loading training data")
    train_loader, train_length = get_dataloader(cfg, "train")
    logger.info(f"Found {train_length} items in training data")
    logger.info("Loading valid data")
    val_loader, val_length = get_dataloader(cfg, "val")
    logger.info(f"Found {val_length} items in valid data")

    trainer.fit(model, train_loader, val_loader)
