#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

from torch import nn
import pandas as pd
import os

from tqdm.auto import tqdm, trange

from sklearn.metrics import classification_report
import json

from torch.utils.data import Dataset
import torch.utils.data as data
from torch import optim
from torch.autograd import Variable
import numpy as np
from nltk.tokenize import TweetTokenizer
import re, string

import logging
import argparse

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


# Define the Dataset class
def load_tensor(filepath):
    tensor = torch.load(filepath)

    return tensor


def load_json(filepath):
    with open(filepath, 'r') as fp:
        json_data = json.load(fp)

    return json_data


class TwitterCOMMsDataset(Dataset):
    def __init__(self, csv_path, img_dir, multimodal_embeds_path, metadata_path):
        """
        Args:
            csv_path (string): Path to the {train_completed|val_completed}.csv file.
            img_dir (string): Directory containing the images
        """
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir

        self.df['exists'] = self.df['filename'].apply(lambda filename: os.path.exists(os.path.join(img_dir, filename)))
        delete_row = self.df[self.df["exists"] == False].index
        self.df = self.df.drop(delete_row)
        self.multimodal_embeds = load_tensor(multimodal_embeds_path)
        self.metadata = load_json(metadata_path)

        assert len(self.df) == self.multimodal_embeds.shape[0], "Number of news in self.df isn't equal to number of tensor"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        img_filename = item['filename']
        topic = item['topic']
        falsified = int(item['falsified'])
        not_falsified = float(not item['falsified'])
        label = np.array(falsified)
        domain = topic.split('_')[0]
        difficulty = topic.split('_')[1]

        image_path = os.path.join(self.img_dir, img_filename)

        assert image_path == self.metadata[str(idx)], "Image path does not match with the metadata"
        multimodal_emb = self.multimodal_embeds[idx]

        return {"multimodal_emb": multimodal_emb,
                "topic": topic,
                "label": label,
                "domain": domain,
                "difficulty": difficulty}


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Net(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(Net, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        out = self.fc(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def train(train_iterator, val_iterator, device):
    net = Net(768)   # blip-2 multimodal: 768, blip-2 unimodal: 512
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)

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
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            net.zero_grad()
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, top_pred = y_preds.topk(1, 1)
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)

            num_correct += sum(top_pred == y).item()
            num_total += batch_size

            if i % 1000 == 0:
                logger.info("Epoch [%d/%d] %d-th batch: Training accuracy: %.3f, loss: %.3f" % (
                epoch + 1, EPOCHS, i, num_correct / num_total, total_loss / num_total))

        logger.info("Epoch [%d/%d]: Training accuracy: %.3f, loss: %.3f" % (
        epoch + 1, EPOCHS, num_correct / num_total, total_loss / num_total))

        test(net, val_iterator, criterion, device)

    return net


def test(net, iterator, criterion, device):
    net.eval()

    with torch.no_grad():
        total_loss = 0
        num_correct = 0
        num_total = 0
        for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

            total_loss += loss.item()

            _, top_pred = y_preds.topk(1, 1)
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)

            num_correct += sum(top_pred == y).item()
            num_total += batch_size

            if i % 1000 == 0:
                logger.info("%d-th batch: Testing accuracy %.3f, loss: %.3f" % (
                i, num_correct / num_total, total_loss / num_total))

        logger.info("Testing accuracy %.3f, loss: %.3f" % (num_correct / num_total, total_loss / num_total))

    return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, required=True, help="batch size")
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, or albef}")
    p.add_argument("--epochs", type=int, required=True, help="number of training epochs")
    p.add_argument("--mode", type=str, required=True, help="{unimodal, multimodal}")

    args = p.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    BATCH_SIZE = args.bs
    base_model = args.base_model
    EPOCHS = args.epochs
    mode = args.mode

    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    # train_data = TwitterCOMMsDataset(csv_path='../raw_data/train_completed.csv',
    #                                  img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
    #                                  multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{base_model}_{mode}_embeds_train.pt',
    #                                  metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{base_model}_{mode}_idx_to_image_path_train.json')  # took ~one hour to construct the dataset
    train_data = TwitterCOMMsDataset(csv_path='../raw_data/val_completed.csv',
                                   img_dir=root_dir + 'twitter-comms/images/val_images/val_tweet_image_ids',
                                   multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{base_model}_{mode}_embeds_valid.pt',
                                   metadata_path=root_dir + f'twitter-comms/processed_data/metadata/{base_model}_{mode}_idx_to_image_path_valid.json'
                                   )
    logger.info(f"Found {train_data.__len__()} items in training data")

    logger.info("Loading valid data")
    val_data = TwitterCOMMsDataset(csv_path='../raw_data/val_completed.csv',
                                   img_dir=root_dir+'twitter-comms/images/val_images/val_tweet_image_ids',
                                   multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{base_model}_{mode}_embeds_valid.pt',
                                   metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{base_model}_{mode}_idx_to_image_path_valid.json'
                                   )
    logger.info(f"Found {val_data.__len__()} items in valid data")

    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE)
    val_iterator = data.DataLoader(val_data,
                                   shuffle=False,
                                   batch_size=BATCH_SIZE)

    logger.info("Start training the model")

    net = train(train_iterator, val_iterator, device)
