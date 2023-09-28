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
class TwitterCOMMsDataset(Dataset):
    def __init__(self, csv_path, img_dir, few_shot_topic=[]):
        """
        Args:
            csv_path (string): Path to the {train_completed|val_completed}.csv file.
            image_folder_dir (string): Directory containing the images
        """
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        
        self.df['exists'] = self.df['filename'].apply(lambda filename: os.path.exists(os.path.join(img_dir, filename)))
        delete_row = self.df[self.df["exists"]==False].index
        self.df = self.df.drop(delete_row)

        # Remove news of the few shot topic
        if 'military' in few_shot_topic:
            self.df['is_military'] = self.df['topic'].apply(lambda topic: 'military' in topic)
            delete_row = self.df[self.df["is_military"]==True].index
            self.df = self.df.drop(delete_row)

        if 'covid' in few_shot_topic:
            self.df['is_covid'] = self.df['topic'].apply(lambda topic: 'covid' in topic)
            delete_row = self.df[self.df["is_covid"]==True].index
            self.df = self.df.drop(delete_row)

        if 'climate' in few_shot_topic:
            self.df['is_climate'] = self.df['topic'].apply(lambda topic: 'climate' in topic)
            delete_row = self.df[self.df["is_climate"]==True].index
            self.df = self.df.drop(delete_row)
    
    def __len__(self):
        return len(self.df)
    
    def remove_URL(self, text):
        """Remove URLs from a sample string"""
        return re.sub(r"http\S+", '', text)
    
    def remove_punc(self, text):
        """Remove punctuation from a sample string"""
        return re.sub(r'[^\w\s]', '', text)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        caption = item['full_text']   # original caption
        caption = ' '.join(tt.tokenize(caption))   # tokenized caption
        caption = self.remove_punc(self.remove_URL(caption))   # remove url & punctuation from the tokenized caption
        
        img_filename = item['filename']
        topic = item['topic']   # e.g. military_hard
        falsified = int(item['falsified'])   # falsified: 1, not falsified: 0
        not_falsified = float(not item['falsified'])
        label = np.array(falsified)
        domain = topic.split('_')[0]
        difficulty = topic.split('_')[1]
        
        raw_image = Image.open(os.path.join(self.img_dir, img_filename)).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)
        # sample = {"image": image, "text_input": [text_input]}   # image shape: [1, 3, 224, 224]

        # features_multimodal = model.extract_features(sample, mode="multimodal")   # [1, 32, 768] ??? image and text might mismatch
        # multimodal_emb = features_multimodal.multimodal_embeds[:, 0, :]   # [1, 768]

        return {"image": image,
                "text_input": text_input,
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


def get_multimodal_emb(batch_image, batch_text_input):
    batch_image = batch_image.squeeze(dim=1)
    samples = {"image": batch_image, "text_input": list(batch_text_input)}
    features_multimodal = model.extract_features(samples, mode="multimodal")
    multimodal_embeds = features_multimodal.multimodal_embeds[:, 0, :]  # [1, 768]

    return multimodal_embeds


def train(train_iterator, val_iterator, device):

    net = Net(768)
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)
    
    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
 
    EPOCHS = 2
    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        num_correct = 0
        num_total = 0
        for i, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
            # Extract batch image and batch text inputs
            batch_image = batch["image"]   # (bs, 1, 3, 224, 224) -> (bs, 3, 224, 224)
            batch_text_input = batch["text_input"]
            labels = batch["label"].to(device)

            # Get multimodal embedding as inputs
            inputs = get_multimodal_emb(batch_image, batch_text_input).to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            # Get the output predictions
            net.zero_grad()
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)   # cross-entropy loss

            # Back-propagate and update the parameters
            loss.backward()
            optimizer.step()

            # Compute total loss of the current epoch
            total_loss += loss.item()

            # Compute the number of correct predictions
            _, top_pred = y_preds.topk(1, 1)
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)

            num_correct += sum(top_pred == y).item()
            num_total += batch_size
            
            if i % 100 == 0:
                logger.info("Epoch [%d/%d] %d-th batch: training accuracy: %.3f, loss: %.3f" % (epoch+1, EPOCHS, i, num_correct/num_total, total_loss/num_total))
            
        logger.info("Epoch [%d/%d]: training accuracy: %.3f, loss: %.3f" % (epoch+1, EPOCHS, num_correct/num_total, total_loss/num_total))
                
        test(net, val_iterator, criterion, device)

    return net


def test(net, iterator, criterion, device):
    
    net.eval()
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        total_loss = 0
        num_correct = dict()
        num_total = dict()
        num_correct["all"] = 0
        num_total["all"] = 0
        num_correct["climate"] = 0
        num_total["climate"] = 0
        num_correct["covid"] = 0
        num_total["covid"] = 0
        num_correct["military"] = 0
        num_total["military"] = 0

        for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
            batch_image = batch["image"]
            batch_text_input = batch["text_input"]
            labels = batch["label"].to(device)

            # Get multimodal embedding as inputs
            inputs = get_multimodal_emb(batch_image, batch_text_input).to(device)
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
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)

            # Compute overall performance
            num_correct["all"] += sum(top_pred==y).item()
            num_total["all"] += batch_size

            # Compute topic-wise performance
            topic_labels = batch["topic"]
            topic_list = ["climate", "covid", "military"]

            for topic in topic_list:
                inds = []
                for ind, topic_label in enumerate(topic_labels):
                    if topic in topic_label:
                        inds.append(ind)
                num_total[topic] += len(inds)
                inds = np.array(inds)
                num_correct[topic] += sum(top_pred[inds] == y[inds])
            
            if i % 100 == 0:
                logger.info("%d-th batch: Testing accuracy %.3f, loss: %.3f" % (
                i, num_correct["all"] / num_total["all"], total_loss / num_total["all"]))
        logger.info("Overall testing accuracy %.3f, climate testing accuracy %.3f, covid testing accuracy %.3f, "
                    "military testing accuracy %.3f, loss: %.3f" % (num_correct["all"]/num_total["all"],
                                                                    num_correct["climate"]/num_total["climate"],
                                                                    num_correct["covid"]/num_total["covid"],
                                                                    num_correct["military"]/num_total["military"],
                                                                    total_loss/num_total["all"]))
                
    return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, required=True, default=64, help="Batch size")
    p.add_argument("--few_shot_topic", type=str, required=True,
                   help="topic that will not be included in the training")

    args = p.parse_args()
    return args

if __name__ == '__main__':

    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    args = parse_args()
    BATCH_SIZE = args.batch_size
    few_shot_topic = args.few_shot_topic

    logger.info("Loading blip-2")
    # Load the model
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name = "blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device
    )
    
    tt = TweetTokenizer()
    logger.info("Loading training data")
    # train_data = TwitterCOMMsDataset(csv_path='../raw_data/train_completed.csv',
    #                                 img_dir='/import/network-temp/yimengg/data/twitter-comms/train/images/train_image_ids',
    #                                  few_shot_topic=[few_shot_topic])   # took ~one hour to construct the dataset
    train_data = TwitterCOMMsDataset(csv_path='../raw_data/val_completed.csv',
                                     img_dir='/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids',
                                     few_shot_topic=[few_shot_topic])
    logger.info(f"Found {train_data.__len__()} items in the training data")
    
    logger.info("Loading validation data")
    val_data = TwitterCOMMsDataset(csv_path='../raw_data/val_completed.csv', 
                                   img_dir='/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids')
    logger.info(f"Found {val_data.__len__()} items in the validation data")

    train_iterator = data.DataLoader(train_data, 
                                     shuffle=True, 
                                     batch_size=BATCH_SIZE)
    val_iterator = data.DataLoader(val_data, 
                                   shuffle=False, 
                                   batch_size=BATCH_SIZE)
    
    logger.info("Start training the model")
    
    net = train(train_iterator, val_iterator, device)



