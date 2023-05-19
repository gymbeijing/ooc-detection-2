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
from sentence_transformers import util
from nltk.tokenize import TweetTokenizer
import re, string

import logging
import numpy as np

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

# # Define the Dataset class

class TwitterCOMMsDataset(Dataset):
    def __init__(self, csv_path, img_dir, seen_topics=['military', 'covid', 'climate'], few_shot_topic=[]):
        """
        Args:
            csv_path (string): Path to the {train_completed|val_completed}.csv file.
            image_folder_dir (string): Directory containing the images.
            selected_topic (string): Topic
        """
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        
        self.df['exists'] = self.df['filename'].apply(lambda filename: os.path.exists(os.path.join(img_dir, filename)))
        delete_row = self.df[self.df["exists"]==False].index
        self.df = self.df.drop(delete_row)
        
        if 'military' in few_shot_topic:
            self.df['is_military'] = self.df['topic'].apply(lambda topic: 'military' in topic)
            delete_row = self.df[self.df["is_military"]==True].index
            keep_row = self.df[self.df["is_military"]==True].sample(n=5, random_state=42).index   # the few shots
            delete_row = np.array(list(set(delete_row)-set(keep_row)))
            self.df = self.df.drop(delete_row)

        if 'covid' in few_shot_topic:
            self.df['is_covid'] = self.df['topic'].apply(lambda topic: 'covid' in topic)
            delete_row = self.df[self.df["is_covid"]==True].index
            keep_row = self.df[self.df["is_covid"]==True].sample(n=5, random_state=42).index   # the few shots
            delete_row = np.array(list(set(delete_row)-set(keep_row)))
            self.df = self.df.drop(delete_row)

        if 'climate' in few_shot_topic:
            self.df['is_climate'] = self.df['topic'].apply(lambda topic: 'climate' in topic)
            delete_row = self.df[self.df["is_climate"]==True].index
            keep_row = self.df[self.df["is_climate"]==True].sample(n=5, random_state=42).index   # the few shots
            delete_row = np.array(list(set(delete_row)-set(keep_row)))
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
        
        topic = item['topic']
        falsified = int(item['falsified'])
        not_falsified = float(not item['falsified'])
#         label = np.array((falsified, not_falsified))
        label = np.array(falsified)
        domain = topic.split('_')[0]
        diff = topic.split('_')[1]
        
        # Preprocess the text
        caption = item['full_text']
        caption = ' '.join(tt.tokenize(caption))
        caption = self.remove_punc(self.remove_URL(caption))
        
        # Open the image
        img_filename = item['filename']
        raw_image = Image.open(os.path.join(self.img_dir, img_filename)).convert('RGB')
        
        # Get the multimodal embedding
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)
        sample = {"image": image, "text_input": [text_input]}   # image shape: [1, 3, 224, 224]
        features = model.extract_features(sample)
        features_image = features.image_embeds   # [1, 512]
        features_text = features.text_embeds   # [1, 512]
#         multimodal_emb = torch.cat((features_image, features_text), 1)
        multimodal_emb = features_image * features_text
        
        cos_sim = util.cos_sim(features_text, features_image)
#         features_image_proj = features_image.image_embeds_proj[:,0,:]   # [1, 256]
#         features_text_proj = features_text.text_embeds_proj[:,0,:]   # [1, 256]
        
#         multimodal_emb = torch.cat((features_image_proj, features_text_proj), 1)
#         multimodal_emb = features_image_proj * features_text_proj   # [1, 256]
#         print(multimodal_emb.shape)

#         similarity = features_image_proj @ features_text_proj.t()

        return {"multimodal_emb": multimodal_emb,
                "topic": topic, 
                "label": label, 
                "domain": domain, 
                "difficulty": diff,
               "similarity": cos_sim}
        
        
    
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


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def train(train_iterator, val_iterator, device):

    net = Net(512)
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)
    
    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    softmax = nn.Softmax(dim=1)
 
    EPOCHS = 5
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
            
            top_pred = torch.zeros_like(labels)
            y_preds = softmax(y_preds)
#             print(y_preds[:, 0])
            top_pred[y_preds[:, 1] >= 0.5] = 1
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)
            
            num_correct += sum(top_pred == y).item()
            num_total += batch_size
            
            if i % 100 == 0:
                logger.info("Epoch [%d/%d] %d-th batch: Training accuracy: %.2f, loss: %.2f" % (epoch+1, EPOCHS, i, num_correct/num_total, total_loss/num_total))
            
        logger.info("Epoch [%d/%d]: Training accuracy: %.2f, loss: %.2f" % (epoch+1, EPOCHS, num_correct/num_total, total_loss/num_total))
                
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
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)
            
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)
            
            total_loss += loss.item()
            
            top_pred = torch.zeros_like(labels)
            y_preds = softmax(y_preds)
#             print(y_preds[:, 0])
            top_pred[y_preds[:, 1] >= 0.5] = 1
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)
            
            num_correct["all"] += sum(top_pred == y).item()
            num_total["all"] += batch_size
            
            # topic-wise performance
            topic_labels = batch["topic"]
            
            topic_list = ["climate", "covid", "military"]
            
            for topic in topic_list:
                inds = []
                # print(topic_labels)   # class 'list'
                for j, tl in enumerate(topic_labels):
                    if topic in tl:
                        inds.append(j)
                num_total[topic] += len(inds)
                inds = np.array(inds)
                num_correct[topic] += sum(top_pred[inds] == y[inds]).item()
            
            if i % 100 == 0:
                logger.info("%d-th batch: Testing accuracy %.2f, loss: %.2f" % (i, num_correct["all"]/num_total["all"], total_loss/num_total["all"]))
            
        logger.info("Overall testing accuracy %.2f, climate testing accuracy %.2f, covid testing accuracy %.2f, military testing accuracy %.2f, loss: %.2f" % (num_correct["all"]/num_total["all"], num_correct["climate"]/num_total["climate"], num_correct["covid"]/num_total["covid"], num_correct["military"]/num_total["military"], total_loss/num_total["all"]))
                
    return


if __name__ == '__main__':

    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    logger.info("Loading the clip model")
    # Load the model
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name = "clip_feature_extractor", model_type="ViT-B-32", is_eval=True, device=device
    )
    # Load the tokenizer
    tt = TweetTokenizer()
    
    logger.info("Loading training data")
    train_data = TwitterCOMMsDataset(csv_path='../data/train_completed.csv',
                                     img_dir='/import/network-temp/yimengg/data/twitter-comms/train/images/train_image_ids',
                                    seen_topics=['military', 'climate'],
                                    few_shot_topic=['covid'])   # took ~one hour to construct the dataset
    logger.info(f"Found {train_data.__len__()} items in training data")
    
    logger.info("Loading valid data")
    val_data = TwitterCOMMsDataset(csv_path='../data/val_completed.csv', 
                                   img_dir='/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids')
    logger.info(f"Found {val_data.__len__()} items in valid data")
    
    BATCH_SIZE = 32
    train_iterator = data.DataLoader(train_data, 
                                     shuffle=True, 
                                     batch_size=BATCH_SIZE)
    val_iterator = data.DataLoader(val_data, 
                                   shuffle=False, 
                                   batch_size=BATCH_SIZE)
    
    logger.info("Start training the model")
    
    net = train(train_iterator, val_iterator, device)



