#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

from torch import nn
import pandas as pd
import os

from tqdm import tqdm

from sklearn.metrics import classification_report
import json
from nltk.tokenize import TweetTokenizer
import re, string
import numpy as np


# # Load the model
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# model, vis_processors, txt_processors = load_model_and_preprocess(
#     name = "blip_feature_extractor", model_type="base", is_eval=True, device=device
# )

# model, vis_processors, text_processors = load_model_and_preprocess(
#     "blip2_image_text_matching", "pretrain", device=device, is_eval=True)


model, vis_processors, txt_processors = load_model_and_preprocess(
    name = "blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device
)

tt = TweetTokenizer()


def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", '', text)
    
def remove_punc(text):
    """Remove punctuation from a sample string"""
    return re.sub(r'[^\w\s]', '', text)

def save_tensor(df, img_dir):
    for idx, row in tqdm(df.iterrows()):
        #item = df.iloc[idx]
        item = row
        caption = item['full_text']
        caption = ' '.join(tt.tokenize(caption))
        caption = remove_punc(remove_URL(caption))

        img_filename = item['filename']
        topic = item['topic']
        falsified = int(item['falsified'])
        not_falsified = float(not item['falsified'])
        label = np.array(falsified)
        domain = topic.split('_')[0]
        diff = topic.split('_')[1]

        raw_image = Image.open(os.path.join(img_dir, img_filename)).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)
        sample = {"image": image, "text_input": [text_input]}   # image shape: [1, 3, 224, 224]

        features_multimodal = model.extract_features(sample, mode="multimodal")   # [1, 32, 768] ??? image and text might mismatch
        multimodal_emb = features_multimodal.multimodal_embeds[:, 0, :]   # [1, 768]
        tensor_filename = img_filename.split('.')[0] + '.pt'

    #     print(os.path.join(img_dir, tensor_filename))
        target_path = os.path.join(img_dir, tensor_filename)
#        if not os.path.exists(target_path):
        print(os.path.join(img_dir, tensor_filename))
        torch.save(multimodal_emb, target_path)




# # Save tensors

if __name__ == '__main__':
    
    val_img_dir='/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids'
    df_val = pd.read_csv('../data/val_completed.csv', index_col=0)
    
    df_val['exists'] = df_val['filename'].apply(lambda filename: os.path.exists(os.path.join(val_img_dir, filename)))
    delete_row = df_val[df_val["exists"]==False].index
    df_val = df_val.drop(delete_row)
    
    save_tensor(df_val, val_img_dir)
    
    train_img_dir='/import/network-temp/yimengg/data/twitter-comms/train/images/train_image_ids'
    df_train = pd.read_csv('../data/train_completed.csv', index_col=0)
    
    df_train['exists'] = df_train['filename'].apply(lambda filename: os.path.exists(os.path.join(train_img_dir, filename)))
    delete_row = df_train[df_train["exists"]==False].index
    df_train = df_train.drop(delete_row)
    
    save_tensor(df_train, train_img_dir)

    






