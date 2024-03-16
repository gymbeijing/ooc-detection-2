import os
from pathlib import Path

import json
import string
import pickle
import random
from abc import *
import numpy as np
import pandas as pd

import torch.nn.functional as F
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from configs.config_CANMD import args

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler
import torch.utils.data as data
from torch.autograd import Variable

import torch
from dataset.newsCLIPpingsDataset import get_dataloader_2


def get_corrected_psuedolabels_model(
    args,
    model,
    device,
    conf_threshold=0.9,
    max_imbalance_multiplier=20):  # choose max_imbalance_multiplier from 5 to 20

    print('***** Start pseudo labeling with correction *****')
    # topic_list = ["military", "climate", "covid"]
    # topic_list.remove(args.few_shot_topic)
    # root_dir = '/import/network-temp/yimengg/data/'
    # tgt_train_dataset = TwitterCOMMsDataset(feather_path='./raw_data/train_completed_exist.feather',
    #                                  img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
    #                                  multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{args.base_model}_multimodal_embeds_train.pt',
    #                                  metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{args.base_model}_idx_to_image_path_train.json',
    #                                  few_shot_topic=','.join(topic_list))  # took ~one hour to construct the dataset
    tgt_train_dataset, _, _ = get_dataloader_2(target_agency="bbc,guardian,usa_today", shuffle=True, batch_size=args.batch_size, phase='train')
    tgt_train_dataloader = data.DataLoader(tgt_train_dataset,
                                     shuffle=True,
                                     batch_size=args.batch_size)

    model.eval()
    
    best_probs = []
    embs = []
    with torch.no_grad():
        for i, batch in enumerate(tgt_train_dataloader):
            embs.append(batch["multimodal_emb"])
            inputs_embeds, labels = batch["multimodal_emb"].to(device), batch["label"].to(device)
            inputs_embeds, labels = Variable(inputs_embeds), Variable(labels)
            outputs = model(inputs_embeds, labels)
            best_probs += torch.softmax(outputs["logits"], -1).tolist()

    current_best_probs = best_probs
    indices = np.array(current_best_probs).max(-1) >= conf_threshold
    pseudolabels = np.array(current_best_probs).argmax(-1)[indices]
    filtered_embs = torch.cat(embs, dim=0)[indices]

    return filtered_embs, pseudolabels


if __name__=="__main__":
    source_types = ['liar']
    source_paths = ['./data/LIAR']
    target_types = ['constraint']
    target_paths = ['./data/Constraint']