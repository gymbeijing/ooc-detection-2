#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

from torch import nn
import pandas as pd
import os

from tqdm import tqdm
import json
from nltk.tokenize import TweetTokenizer
import re, string
import numpy as np
import logging
import argparse


# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

tt = TweetTokenizer()


class NewsDataset(Dataset):
    def __init__(self, img_dir, df, vis_processors, txt_processors):
        self.img_dir = img_dir
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        caption = item['full_text']  # original caption
        caption = ' '.join(tt.tokenize(caption))  # tokenized caption
        caption = self.remove_punc(self.remove_url(caption))  # remove url & punctuation from the tokenized caption

        img_filename = item['filename']
        image_path = os.path.join(self.img_dir, img_filename)

        raw_image = Image.open(image_path).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)

        return image, text_input, image_path

    def remove_url(self, text):
        """Remove URLs from a sample string"""
        return re.sub(r"http\S+", '', text)

    def remove_punc(self, text):
        """Remove punctuation from a sample string"""
        return re.sub(r'[^\w\s]', '', text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, required=True, help="train or valid")

    args = p.parse_args()
    return args


def get_img_dir_and_df(phase):
    if phase == 'valid':
        val_img_dir = '/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids'
        df_val = pd.read_csv('../raw_data/val_completed.csv', index_col=0)

        df_val['exists'] = df_val['filename'].apply(
            lambda filename: os.path.exists(os.path.join(val_img_dir, filename)))
        delete_row = df_val[df_val["exists"] == False].index
        df_val = df_val.drop(delete_row)

        return val_img_dir, df_val
    if phase == 'train':
        train_img_dir = '/import/network-temp/yimengg/data/twitter-comms/train/images/train_image_ids'
        df_train = pd.read_csv('../raw_data/train_completed.csv', index_col=0)

        df_train['exists'] = df_train['filename'].apply(
            lambda filename: os.path.exists(os.path.join(train_img_dir, filename)))
        delete_row = df_train[df_train["exists"] == False].index
        df_train = df_train.drop(delete_row)

        return train_img_dir, df_train

    return None, None


def get_multimodal_feature(dataloader, model):
    temp_image_path_list = []
    temp_multimodal_embeds_list = []
    for i, (batch_image, batch_text_input, batch_image_path) in tqdm(enumerate(dataloader, 0)):
        batch_image = batch_image.squeeze(dim=1)
        samples = {"image": batch_image, "text_input": list(batch_text_input)}
        features_multimodal = model.extract_features(samples, mode="multimodal")
        multimodal_embeds = features_multimodal.multimodal_embeds[:, 0, :]  # [1, 768]

        temp_image_path_list += list(batch_image_path)
        temp_multimodal_embeds_list.append(multimodal_embeds.detach().cpu())

    out_dict = {image_path: index for index, image_path in enumerate(temp_image_path_list)}   # image_path unique?
    out_tensor = torch.cat(temp_multimodal_embeds_list, dim=0)

    return out_dict, out_tensor


def save_tensor(tensor, dest):
    torch.save(tensor, dest)
    return


def save_json(json_dict, dest):
    with open(dest, 'w', encoding='utf8') as fp:
        json.dump(json_dict, fp, indent=4, ensure_ascii=False, sort_keys=False)
    return


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    phase = args.phase
    logger.info(f'phase: {phase}')

    # Get image directory and dataframe
    img_dir, df = get_img_dir_and_df(phase)
    logger.info(f'image directory: {img_dir}')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'device: {device}')

    # Load the model
    logger.info("Loading blip-2")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device
    )

    logger.info("Preparing dataset and dataloader")
    image_text_metadata = NewsDataset(img_dir, df, vis_processors, txt_processors)
    image_text_metadata_loader = data.DataLoader(image_text_metadata, shuffle=False, batch_size=256)

    logger.info("Getting multimodal feature")
    image_path_dict, multimodal_feature_tensor = get_multimodal_feature(image_text_metadata_loader, model)

    logger.info("Saving tensor")
    save_tensor(multimodal_feature_tensor,
                f'/import/network-temp/yimengg/data/twitter-comms/processed_data/tensor/multimodal_embeds_{phase}.pt')
    logger.info("Saving dictionary")
    save_json(image_path_dict,
              f'/import/network-temp/yimengg/data/twitter-comms/processed_data/metadata/image_path_to_idx_{phase}.json')


