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
from utils.helper import save_tensor, save_json, remove_url, remove_punc


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
        # caption = item['full_text_perturb']  # perturbed caption
        caption = item['full_text']  # original caption
        # caption = item['rephrased_gpt4']   # for mini_toy df
        caption = ' '.join(tt.tokenize(caption))  # tokenized caption
        caption = remove_punc(remove_url(caption))  # remove url & punctuation from the tokenized caption

        img_filename = item['filename_negative']
        image_path = os.path.join(self.img_dir, img_filename)

        raw_image = Image.open(image_path).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)

        return image, text_input, image_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, required=True, help="{train ,valid, toy, mini_toy}")
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
    p.add_argument("--mode", type=str, required=True, default="multimodal", help="{unimodal, multimodal}")

    args = p.parse_args()
    return args


def get_img_dir_and_df(phase):
    if phase == 'valid':
        val_img_dir = '/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids'
        df_val = pd.read_csv('raw_data/val_completed.csv', index_col=0)

        df_val['exists'] = df_val['filename'].apply(
            lambda filename: os.path.exists(os.path.join(val_img_dir, filename)))
        delete_row = df_val[df_val["exists"] == False].index
        df_val = df_val.drop(delete_row)

        return val_img_dir, df_val
    if phase == 'train':
        train_img_dir = '/import/network-temp/yimengg/data/twitter-comms/train/images/train_image_ids'
        df_train = pd.read_csv('raw_data/train_completed.csv', index_col=0)

        df_train['exists'] = df_train['filename'].apply(
            lambda filename: os.path.exists(os.path.join(train_img_dir, filename)))
        delete_row = df_train[df_train["exists"] == False].index
        df_train = df_train.drop(delete_row)

        return train_img_dir, df_train
    if phase == 'toy' or phase == 'mini_toy':
        toy_img_dir = '/import/network-temp/yimengg/data/twitter-comms/train/images/train_image_ids'
        # df_toy = pd.read_feather('./raw_data/toy_completed_exist_augmented.feather')
        # df_toy = pd.read_feather('./raw_data/toy_completed_exist.feather')
        # df_toy = pd.read_feather('./raw_data/mini_toy_completed_exist_rephrased.feather')
        df_toy = pd.read_feather('./raw_data/toy_completed_exist_triplet.feather')

        return toy_img_dir, df_toy

    return None, None


def get_multimodal_feature(dataloader, model, mode):
    temp_image_path_list = []
    temp_multimodal_embeds_list = []
    for i, (batch_image, batch_text_input, batch_image_path) in tqdm(enumerate(dataloader, 0)):
        batch_image = batch_image.squeeze(dim=1)
        samples = {"image": batch_image, "text_input": list(batch_text_input)}
        if base_model == 'blip-2':
            if mode == 'multimodal':
                features = model.extract_features(samples, mode="multimodal")
                multimodal_embeds = features.multimodal_embeds[:, 0, :]  # [bs, 1, 768]
            else:   # unimodal
                text_features = model.extract_features(samples, mode="text")
                text_embeds = text_features.text_embeds[:, 0, :]   # [bs, 256]

                image_features = model.extract_features(samples, mode="image")
                image_embeds = image_features.image_embeds[:, 0, :]   # [bs, 256]

                # multimodal_embeds = torch.cat((image_embeds_proj, text_embeds_proj), dim=1)   # bad performance ~0.5
                multimodal_embeds = image_embeds * text_embeds   # bad performance ~0.5
        elif base_model == 'albef':
            features_multimodal = model.extract_features(samples)
            multimodal_embeds = features_multimodal.multimodal_embeds[:, 0, :]  # [bs, 1, 768]
        else:   # base_model == 'clip'
            features = model.extract_features(samples)
            features_image = features.image_embeds  # [1, 512]   should it be features.image_embeds_proj?
            features_text = features.text_embeds  # [1, 512]   should it be features.text_embeds_proj
            multimodal_embeds = features_image * features_text   # (? not sure if it's correct, just a placeholder for now)

        temp_image_path_list += list(batch_image_path)
        temp_multimodal_embeds_list.append(multimodal_embeds.detach().cpu())

    out_dict = {index: image_path for index, image_path in enumerate(temp_image_path_list)}   # Caution: image_path not unique
    out_tensor = torch.cat(temp_multimodal_embeds_list, dim=0)
    assert len(out_dict) == out_tensor.shape[0], "The number of metadata isn't equal to the number of tensors "

    return out_dict, out_tensor


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    phase = args.phase
    base_model = args.base_model
    mode = args.mode
    logger.info(f'phase: {phase}')
    logger.info(f'base model: {base_model}')
    logger.info(f'feature mode: {mode}')
    assert base_model == 'clip' or base_model == 'blip-2' or base_model == 'albef', "Please specify a valid base_model."

    # Get image directory and dataframe
    img_dir, df = get_img_dir_and_df(phase)
    logger.info(f'image directory: {img_dir}')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'device: {device}')

    # Load the model
    if base_model == 'blip-2':
        logger.info("Loading blip-2")
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device
        )
    elif base_model == 'albef':
        logger.info("Loading albef")
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="albef_feature_extractor", model_type="base", is_eval=True, device=device
        )
    else:   # base_model == 'clip'
        logger.info("Loading clip")
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="clip_feature_extractor", model_type="ViT-B-32", is_eval=True, device=device
        )

    logger.info("Preparing dataset and dataloader")
    image_text_metadata = NewsDataset(img_dir, df, vis_processors, txt_processors)
    image_text_metadata_loader = data.DataLoader(image_text_metadata, shuffle=False, batch_size=256)

    logger.info("Getting multimodal feature")
    image_path_dict, multimodal_feature_tensor = get_multimodal_feature(image_text_metadata_loader, model, mode)

    logger.info("Saving tensor")
    root_dir = '/import/network-temp/yimengg/data/twitter-comms/processed_data/'
    save_tensor(multimodal_feature_tensor,
                root_dir+f'tensor/{base_model}_{mode}_embeds_{phase}_negative.pt')
    logger.info("Saving dictionary")
    save_json(image_path_dict,
              root_dir+f'metadata/{base_model}_{mode}_idx_to_image_path_{phase}_negative.json')


