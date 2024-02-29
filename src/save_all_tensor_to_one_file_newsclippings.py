#!/usr/bin/env python
# coding: utf-8

'''
Converts NewsCLIPings's caption and image in a json file to multimodal embeddings

python -m src.save_all_tensor_to_one_file_newsclippings --phase PHASE --split SPLIT --base_model blip-2 --mode MODE
'''

import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
from lavis.models import load_model_and_preprocess

import pandas as pd
import os

from tqdm import tqdm
import json
from nltk.tokenize import TweetTokenizer
import numpy as np
import logging
import argparse
from utils.helper import save_tensor, save_json, remove_url, remove_punc

from torchvision import transforms

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


class NewsDataset(Dataset):
    def __init__(self, img_dir, data_dict, vis_processors, txt_processors):
        self.img_dir = img_dir
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.annotations = data_dict["annotations"]
        self.transforms = transforms.Compose([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        caption = visual_news_data_mapping[ann["id"]]["caption"]  # original caption
        # caption = ' '.join(tt.tokenize(caption))  # tokenized caption
        # caption = remove_punc(remove_url(caption))  # remove url & punctuation from the tokenized caption

        image_path = visual_news_data_mapping[ann["image_id"]]["image_path"]
        # print(image_path)
        news_source = image_path.split('/')[1]
        image_path = '/'.join(image_path.split('/')[1:])
        image_path = os.path.join(self.img_dir, image_path)

        raw_image = Image.open(image_path).convert('RGB')
        ### for image augmentation ###
        raw_image = self.transforms(raw_image)
        ##############################
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)

        falsified = ann["falsified"]

        return image, text_input, image_path, falsified, news_source


def get_multimodal_feature(dataloader, model, mode):
    temp_image_path_list = []
    temp_multimodal_embeds_list = []
    temp_label_list = []
    temp_news_source_list = []
    for i, (batch_image, batch_text_input, batch_image_path, batch_label, batch_news_source) in tqdm(enumerate(dataloader, 0)):
        batch_image = batch_image.squeeze(dim=1)
        samples = {"image": batch_image, "text_input": list(batch_text_input)}
        if base_model == 'blip-2':
            if mode == 'multimodal':
                features = model.extract_features(samples, mode="multimodal")
                multimodal_embeds = features.multimodal_embeds[:, 0, :]  # [bs, 1, 768]
                # multimodal_embeds = features.multimodal_embeds[:, :, :].mean(dim=1)  # [bs, 1, 768]
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
        temp_label_list.append(batch_label.detach().cpu())
        temp_news_source_list += list(batch_news_source)

    out_dict = {index: image_path for index, image_path in enumerate(temp_image_path_list)}   # Caution: image_path not unique
    out_tensor = torch.cat(temp_multimodal_embeds_list, dim=0)
    out_label_tensor = torch.cat(temp_label_list, dim=0)
    out_news_source_dict = {"news_source": temp_news_source_list}
    assert len(out_dict) == out_tensor.shape[0], "The number of metadata doesn't equal to the number of tensors "
    assert out_label_tensor.shape[0] == out_tensor.shape[0], "The number of labels doesn't equal to the number of tensors"
    assert len(temp_news_source_list) == out_tensor.shape[0], "The number of labels doesn't equal to the number of tensors"

    return out_dict, out_tensor, out_label_tensor, out_news_source_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, required=True, help="{train, valid, test}")
    p.add_argument("--split", type=str, required=True, help="{semantics_clip_text_image, semantics_clip_text_text, person_sbert_text_text, scene_resnet_place, merged_balanced}")
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
    p.add_argument("--mode", type=str, required=True, default="multimodal", help="{unimodal, multimodal}")

    args = p.parse_args()
    return args


def get_img_dir_and_json(phase, split):
    
    if phase == 'train':
        train_img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
        train_data_path = f"/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/{split}/train.json"
        logger.info(f"Reading json file from {train_data_path}")
        train_data = json.load(open(train_data_path))

        # df_train['exists'] = df_train['filename'].apply(
        #     lambda filename: os.path.exists(os.path.join(train_img_dir, filename)))
        # delete_row = df_train[df_train["exists"] == False].index
        # df_train = df_train.drop(delete_row)

        return train_img_dir, train_data
    if phase == 'valid':
        val_img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
        val_data_path = f"/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/{split}/val.json"
        logger.info(f"Reading json file from {val_data_path}")
        val_data = json.load(open(val_data_path))

        return val_img_dir, val_data
    if phase == 'test':
        test_img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
        test_data_path = f"/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/{split}/test.json"
        logger.info(f"Reading json file from {test_data_path}")
        test_data = json.load(open(test_data_path))

        # df_train['exists'] = df_train['filename'].apply(
        #     lambda filename: os.path.exists(os.path.join(train_img_dir, filename)))
        # delete_row = df_train[df_train["exists"] == False].index
        # df_train = df_train.drop(delete_row)

        return test_img_dir, test_data

    
if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    phase = args.phase
    split = args.split
    base_model = args.base_model
    mode = args.mode
    logger.info(f'phase: {phase}')
    logger.info(f'split: {split}')
    logger.info(f'base model: {base_model}')
    logger.info(f'feature mode: {mode}')
    assert base_model == 'clip' or base_model == 'blip-2' or base_model == 'albef', "Please specify a valid base_model."

    visual_news_data = json.load(open("/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}

    # Get image directory and dataframe
    img_dir, df = get_img_dir_and_json(phase, split)
    logger.info(f'image directory: {img_dir}, split: {split}')

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
    image_path_dict, multimodal_feature_tensor, label_tensor, news_source_dict = get_multimodal_feature(image_text_metadata_loader, model, mode)

    root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    logger.info(f"Saving tensor to {root_dir}/tensor/{base_model}_{split}_{mode}_embeds_{phase}_GaussianBlur.pt")
    save_tensor(multimodal_feature_tensor,
                f'{root_dir}/tensor/{base_model}_{split}_{mode}_embeds_{phase}_GaussianBlur.pt')
    
    logger.info(f"Saving dictionary to {root_dir}/metadata/{base_model}_{split}_{mode}_idx_to_image_path_{phase}_GaussianBlur.json")
    save_json(image_path_dict,
              f'{root_dir}/metadata/{base_model}_{split}_{mode}_idx_to_image_path_{phase}_GaussianBlur.json')
    
    logger.info(f"Saving list to {root_dir}/label/{base_model}_{split}_{mode}_label_{phase}_GaussianBlur.pt")
    save_tensor(label_tensor,
              f'{root_dir}/label/{base_model}_{split}_{mode}_label_{phase}_GaussianBlur.pt')
    
    logger.info(f"Saving dictionary to {root_dir}/news_source/{base_model}_{split}_{mode}_news_source_{phase}_GaussianBlur.json")
    save_json(news_source_dict,
              f'{root_dir}/news_source/{base_model}_{split}_{mode}_news_source_{phase}_GaussianBlur.json')
    # print(news_source_dict)