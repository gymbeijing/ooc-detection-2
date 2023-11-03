#!/usr/bin/env python
# coding: utf-8

import torch
from lavis.models import load_model_and_preprocess
import pandas as pd
import os
from PIL import Image
import logging
import argparse
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.utils.data as data

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


class NewsDataset(Dataset):
    def __init__(self, img_dir, df, vis_processors):
        self.img_dir = img_dir
        self.vis_processors = vis_processors
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        img_filename = item['filename']
        image_path = os.path.join(self.img_dir, img_filename)

        raw_image = Image.open(image_path).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        return image, image_path


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


def get_image_caption(dataloader, model):
    out_image_caption_list = []
    temp_image_path_list = []
    for i, (batch_image, batch_image_path) in tqdm(enumerate(dataloader, 0)):
        batch_image = batch_image.squeeze(dim=1)
        print(batch_image.shape)
        images = {"image": batch_image}
        generated_image_captions = model.generate(images)
        out_image_caption_list.append(generated_image_captions)
        temp_image_path_list += list(batch_image_path)

    out_image_path_dict = {index: image_path for index, image_path in enumerate(temp_image_path_list)}
    # assert len(out_image_path_dict) == len(out_image_caption_list), "The number of metadata isn't equal to the number of generated captions "

    return out_image_path_dict, out_image_caption_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, required=True, help="train or valid")

    args = p.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    phase = args.phase
    logger.info(f'phase: {phase}')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'device: {device}')

    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)

    # val_feather_path = '../raw_data/val_completed_exist.feather'
    # val_df = pd.read_feather(val_feather_path)  # already drop the non-exists
    # val_img_dir = '/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids'

    # Get image directory and dataframe
    img_dir, df = get_img_dir_and_df(phase)
    logger.info(f'image directory: {img_dir}')

    logger.info("Preparing dataset and dataloader")
    image_metadata = NewsDataset(img_dir, df[:64], vis_processors)
    image_metadata_loader = data.DataLoader(image_metadata, shuffle=False, batch_size=64)

    logger.info("Generating captions")
    image_path_dict, image_caption_list = get_image_caption(image_metadata_loader, model)

    print(image_caption_list)

    # item = val_df.iloc[0]
    # text = item['full_text']  # original caption
    #
    # img_filename = item['filename']
    # image_path = os.path.join(val_img_dir, img_filename)
    #
    # raw_image = Image.open(image_path).convert('RGB')
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # # generate caption
    # generated_text = model.generate({"image": image})
    # # ['a large fountain spewing water into the air']
    # print(generated_text)