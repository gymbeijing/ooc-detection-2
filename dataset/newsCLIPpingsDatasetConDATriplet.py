from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torch


class NewsCLIPpingsDatasetConDATriplet(Dataset):
    def __init__(self, img_dir, original_multimodal_embeds_path, positive_multimodal_embeds_path, 
                 negative_multimodal_embeds_path, label_path, news_source_path, target_domain=None, phase="test"):
        """
        Args:
            img_dir (string): directory that stores the images
            original_multimodal_embeds_path (string): path to the file that stores the original embeddings
            positive_multimodal_embeds_path (string): path to the file that stores the positive (GaussianBlur) embeddings
            negative_multimodal_embeds_path (string): path to the file that stores the negative embeddings
            label_path (string): path to the file that stores the labels
            metadata_path (string): path to the file that stores the metadata
            target_domain (list): list of target domains
            phase (string): {train, val}
        """
        if target_domain is None:
            target_domain = []

        self.img_dir = img_dir
        self.original_multimodal_embeds = load_tensor(original_multimodal_embeds_path)
        self.positive_multimodal_embeds = load_tensor(positive_multimodal_embeds_path)
        self.negative_multimodal_embeds = load_tensor(negative_multimodal_embeds_path)
        self.labels = load_tensor(label_path).type(torch.LongTensor)
        self.domain_labels = load_json(news_source_path)["news_source"]

        self.target_domain = target_domain

        self.phase = phase

        assert self.labels.shape[0] == self.original_multimodal_embeds.shape[0], \
            "The number of news in self.df doesn't equal to number of tensor"
        assert self.original_multimodal_embeds.shape[0] == self.positive_multimodal_embeds.shape[0], \
            "The number original news items doesn't equal to the number of positive news items"
        assert self.original_multimodal_embeds.shape[0] == self.negative_multimodal_embeds.shape[0], \
            "The number original news items doesn't equal to the number of negative news items"

        # if not excluding any topic
        self.row_kept = set(range(self.original_multimodal_embeds.shape[0]))

        # Exclude target domain
        if self.phase == 'train':
        # print(f"phase: {self.phase}, excluded domain: {self.target_domain}")
            row_excluded = [i for i, x in enumerate(self.domain_labels) if x in self.target_domain]
            self.row_kept = self.row_kept.difference(row_excluded)

        self.row_kept = list(self.row_kept)

    def __len__(self):
        return len(self.row_kept)

    def __getitem__(self, idx):
        mapped_idx = self.row_kept[idx]
        original_multimodal_emb = self.original_multimodal_embeds[mapped_idx]
        positive_multimodal_emb = self.positive_multimodal_embeds[mapped_idx]
        negative_multimodal_emb = self.negative_multimodal_embeds[mapped_idx]
        original_label = self.labels[mapped_idx]
        domain_label = self.domain_labels[mapped_idx]

        return {"original_multimodal_emb": original_multimodal_emb,
                "positive_multimodal_emb": positive_multimodal_emb,
                "negative_multimodal_emb": negative_multimodal_emb,
                "original_label": original_label,
                "domain_label": domain_label}
    

def get_dataset(root_dir, data_dir, img_dir, split, phase, target_domain):
    # print(f"      split: {split}")
    original_multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'
    positive_multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'
    negative_multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_GaussianBlur.pt'   # placeholder
    label_path = f'{root_dir}/label/blip-2_{split}_multimodal_label_{phase}_GaussianBlur.pt'   # original and positive share the labels
    news_source_path = f'{root_dir}/news_source/blip-2_{split}_multimodal_news_source_{phase}_GaussianBlur.json'
    target_domain = target_domain
    dataset = NewsCLIPpingsDatasetConDATriplet(img_dir, original_multimodal_embeds_path, positive_multimodal_embeds_path, negative_multimodal_embeds_path, label_path, news_source_path, target_domain, phase)
    return dataset


def get_dataloader(cfg, target_domain, shuffle, phase='test'):   # to be put into cfg
    root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    data_dir = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/'
    img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
    if phase == 'train':
        # print(f"phase: {phase}")
        split_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        split_datasets = []
        for split in split_list:
            dataset = get_dataset(root_dir=root_dir, data_dir=data_dir, img_dir=img_dir, split=split, phase=phase, target_domain=target_domain)
            split_datasets.append(dataset)

        train_dataset = data.ConcatDataset(split_datasets)
        sampler = RandomSampler(train_dataset)   # randomly sampling, order determined by torch.manual_seed()
        batch_sampler = BatchSampler(sampler, batch_size=cfg.args.batch_size, drop_last=True)   # the need to set drop_last=True is the reason why we need batch_sampler
        train_loader = data.DataLoader(train_dataset, batch_sampler=batch_sampler)   # cannot be shuffled
        return train_loader, train_dataset.__len__()
    else:  # phase=='val'
        # print(f"phase: {phase}")
        split_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        split_datasets = []
        for split in split_list:
            dataset = get_dataset(root_dir=root_dir, data_dir=data_dir, img_dir=img_dir, split=split, phase=phase, target_domain=target_domain)
            split_datasets.append(dataset)

        test_dataset = data.ConcatDataset(split_datasets)
        test_loader = data.DataLoader(test_dataset,
                                       shuffle=shuffle,
                                       batch_size=cfg.args.batch_size)
        return test_loader, test_dataset.__len__()
