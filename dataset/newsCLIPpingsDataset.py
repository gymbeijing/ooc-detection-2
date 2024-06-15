from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data
import torch


class NewsCLIPpingsDataset(Dataset):
    def __init__(self, target_agency, img_dir, multimodal_embeds_path, label_path, news_source_path, phase="test"):
        """
        Args:
            target_domain (string): target domain
            img_dir (string): directory containing the images
            multimodal_embeds_path (string): file path that stores the blip-2 embeddings
            label_path (string): directory that stores the labels
            phase (string): {train, val, test}
        """

        self.img_dir = img_dir
        # self.data = load_json(os.path.join(data_dir, split_name, f"{phase}.json"))["annotations"]
        self.target_agency = target_agency
        self.multimodal_embeds = load_tensor(multimodal_embeds_path)
        self.label = load_tensor(label_path).type(torch.LongTensor)
        self.news_source = load_json(news_source_path)["news_source"]
        self.phase = phase

        self.row_kept = set(range(self.multimodal_embeds.shape[0]))
        agencies = ["bbc", "guardian", "washington_post", "usa_today"]
        # topics = ["arts_culture", "culture", "film", "music", "artsanddesign",
        #           "world", 
        #           "international_relations", 
        #           "law_crime", "law",
        #           "science_technology", "technology", "science",
        #           "football", "sport", "sports",
        #           "politics_elections", "politics",
        #           "business", "business_economy",
        #           "media", 
        #           "environment"]
        # topics = ["law_crime", "law",
        #           "media",
        #           "education",
        #           "books",
        #           "society",
        #           "edinburgh",
        #           "leeds",
        #           "stage"]
        # if target_domain in topics and phase=="train":
        target_agency = target_agency.split(",")
        # if target_agency in agencies and phase=="train":
        if phase=="train":
        # if phase=="train" or phase=="test":   # for canmd and real_fnd (test_realFND.py) and mdaws
            # row_excluded = [i for i, x in enumerate(self.news_source) if x == target_agency or x == 'washington_post']
            row_excluded = [i for i, x in enumerate(self.news_source) if x in target_agency]
            # row_excluded = [i for i, x in enumerate(self.topic) if x in target_domain or x not in topics]
            self.row_kept = self.row_kept.difference(row_excluded)
        # print(len(row_excluded))
        
        self.row_kept = list(self.row_kept)
        self.domain_map_to_idx = {"bbc": 0, "guardian": 1, "usa_today": 2, "washington_post": 3}


    def __len__(self):
        return len(self.row_kept)
    
    def __getitem__(self, idx):

        mapped_idx = self.row_kept[idx]
        multimodal_emb = self.multimodal_embeds[mapped_idx]
        falsified = self.label[mapped_idx]
        news_source = self.news_source[mapped_idx]
        domain_id = self.domain_map_to_idx[news_source]

        return {"multimodal_emb": multimodal_emb, 
                "label": falsified,
                "news_source": news_source,
                "domain_id": domain_id}
    

def get_dataloader(target_domain, shuffle, batch_size, phase='test'):
    # domain: split
    root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    data_dir = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/'
    img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
    if phase=='train':
        source_domain_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        if target_domain in source_domain_list:
            source_domain_list.remove(target_domain)
        source_domain_datasets = []
        for source_domain in source_domain_list:
            print(f"source domain: {source_domain}")
            multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{source_domain}_multimodal_embeds_{phase}_original.pt'
            label_path = f'{root_dir}/label/blip-2_{source_domain}_multimodal_label_{phase}_GaussianBlur.pt'
            news_source_path = f'{root_dir}/news_source/blip-2_{source_domain}_multimodal_news_source_{phase}_GaussianBlur.json'
            dataset = NewsCLIPpingsDataset(data_dir, source_domain, img_dir, multimodal_embeds_path, label_path, news_source_path, phase)
            source_domain_datasets.append(dataset)
        
        train_data = data.ConcatDataset(source_domain_datasets)
        train_loader = data.DataLoader(train_data,
                                       shuffle=shuffle,
                                       batch_size=batch_size)
        return train_loader, train_data.__len__()
    if phase=='test':
        multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{target_domain}_multimodal_embeds_{phase}_original.pt'
        label_path = f'{root_dir}/label/blip-2_{target_domain}_multimodal_label_{phase}_GaussianBlur.pt'
        news_source_path = f'{root_dir}/news_source/blip-2_{target_domain}_multimodal_news_source_{phase}_GaussianBlur.json'
        test_data = NewsCLIPpingsDataset(data_dir, target_domain, img_dir, multimodal_embeds_path, label_path, news_source_path, phase)
        test_loader = data.DataLoader(test_data,
                                      shuffle=shuffle,
                                      batch_size=batch_size)
        return test_loader, test_data.__len__()
    

def get_dataloader_2(target_agency, shuffle, batch_size, phase='test'):
    # domain: news source
    root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    data_dir = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/'
    img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
    if phase=='train':
        print(f"phase: {phase}")
        split_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        split_datasets = []
        for split in split_list:
            print(f"      split: {split}")
            multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'
            label_path = f'{root_dir}/label/blip-2_{split}_multimodal_label_{phase}_GaussianBlur.pt'
            news_source_path = f'{root_dir}/news_source/blip-2_{split}_multimodal_news_source_{phase}_GaussianBlur.json'
            dataset = NewsCLIPpingsDataset(target_agency, img_dir, multimodal_embeds_path, label_path, news_source_path, phase)
            split_datasets.append(dataset)
        
        train_data = data.ConcatDataset(split_datasets)
        train_loader = data.DataLoader(train_data,
                                       shuffle=shuffle,
                                       batch_size=batch_size)
        # return train_data, train_loader, train_data.__len__()   # for canmd and real_fnd (train_agentNews.py) and mdaws
        return train_loader, train_data.__len__()
    if phase=='test':
        print(f"phase: {phase}")
        split_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        split_datasets = []
        for split in split_list:
            print(f"      split: {split}")
            multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'
            label_path = f'{root_dir}/label/blip-2_{split}_multimodal_label_{phase}_GaussianBlur.pt'
            news_source_path = f'{root_dir}/news_source/blip-2_{split}_multimodal_news_source_{phase}_GaussianBlur.json'
            dataset = NewsCLIPpingsDataset(target_agency, img_dir, multimodal_embeds_path, label_path, news_source_path, phase)
            split_datasets.append(dataset)
        
        test_data = data.ConcatDataset(split_datasets)
        test_loader = data.DataLoader(test_data,
                                      shuffle=shuffle,
                                      batch_size=batch_size)
        return test_loader, test_data.__len__()
    

def get_dataloader_3(target_domain, shuffle, batch_size, phase='test'):
    # domain: topic
    target_domain = target_domain.split(',')
    # print(target_domain)
    root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    data_dir = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/'
    img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
    if phase=='train':
        print(f"phase: {phase}")
        split_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        split_datasets = []
        for split in split_list:
            print(f"      split: {split}")
            multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'
            label_path = f'{root_dir}/label/blip-2_{split}_multimodal_label_{phase}_GaussianBlur.pt'
            news_source_path = f'{root_dir}/news_source/blip-2_{split}_multimodal_news_source_{phase}_GaussianBlur.json'
            topic_path = f'{root_dir}/topic/{split}_topic_{phase}_GaussianBlur.json'
            dataset = NewsCLIPpingsDataset(target_domain, img_dir, multimodal_embeds_path, label_path, news_source_path, topic_path, phase)
            split_datasets.append(dataset)
        
        train_data = data.ConcatDataset(split_datasets)
        train_loader = data.DataLoader(train_data,
                                       shuffle=shuffle,
                                       batch_size=batch_size)
        return train_loader, train_data.__len__()
    if phase=='test':
        print(f"phase: {phase}")
        split_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        split_datasets = []
        for split in split_list:
            print(f"      split: {split}")
            multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'
            label_path = f'{root_dir}/label/blip-2_{split}_multimodal_label_{phase}_GaussianBlur.pt'
            news_source_path = f'{root_dir}/news_source/blip-2_{split}_multimodal_news_source_{phase}_GaussianBlur.json'
            topic_path = f'{root_dir}/topic/{split}_topic_{phase}_GaussianBlur.json'
            dataset = NewsCLIPpingsDataset(target_domain, img_dir, multimodal_embeds_path, label_path, news_source_path, topic_path, phase)
            split_datasets.append(dataset)
        
        test_data = data.ConcatDataset(split_datasets)
        test_loader = data.DataLoader(test_data,
                                      shuffle=False,
                                      batch_size=batch_size)
        return test_loader, test_data.__len__()

if __name__ == "__main__":
    dataloader, length = get_dataloader_3('arts_culture,culture,film,music,artsanddesign', True, 256, 'test')
    print(length)
