from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data


class NewsCLIPpingsDataset(Dataset):
    def __init__(self, data_dir, split_name, img_dir, multimodal_embeds_path, label_path, phase="test"):
        """
        Args:
            data_dir (string): directory that stores the dataset
            split_name (string): {merged_balanced, person_sbert_text_text, scene_resnet_place, semantics_clip_text_image, semantics_clip_text_text}
            img_dir (string): directory containing the images
            multimodal_embeds_path (string): file path that stores the blip-2 embeddings
            label_path (string): directory that stores the labels
            phase (string): {train, val, test}
        """

        self.img_dir = img_dir
        # self.data = load_json(os.path.join(data_dir, split_name, f"{phase}.json"))["annotations"]
        self.multimodal_embeds = load_tensor(multimodal_embeds_path)
        self.label = load_tensor(label_path)
        self.phase = phase

    def __len__(self):
        return self.multimodal_embeds.shape[0]
    
    def __getitem__(self, idx):
        multimodal_emb = self.multimodal_embeds[idx]
        falsified = self.label[idx]

        return multimodal_emb, falsified
    

def get_dataloader(target_domain, shuffle, batch_size, phase='test'):
    root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    data_dir = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/'
    img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
    if phase=='train':
        source_domain_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
        source_domain_list.remove(target_domain)
        source_domain_datasets = []
        for source_domain in source_domain_list:
            print(f"source domain: {source_domain}")
            multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{source_domain}_multimodal_embeds_{phase}_GaussianBlur.pt'
            label_path = f'{root_dir}/label/blip-2_{source_domain}_multimodal_label_{phase}_GaussianBlur.pt'
            dataset = NewsCLIPpingsDataset(data_dir, source_domain, img_dir, multimodal_embeds_path, label_path, phase)
            source_domain_datasets.append(dataset)
        
        train_data = data.ConcatDataset(source_domain_datasets)
        train_loader = data.DataLoader(train_data,
                                       shuffle=shuffle,
                                       batch_size=batch_size)
        return train_loader, train_data.__len__()
    if phase=='test':
        multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{target_domain}_multimodal_embeds_{phase}_GaussianBlur.pt'
        label_path = f'{root_dir}/label/blip-2_{target_domain}_multimodal_label_{phase}_GaussianBlur.pt'
        test_data = NewsCLIPpingsDataset(data_dir, target_domain, img_dir, multimodal_embeds_path, label_path, phase)
        test_loader = data.DataLoader(test_data,
                                      shuffle=shuffle,
                                      batch_size=batch_size)
        return test_loader, test_data.__len__()
    

if __name__ == "__main__":
    dataloader, length = get_dataloader('semantics_clip_text_text', True, 256, 'test')
    print(length)
