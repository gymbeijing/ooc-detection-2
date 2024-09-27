import logging
import os
import subprocess
from itertools import count
from multiprocessing import Process
from model.conDA import ContrastiveLearningAndTripletLossModule, ProjectionMLP, MLLMClassificationHead, ContrastiveLearningLossZModule

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, BatchSampler
from tqdm import tqdm
# from transformers import *
from itertools import cycle
from functools import reduce
from torch.utils import data

from dataset.newsCLIPpingsDatasetConDATriplet import NewsCLIPpingsDatasetConDATriplet
from configs.configConDANews import ConfigConDANews
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score
import numpy as np
import argparse


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}')]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]
    emb_list = []
    z_list = []
    agency_list = []

    with tqdm(records, desc=desc) as loop, torch.no_grad():
        for example in loop:
            # print(example)
            for data in example:
                emb, labels, agency = data["original_multimodal_emb"], data["original_label"], data["domain_label"]
                emb_list.append(emb)
                agency_list += agency
                emb, labels = emb.to(device), labels.to(device)

                ###### For the z instead of h input to the model ######
                z = model.mlp(emb)
                z_list.append(z.cpu())
                ###############

        emb_tensor = torch.cat(emb_list)
        z_tensor = torch.cat(z_list)

    return emb_tensor, z_tensor, agency_list


class ConfigConDANews(object):
    def __init__(self):
        parser, args = parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.args.in_dim = 768
        self.args.proj_dim = 500   # original: 300
        self.args.hidden_size = 500   # original: 768 (for h->cls), 500 is for z->cls
        self.args.num_labels = 2
        self.args.learning_rate = 1e-4   # original: 2e-5
        self.args.model_save_path = "./saved_model"
        self.args.model_save_name = "ConDANews.pt"
        self.args.classifier_dropout = 0.2


def parse():
    p = argparse.ArgumentParser()

    p.add_argument("--batch_size", type=int, required=False, default=256, help="batch size")

    args = p.parse_args()
    return p, args


def get_dataset(root_dir, data_dir, img_dir, split, phase, target_domain=None):
    # print(f"      split: {split}")
    original_multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'
    positive_multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_GaussianBlur.pt'
    negative_multimodal_embeds_path = f'{root_dir}/tensor/blip-2_{split}_multimodal_embeds_{phase}_original.pt'   # placeholder
    label_path = f'{root_dir}/label/blip-2_{split}_multimodal_label_{phase}_GaussianBlur.pt'   # original and positive share the labels
    news_source_path = f'{root_dir}/news_source/blip-2_{split}_multimodal_news_source_{phase}_GaussianBlur.json'
    target_domain = target_domain
    dataset = NewsCLIPpingsDatasetConDATriplet(img_dir, original_multimodal_embeds_path, positive_multimodal_embeds_path, negative_multimodal_embeds_path, label_path, news_source_path, target_domain, phase)
    return dataset


def get_dataloader(cfg, phase='test'):   # to be put into cfg
    root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    data_dir = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/'
    img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
    split_list = os.listdir(data_dir)   # ['semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text', 'merged_balanced', 'semantics_clip_text_image']
    split_datasets = []
    for split in split_list:
        dataset = get_dataset(root_dir=root_dir, data_dir=data_dir, img_dir=img_dir, split=split, phase=phase, target_domain="guardian")
        split_datasets.append(dataset)

    test_dataset = data.ConcatDataset(split_datasets)
    test_loader = data.DataLoader(test_dataset,
                                  shuffle=False,
                                  batch_size=cfg.args.batch_size)
    return test_loader


if __name__ == "__main__":
    root_dir = '/import/network-temp/yimengg/data/'
    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cfg = ConfigConDANews()
    val_iterator = get_dataloader(cfg)

    # (1) classification MLP
    mllm_cls_head = MLLMClassificationHead(cfg).to(device)

    # (2) projection MLP
    mlp = ProjectionMLP(cfg).to(device)

    # (3) the entire contrastive learning framework
    model = ContrastiveLearningLossZModule(model=mllm_cls_head, mlp=mlp, loss_type="simclr", logger=None, device=device,
                                        lambda_w=0.5, lambda_mmd=1.0)
    model.load_state_dict(torch.load('./saved_model/ConDANews.pt')["model_state_dict"])
    
    emb_tensor, z_tensor, agency_list = validate(model, device, val_iterator)
    print(emb_tensor.shape)
    torch.save(emb_tensor, './output/newsclip_emb.pt')
    print(z_tensor.shape)
    torch.save(z_tensor, './output/z_B.pt')
    print(len(agency_list))
    
    import json
    with open('./output/agency.json', 'w') as f:
        json.dump(agency_list, f)