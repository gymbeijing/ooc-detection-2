import logging
import os
import subprocess
from itertools import count
from multiprocessing import Process
from model.conDA import ContrastiveLearningAndTripletLossModule, ProjectionMLP, MLLMClassificationHead, ContrastiveLearningAndTripletLossZModule

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
# from transformers import *
from itertools import cycle
from functools import reduce
from torch.utils import data

from dataset.twitterCOMMsDatasetConDATriplet import TwitterCOMMsDatasetConDATriplet
from configs.configConDA import ConfigConDA
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
    topic_list = []

    with tqdm(records, desc=desc) as loop, torch.no_grad():
        targets = []
        outputs = []
        for example in loop:
            losses = []
            logit_votes = []
            # print(example)
            for data in example:
                emb, labels, topic = data["original_multimodal_emb"], data["original_label"], data["topic"]
                emb_list.append(emb)
                topic_list += topic
                emb, labels = emb.to(device), labels.to(device)

                ###### For the z instead of h input to the model ######
                z = model.mlp(emb)
                z_list.append(z.cpu())
                ###############

        emb_tensor = torch.cat(emb_list)
        z_tensor = torch.cat(z_list)

    return emb_tensor, z_tensor, topic_list


class ConfigConDA(object):
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
        self.args.learning_rate = 2e-4   # original: 2e-5
        self.args.model_save_path = "./saved_model"
        self.args.model_save_name = "ConDA_M.pt"
        self.args.classifier_dropout = 0.2


def parse():
    p = argparse.ArgumentParser()

    p.add_argument("--batch_size", type=int, required=False, default=256, help="batch size")

    args = p.parse_args()
    return p, args


if __name__ == "__main__":
    root_dir = '/import/network-temp/yimengg/data/'
    val_dataset = TwitterCOMMsDatasetConDATriplet(triplet_feather_path='./raw_data/val_completed_exist.feather',
                                                img_dir=root_dir + 'twitter-comms/images/val_images/val_tweet_image_ids',
                                                original_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_valid.pt',
                                                positive_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_valid.pt',
                                                negative_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_valid.pt',
                                                metadata_path=root_dir + f'twitter-comms/processed_data/metadata/blip-2_multimodal_idx_to_image_path_valid.json',
                                                )
    val_iterator = data.DataLoader(val_dataset,
                                shuffle=False,
                                batch_size=256)
    
    # toy_dataset = TwitterCOMMsDatasetConDATriplet(triplet_feather_path='./raw_data/toy_completed_exist_triplet.feather',
    #                                            img_dir=root_dir + 'twitter-comms/train/images/train_image_ids',
    #                                            original_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_toy_original.pt',
    #                                         #    original_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_train_original.pt',
    #                                            positive_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_toy_positive.pt',
    #                                         #    positive_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_toy_synonym_replacement_GaussianBlur.pt',
    #                                         #    positive_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_train_GaussianBlur.pt',
    #                                            negative_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_toy_negative.pt',
    #                                         #    negative_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/blip-2_multimodal_embeds_train_negative.pt',
    #                                         #    augmented_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_mini_toy_rephrased.pt',
    #                                            metadata_path=root_dir + f'twitter-comms/processed_data/metadata/blip-2_multimodal_idx_to_image_path_toy_original.json'
    #                                            )
    
    ###### For sampling ######
    random_sampler = data.RandomSampler(val_dataset, num_samples=5000)
    val_iterator = data.DataLoader(val_dataset, batch_size=256, sampler=random_sampler)
    ######

    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cfg = ConfigConDA()

    # (1) classification MLP
    mllm_cls_head = MLLMClassificationHead(cfg).to(device)

    # (2) projection MLP
    mlp = ProjectionMLP(cfg).to(device)

    # (3) the entire contrastive learning framework
    model = ContrastiveLearningAndTripletLossZModule(model=mllm_cls_head, mlp=mlp, loss_type="simclr", logger=None, device=device,
                                        lambda_w=0.5, lambda_mmd=1.0)
    model.load_state_dict(torch.load('./saved_model/ConDA_M.pt')["model_state_dict"])
    
    emb_tensor, z_tensor, topic_list = validate(model, device, val_iterator)
    # print(emb_tensor.shape)
    torch.save(emb_tensor, './output/emb_val_5k.pt')
    print(z_tensor.shape)
    torch.save(z_tensor, './output/z_M_val_5k.pt')
    print(len(topic_list))
    
    import json
    with open('./output/topic_val_5k.json', 'w') as f:
        json.dump(topic_list, f)