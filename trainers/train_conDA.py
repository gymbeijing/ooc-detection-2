import argparse
import logging
import os
import subprocess
from itertools import count
from multiprocessing import Process

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from transformers import *
from itertools import cycle

from dataset.twitterCOMMsDataset import get_dataloader
from configs.configTwoTasks import ConfigTwoTasks


# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def train(train_iterator, val_iterator, device):
    return


if __name__ == '__main__':
    cfg = ConfigTwoTasks()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    train_loader, train_length = get_dataloader(cfg, shuffle=True, phase="train")
    logger.info(f"Found {train_length} items in training data")

    logger.info("Loading valid data")
    val_loader, val_length = get_dataloader(cfg, shuffle=False, phase="val")
    logger.info(f"Found {val_length} items in valid data")

