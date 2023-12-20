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


def train(model: nn.Module, mlp: nn.Module, loss_type: str, optimizer, device: str, src_loader: DataLoader,
          tgt_loader: DataLoader, summary_writer: SummaryWriter, desc='Train', lambda_w=0.5):
    model.train()

    src_train_accuracy = 0
    tgt_train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0
    train_iteration = 0

    if len(src_loader) == len(tgt_loader):
        double_loader = enumerate(zip(src_loader, tgt_loader))
    elif len(src_loader) < len(tgt_loader):
        print("Src smaller than Tgt")
        double_loader = enumerate(zip(cycle(src_loader), tgt_loader))
    else:
        double_loader = enumerate(zip(src_loader, cycle(tgt_loader)))
    with tqdm(double_loader, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        torch.cuda.empty_cache()
        for i, (src_data, tgt_data) in loop:
            src_texts, src_masks, src_texts_perturb, src_masks_perturb, src_labels = src_data[0], src_data[1], src_data[
                2], src_data[3], src_data[4]
            src_texts, src_masks, src_labels = src_texts.to(device), src_masks.to(device), src_labels.to(device)
            src_texts_perturb, src_masks_perturb = src_texts_perturb.to(device), src_masks_perturb.to(device)
            batch_size = src_texts.shape[0]

            tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb, tgt_labels = tgt_data[0], tgt_data[1], tgt_data[
                2], tgt_data[3], tgt_data[4]
            tgt_texts, tgt_masks, tgt_labels = tgt_texts.to(device), tgt_masks.to(device), tgt_labels.to(device)
            tgt_texts_perturb, tgt_masks_perturb = tgt_texts_perturb.to(device), tgt_masks_perturb.to(device)

            optimizer.zero_grad()

            output_dic = model(src_texts, src_masks, src_texts_perturb, src_masks_perturb,
                               tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb,
                               src_labels, tgt_labels)

            loss = output_dic.total_loss

            loss.backward()

            optimizer.step()

            src_batch_accuracy = accuracy_sum(output_dic.src_logits, src_labels)
            src_train_accuracy += src_batch_accuracy
            tgt_batch_accuracy = accuracy_sum(output_dic.tgt_logits, tgt_labels)
            tgt_train_accuracy += tgt_batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), src_acc=src_train_accuracy / train_epoch_size,
                             tgt_acc=tgt_train_accuracy / train_epoch_size,
                             mmd=output_dic.mmd.item(), 
                             src_LCE_real=output_dic.src_ce_loss_real.item(),
                             src_LCE_perturb=output_dic.src_ce_loss_perturb.item())

    return {
        "train/src_accuracy": src_train_accuracy,
        "train/tgt_accuracy": tgt_train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


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

