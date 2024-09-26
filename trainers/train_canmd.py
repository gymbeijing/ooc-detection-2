"""
python -m trainers.train_canmd --few_shot_topic military
"""
import numpy as np
import random

import torch

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
# from dataloader import preprocess, tokenize, get_loader
from configs.config_CANMD import *
from tqdm import tqdm
import os
import logging
import torch.utils.data as data
from dataset.twitterCOMMsDataset import TwitterCOMMsDataset
from model.canmd import ContrastiveModel
from torch.autograd import Variable
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from utils.helper import accuracy_at_eer, compute_auc

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation(args, model, eval_dataloader):
    model.eval()
    eval_preds = []
    eval_labels = []
    eval_losses = []
    topic_label_list = []
    
    tqdm_dataloader = tqdm(eval_dataloader)
    for _, batch in enumerate(tqdm_dataloader):
        # batch = tuple(t.to(args.device) for t in batch)
        inputs_embeds, labels = batch["multimodal_emb"].to(device), batch["label"].to(device)
        inputs_embeds, labels = Variable(inputs_embeds), Variable(labels)
        outputs = model(inputs_embeds, labels)

        topic_labels = batch["domain_name"]
        topic_label_list += topic_labels

        loss = outputs["loss"]
        logits = outputs["logits"]
        eval_preds += torch.argmax(logits, dim=1).cpu().numpy().tolist()
        eval_labels += labels.cpu().numpy().tolist()
        eval_losses.append(loss.item())

        tqdm_dataloader.set_description('Eval bacc: {:.4f}, acc: {:.4f}, f1: {:.4f}, loss: {:.4f}'.format(
            balanced_accuracy_score(eval_labels, eval_preds),
            np.mean(np.array(eval_labels)==np.array(eval_preds)), 
            f1_score(eval_labels, eval_preds),
            np.mean(eval_losses)
        ))

    final_bacc = balanced_accuracy_score(eval_labels, eval_preds)
    final_acc = np.mean(np.array(eval_preds)==np.array(eval_labels))
    final_f1 = f1_score(eval_labels, eval_preds)
    final_precision = precision_score(eval_labels, eval_preds)
    final_recall = recall_score(eval_labels, eval_preds)

    inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if "covid" in topic_fullname]   # manually set
    cls_report = classification_report(np.array(eval_labels)[inds], np.array(eval_preds)[inds], digits=4, zero_division=0)
    auc_score = compute_auc(np.array(eval_labels)[inds], np.array(eval_preds)[inds])
    print(cls_report)
    print(auc_score)
    
    return final_bacc, final_acc, final_f1, final_precision, final_recall


def train(args):
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.output_dir:
        args.output_dir = 'canmd_output'
    export_root = os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir)
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    
    t_total = len(train_dataloader) * args.num_train_epochs   # total number of training steps

    model = ContrastiveModel(args)   #args?
    
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    global_step = 0
    print('***** Running training *****')
    print('Batch size = %d', args.batch_size)
    print('Num steps = %d', t_total)
    best_bacc, best_acc, best_f1, _, _ = evaluation(args, model, val_dataloader)
    for epoch in range(1, args.num_train_epochs+1):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc='Epoch {}'.format(epoch))):
            # batch = tuple(t.to(args.device) for t in batch)
            inputs_embeds, labels = batch["multimodal_emb"].to(device), batch["label"].to(device)
            inputs_embeds, labels = Variable(inputs_embeds), Variable(labels)
            outputs = model(inputs_embeds, labels)

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            global_step += 1

        eval_bacc, eval_acc, eval_f1, _, _ = evaluation(args, model, val_dataloader)
        if eval_bacc + eval_acc + eval_f1 >= best_bacc + best_acc + best_f1:
            best_bacc = eval_bacc
            best_acc = eval_acc
            best_f1 = eval_f1
            print(f"best_bacc: {best_bacc}, best_acc: {best_acc}, best_f1: {best_f1}")
            ### save the model ###
            torch.save(model.state_dict(), os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir, 'pretrained_model.ckpt'))


if __name__ == '__main__': 
    args.num_train_epochs = 5  # number of epoch can be chosen from 2 to 5
    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    train_data = TwitterCOMMsDataset(feather_path='./raw_data/train_completed_exist.feather',
                                     img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
                                     multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{args.base_model}_multimodal_embeds_train.pt',
                                     metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{args.base_model}_idx_to_image_path_train.json',
                                     few_shot_topic=args.few_shot_topic)  # took ~one hour to construct the dataset
    logger.info(f"Found {train_data.__len__()} items in training data")

    logger.info("Loading valid data")
    val_data = TwitterCOMMsDataset(feather_path='./raw_data/val_completed_exist.feather',
                                   img_dir=root_dir+'twitter-comms/images/val_images/val_tweet_image_ids',
                                   multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{args.base_model}_multimodal_embeds_valid.pt',
                                   metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{args.base_model}_multimodal_idx_to_image_path_valid.json',
                                #    few_shot_topic=args.few_shot_topic
                                   )
    logger.info(f"Found {val_data.__len__()} items in valid data")

    train_dataloader = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=args.batch_size)
    val_dataloader = data.DataLoader(val_data,
                                   shuffle=False,
                                   batch_size=args.batch_size)
    train(args)