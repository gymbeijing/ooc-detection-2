"""
python -m trainers.adapt_canmdNews --few_shot_topic bbc,guardian
"""
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

# from dataloader import preprocess, tokenize, get_loader
from configs.config_CANMD import *
from tqdm import tqdm
import json
import os
import logging
import torch.utils.data as data
from dataset.newsCLIPpingsDataset import get_dataloader_2
from model.canmd import ContrastiveModel
from torch.autograd import Variable

from torch.utils.data import DataLoader, RandomSampler
from utils.utils_canmd import *

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
    
    tqdm_dataloader = tqdm(eval_dataloader)
    for _, batch in enumerate(tqdm_dataloader):
        # batch = tuple(t.to(args.device) for t in batch)
        
        inputs_embeds, labels = batch["multimodal_emb"].to(device), batch["label"].to(device)
        inputs_embeds, labels = Variable(inputs_embeds), Variable(labels)
        outputs = model(inputs_embeds, labels)

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
    
    return final_bacc, final_acc, final_f1, final_precision, final_recall


def adapt(args):
    def sample_source_batch(args, target_labels, source_dataset, source_label_dict, source_pointer):
        ### to be modified ###
        # sampling source batch
        output_idx = []
        for label in target_labels.tolist():
            next_idx = source_pointer[label] % len(source_label_dict[label])   # random
            output_idx.append(source_label_dict[label][next_idx])
            source_pointer[label] += 1

        # preparing inputs
        embs, labels = [], []
        for idx in output_idx:
            embs.append(source_dataset[idx]["multimodal_emb"])
            # print(type(source_dataset[idx]["label"]))
            labels.append(torch.tensor(source_dataset[idx]["label"]))
            # all_input_ids.append(source_dataset[idx][0].unsqueeze(0))
            # if 'roberta' not in args.lm_model:
            #     all_token_type_ids.append(source_dataset[idx][1].unsqueeze(0))
            # all_attention_mask.append(source_dataset[idx][-2].unsqueeze(0))
            # labels.append(source_dataset[idx][-1].unsqueeze(0))
        
        # all_input_ids = torch.vstack(all_input_ids)
        # if 'roberta' not in args.lm_model:
        #     all_token_type_ids = torch.vstack(all_token_type_ids)
        # all_attention_mask = torch.vstack(all_attention_mask)
        # labels = torch.vstack(labels).squeeze()

        embs = torch.stack(embs, dim=0)
        labels = torch.tensor(labels)

        # if 'roberta' not in args.lm_model:
        #     return all_input_ids, all_token_type_ids, all_attention_mask, labels
        # else:
        #     return all_input_ids, all_attention_mask, labels
        return embs, labels
        ######
    
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.output_dir:
        args.output_dir = 'canmd_output'
    export_root = os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir)
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    
    model = ContrastiveModel(args)
    model.load_state_dict(torch.load(os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir, f'pretrained_model_news_{args.few_shot_topic}.ckpt')))

    source_pointer = [0] * 2
    source_label_dict = {0: [], 1: []}
    # data, labels = get_dataset(args, 'train', args.source_data_type, args.source_data_path).load_dataset()
    labels = []
    for idx in tqdm(range(len(src_train_data))):
        # print(src_train_data[idx]["label"])
        labels.append(int(src_train_data[idx]["label"]))
    for idx, label in enumerate(labels):
        source_label_dict[label].append(idx)
    for key in source_label_dict.keys():
        random.shuffle(source_label_dict[key])
    
    # inputs = preprocess(args, data)
    # all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, inputs, tokenizer)
    # if 'roberta' in args.lm_model:
    #     source_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, torch.tensor(labels))
    # else:
    #     source_dataset = torch.utils.data.TensorDataset(all_input_ids, all_token_type_ids, \
    #                         all_attention_mask, torch.tensor(labels))
    # source_dataloader = torch.utils.data.DataLoader(
    #     source_dataset, sampler=RandomSampler(source_dataset), batch_size=args.train_batchsize)
    
    # val_dataloader = get_loader(args, mode='target_val', tokenizer=tokenizer)
    # test_dataloader = get_loader(args, mode='target_test', tokenizer=tokenizer)
    t_total = len(source_dataloader) * args.num_train_epochs  # notice t_total is not be accurate
    
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    global_step = 0
    print('***** Running adaptation *****')
    print('Batch size = {}'.format(args.batch_size))
    print('Num steps = {}'.format(t_total))
    best_bacc, best_acc, best_f1, _, _ = evaluation(args, model, val_dataloader)
    for epoch in range(1, args.num_train_epochs+1):
        ### Get the psuedo labeled data ###
        filtered_data, pseudolabels = get_corrected_psuedolabels_model(
                                        args, model, device=device,
                                        conf_threshold=args.conf_threshold)
        ### Prepare the target data with the pseudo label ###
        target_dataset = torch.utils.data.TensorDataset(
                filtered_data,
                torch.tensor(pseudolabels))
        
        target_dataloader = DataLoader(
                target_dataset,  
                sampler=RandomSampler(target_dataset),
                batch_size=args.batch_size)
        ######################################

        model.train()
        for step, batch in enumerate(tqdm(target_dataloader, desc='Epoch {}'.format(epoch))):
            ### batch data ###
            source_batch = sample_source_batch(args, batch[-1], src_train_data, source_label_dict, source_pointer)
            # source_batch = tuple(t.to(args.device) for t in source_batch)
            target_batch = tuple(t.to(args.device) for t in batch)

            src_emb, src_labels = source_batch[0].to(device), source_batch[1].to(device)
            src_emb, src_labels = Variable(src_emb), Variable(src_labels)
            
            tgt_emb, tgt_labels = target_batch
            outputs = model.forward_ours(
                src_emb, tgt_emb, src_labels, tgt_labels,
                alpha=args.alpha)
            ###

            loss = outputs['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            global_step += 1

        eval_bacc, eval_acc, eval_f1, _, _ = evaluation(args, model, val_dataloader)
        if eval_bacc + eval_acc + eval_f1 >= best_bacc + best_acc + best_f1:  # use sum to validate model
            best_bacc = eval_bacc
            best_acc = eval_acc
            best_f1 = eval_f1
            print(f"best_bacc: {best_bacc}, best_acc: {best_acc}, best_f1: {best_f1}")



if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)
    print(args)
    root_dir = '/import/network-temp/yimengg/data/'
    logger.info("Loading training data")
    src_train_data, source_dataloader, src_train_len = get_dataloader_2(target_agency=args.few_shot_topic, shuffle=True, batch_size=args.batch_size, phase='train')
    logger.info(f"Found {src_train_len} items in training data")

    topic_list = ["bbc", "guardian", "usa_today", "washington_post"]
    for topic in args.few_shot_topic.split(","):
        topic_list.remove(topic)
    logger.info("Loading valid data")
    val_dataloader, test_len = get_dataloader_2(target_agency="bbc,guardian,usa_today", shuffle=False, batch_size=args.batch_size, phase='test')
    logger.info(f"Found {test_len} items in valid data")

    adapt(args)