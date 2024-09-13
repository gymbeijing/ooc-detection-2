#!/usr/bin/env python
# coding: utf-8
# Stores repetitively used helper functions

import torch
import json
import re
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score


def save_tensor(tensor, dest):
    torch.save(tensor, dest)
    return


def save_json(json_dict, dest):
    with open(dest, 'w', encoding='utf8') as fp:
        json.dump(json_dict, fp, indent=4, ensure_ascii=False, sort_keys=False)
    return


def remove_url(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", '', text)


def remove_punc(text):
    """Remove punctuation from a sample string"""
    return re.sub(r'[^\w\s]', '', text)


def load_tensor(filepath):
    tensor = torch.load(filepath)
    return tensor


def load_json(filepath):
    with open(filepath, 'r') as fp:
        json_data = json.load(fp)
    return json_data


def compute_eer(y_true, y_scores):
    # Compute false positive rates, true positive rates, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # False negative rate is 1 - TPR
    fnr = 1 - tpr

    # Find the point where FPR and FNR are closest
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    return eer, eer_threshold

def accuracy_at_eer(y_true, y_scores):
    # Compute EER and the corresponding threshold
    eer, eer_threshold = compute_eer(y_true, y_scores)

    # Classify data based on the EER threshold
    y_pred = (y_scores >= eer_threshold).astype(int)

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, eer, eer_threshold

def compute_auc(y_true, y_scores):
    auc_score = roc_auc_score(y_true, y_scores)

    return auc_score