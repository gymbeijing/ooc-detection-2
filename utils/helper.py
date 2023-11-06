#!/usr/bin/env python
# coding: utf-8
# Stores repetitively used helper functions

import torch
import json
import re


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