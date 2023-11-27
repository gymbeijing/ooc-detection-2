#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import re
from nltk.tokenize import TweetTokenizer
import sys


# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

# Tokenizer
tt = TweetTokenizer()

# # Environment variable
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def remove_url(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", '', text)


def remove_punc(text):
    """Remove punctuation from a sample string"""
    return re.sub(r'[^\w\s]', '', text)


def preprocess(text):
    preprocessed_text = ' '.join(tt.tokenize(text))
    preprocessed_text = remove_punc(remove_url(preprocessed_text))
    return preprocessed_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", type=str, required=True, help="{climate, covid, military}")

    args = p.parse_args()
    return args


if __name__ == "__main__":
    param_dict = {"climate": {"min_cluster_size": 400, "cluster_selection_epsilon": 0.56},
                  "covid": {"min_cluster_size": 800, "cluster_selection_epsilon": 0.65},
                  "military": {"min_cluster_size": 100, "cluster_selection_epsilon": 0.6}}
    # Parse arguments
    args = parse_args()
    topic = args.topic

    logging.info("Prepare topic model")
    param = param_dict[topic]
    hdbscan_model = HDBSCAN(min_cluster_size=param["min_cluster_size"], metric='euclidean',
                            cluster_selection_method='eom', cluster_selection_epsilon=param["cluster_selection_epsilon"],
                            prediction_data=True)
    umap_model = UMAP(n_neighbors=10, n_components=20, min_dist=0.0, metric='cosine')
    topic_model = BERTopic(hdbscan_model=hdbscan_model, umap_model=umap_model)

    logging.info("Read .feather file")
    train_feather_path = '../raw_data/train_completed_exist.feather'
    train_df = pd.read_feather(train_feather_path)  # already drop the non-exists

    val_feather_path = '../raw_data/val_completed_exist.feather'
    val_df = pd.read_feather(val_feather_path)  # already drop the non-exists

    df = pd.concat([train_df, val_df])

    logging.info("Prepare documents")
    df['preprocessed_full_text'] = df['full_text'].apply(lambda t: preprocess(t))
    temp_df = df[df['topic'].str.contains(topic)]
    tweet_texts = temp_df['preprocessed_full_text'].unique().tolist()

    logging.info(f"Model fit and transform documents(len={len(tweet_texts)}, size={sys.getsizeof(tweet_texts)})")
    topics, probs = topic_model.fit_transform(tweet_texts)
    # topic_model = topic_model.fit(tweet_texts)
    logging.info("Get topic info")
    topic_info = topic_model.get_topic_info()[["Topic", "Count", "Name"]]
    print(topic_info)

    SAMPLE_SIZE = 1000
    for i in range(SAMPLE_SIZE):
        print(f"[News caption]: {tweet_texts[i]}. [Topic label]: {topics[i]}")



