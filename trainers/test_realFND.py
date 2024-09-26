import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch import nn
from torch import optim
from torch.autograd import Variable
from tqdm.auto import tqdm

from utils.helper import save_tensor, load_tensor, load_json
from dataset.twitterCOMMsDataset import TwitterCOMMsDataset

from sklearn.metrics import f1_score
import numpy as np

import logging
import argparse
from model.realFND import FakeNewsClassifier, Policy
from configs.configRealFND import ConfigRealFND
from sklearn.metrics import f1_score, classification_report
from utils.helper import accuracy_at_eer, compute_auc


"""
Remember to change topic list to the few_shot_topic
"""

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


def test(policy, net, iterator, criterion, device):
    policy.eval()
    net.eval()
    softmax = nn.Softmax(dim=1)
    T = 20
    sigma = 0.01

    with torch.no_grad():
        total_loss = 0
        num_correct = dict()
        num_total = dict()
        f1 = dict()
        cls_report = dict()
        auc_score = dict()
        num_correct["all"] = 0
        num_total["all"] = 0
        num_correct["climate"] = 0
        num_total["climate"] = 0
        num_correct["covid"] = 0
        num_total["covid"] = 0
        num_correct["military"] = 0
        num_total["military"] = 0

        y_pred_list = []
        y_true_list = []
        f1["climate"] = 0
        f1["covid"] = 0
        f1["military"] = 0
        topic_label_list = []
        for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            # Get the domain-invariant feature
            for t in range(T):
                probs = policy(inputs)
                action = probs.argmax(dim=1)

                for r_idx in range(action.shape[0]):
                    if action[r_idx] < 768:
                        c_idx = action % 768
                        inputs[r_idx, c_idx] += sigma
                    else:
                        c_idx = action % 768
                        inputs[r_idx, c_idx] -= sigma

            # Get the output predictions
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

            # Compute total loss of the current epoch
            total_loss += loss.item()

            # Compute the number of correct predictions
            # top_pred = torch.zeros_like(labels)
            # y_preds = softmax(y_preds)
            # top_pred[y_preds[:, 1] >= threshold] = 1
            y = labels.cpu()
            cur_batch_size = y.shape[0]
            # top_pred = top_pred.cpu().view(cur_batch_size)
            
            top_pred = y_preds.argmax(dim=1).cpu()

            y_pred_list.append(top_pred.cpu())  # [bs, 2]?
            y_true_list.append(y.cpu())  # [bs, 2]?

            # Compute overall performance
            num_correct["all"] += sum(top_pred == y).item()
            num_total["all"] += cur_batch_size

            # Compute topic-wise performance
            topic_labels = batch["topic"]
            topic_label_list += topic_labels
            topic_list = ["climate"]

            for topic in topic_list:
                inds = []
                for ind, topic_label in enumerate(topic_labels):
                    if topic in topic_label:
                        inds.append(ind)
                num_total[topic] += len(inds)
                inds = np.array(inds)
                num_correct[topic] += sum(top_pred[inds] == y[inds])

            if i % 1000 == 0:
                logger.info("%d-th batch: Testing accuracy %.4f, loss: %.4f" % (
                    i, num_correct["all"] / num_total["all"], total_loss / num_total["all"]))
                
        for topic in topic_list:
            inds = [idx for idx, topic_fullname in enumerate(topic_label_list) if topic in topic_fullname]
            f1[topic] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')
            print(f"f1: {f1[topic]}")
            cls_report[topic] = classification_report(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], digits=4, zero_division=0)
            print(f"classification report: {cls_report[topic]}")
            auc_score[topic] = compute_auc(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds])
            print(f"auc score: {auc_score[topic]}")

        logger.info("Overall testing accuracy %.4f, climate testing accuracy %.4f, covid testing accuracy %.4f, "
                    "military testing accuracy %.4f, loss: %.4f" % (num_correct["all"] / num_total["all"],
                                                                    num_correct["climate"] / (num_total["climate"]+0.0001),
                                                                    num_correct["covid"] / (num_total["covid"]+0.0001),
                                                                    num_correct["military"] / (num_total["military"]+0.0001),
                                                                    total_loss / num_total["all"]))
        print(f"f1: {f1}")

    return torch.cat(y_pred_list, dim=0), torch.cat(y_true_list, dim=0)


if __name__ == '__main__':

    base_model = 'blip-2'
    BATCH_SIZE = 256


    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading valid data")
    val_data = TwitterCOMMsDataset(feather_path='./raw_data/val_completed_exist.feather',
                                   img_dir=root_dir+'twitter-comms/images/val_images/val_tweet_image_ids',
                                   multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{base_model}_multimodal_embeds_valid.pt',
                                   metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{base_model}_multimodal_idx_to_image_path_valid.json',
                                   few_shot_topic=['military', 'covid']
                                   )
    logger.info(f"Found {val_data.__len__()} items in valid data")

    val_iterator = data.DataLoader(val_data,
                                   shuffle=False,
                                   batch_size=BATCH_SIZE)


    fake_news_classifier = FakeNewsClassifier()
    fake_news_classifier.cuda()
    fake_news_classifier.eval()
    fake_news_classifier.load_state_dict(torch.load(os.path.join('./', 'real_fnd_output', 'fake_news_classifier_climate.ckpt')))

    policy_network = Policy(768 * 2)
    policy_network.cuda()
    policy_network.eval()
    policy_network.load_state_dict(torch.load(os.path.join('./', 'real_fnd_output', 'policy_network_climate.ckpt')))

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    test(policy_network, fake_news_classifier, val_iterator, criterion, device)