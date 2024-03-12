import torch

from torch import nn
import pandas as pd
import os

from tqdm.auto import tqdm

from sklearn.metrics import classification_report
import json

from torch.utils.data import Dataset
import torch.utils.data as data
from torch import optim
from torch.autograd import Variable
import numpy as np

import logging
import argparse
from sklearn.metrics import f1_score
import numpy as np

from dataset.newsCLIPpingsDataset import get_dataloader_2

"""
python -m trainers.blip2_full_train_newsCLIPpings --bs 256 --epochs 10 --target_agency bbc
python -m trainers.blip2_full_train_newsCLIPpings --bs 256 --epochs 10 --target_domain uk-news
"""

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Net(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(Net, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        out = self.fc(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def train(train_iterator, val_iterator, device):
    net = Net(768)   # blip-2 multimodal: 768, blip-2 unimodal: 512
    net.cuda()
    net.train()
    net.weight_init(mean=0, std=0.02)

    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        num_correct = 0
        num_total = 0
        for i, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            net.zero_grad()
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, top_pred = y_preds.topk(1, 1)
            y = labels.cpu()
            batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(batch_size)

            num_correct += sum(top_pred == y).item()
            num_total += batch_size

            # if i % 1000 == 0:
            #     logger.info("Epoch [%d/%d] %d-th batch: Training accuracy: %.4f, loss: %.4f" % (
            #     epoch + 1, EPOCHS, i, num_correct / num_total, total_loss / num_total))

        logger.info("Epoch [%d/%d]: Training accuracy: %.4f, loss: %.4f" % (
        epoch + 1, EPOCHS, num_correct / num_total, total_loss / num_total))

        test(net, val_iterator, criterion, device)

    return net


def test(net, iterator, criterion, device):
    net.eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        # total_loss = 0
        # num_correct = 0
        # num_total = 0
        # for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
        #     inputs = batch["multimodal_emb"].to(device)
        #     labels = batch["label"].to(device)
        #     inputs, labels = Variable(inputs), Variable(labels)

        #     y_preds = net(inputs)
        #     loss = criterion(y_preds, labels)

        #     total_loss += loss.item()

        #     _, top_pred = y_preds.topk(1, 1)
        #     y = labels.cpu()
        #     batch_size = y.shape[0]
        #     top_pred = top_pred.cpu().view(batch_size)

        #     num_correct += sum(top_pred == y).item()
        #     num_total += batch_size

        #     # if i % 1000 == 0:
        #     #     logger.info("%d-th batch: Testing accuracy %.4f, loss: %.4f" % (
        #     #     i, num_correct / num_total, total_loss / num_total))

        # logger.info("Testing accuracy %.4f, loss: %.4f" % (num_correct / num_total, total_loss / num_total))

        total_loss = 0
        num_correct = dict()
        num_total = dict()
        targets = []
        outputs = []
        domain_labels_list = []
        # topics = {
        #     # "arts_culture": ["arts_culture", "culture", "film", "music", "artsanddesign"],
        #     # "world": ["world"], 
        #     # "international_relations": ["international_relations"], 
        #     "law_crime": ["law_crime", "law"],
        #     # "science_technology": ["science_technology", "technology", "science"],
        #     # "football": ["football", "sport", "sports"],
        #     # "politics_elections": ["politics_elections", "politics"],
        #     # "business": ["business", "business_economy"],
        #     "media": ["media"], 
        #     # "environment": ["environment"],
        #     # "fashion": ["fashion"],
        #     # "education": ['education'],
        #     # "money": ['money'],
        #     # "travel": ['travel'],
        #     "education": ['education'],
        #     "books": ['books'],
        #     "society": ['society'],
        #     "edinburgh": ['edinburgh'],
        #     "leeds": ["leeds"],
        #     "stage": ["stage"],
        #     }
        num_correct["all"] = 0
        num_total["all"] = 0
        num_correct["bbc"] = 0
        num_total["bbc"] = 0
        num_correct["guardian"] = 0
        num_total["guardian"] = 0
        num_correct["usa_today"] = 0
        num_total["usa_today"] = 0
        num_correct["washington_post"] = 0
        num_total["washington_post"] = 0
        # for topic in topics.keys():
        #     num_correct[topic] = 0
        #     num_total[topic] = 0

        y_pred_list = []
        y_true_list = []
        for i, batch in tqdm(enumerate(iterator, 0), desc='iterations'):
            inputs = batch["multimodal_emb"].to(device)
            labels = batch["label"].to(device)
            domain_labels = batch["news_source"]
            domain_labels_list += list(domain_labels)
            inputs, labels = Variable(inputs), Variable(labels)

            # Get the output predictions
            y_preds = net(inputs)
            loss = criterion(y_preds, labels)

            # Compute total loss of the current epoch
            total_loss += loss.item()

            # Compute the number of correct predictions
            top_pred = torch.zeros_like(labels)
            y_preds = softmax(y_preds)
            top_pred[y_preds[:, 1] >= 0.5] = 1
            y = labels.cpu()
            cur_batch_size = y.shape[0]
            top_pred = top_pred.cpu().view(cur_batch_size)

            y_pred_list.append(top_pred)  # [bs, 2]?
            y_true_list.append(y.cpu())  # [bs, 2]?

            # Compute overall performance
            num_correct["all"] += sum(top_pred == y).item()
            num_total["all"] += cur_batch_size

            # Compute topic-wise performance
            topic_labels = batch["news_source"]
            topic_list = ["bbc", "guardian", "usa_today", "washington_post"]

            # topic_labels = batch["topic"]

            for topic in topic_list:
                inds = []
                for ind, topic_label in enumerate(topic_labels):
                    if topic in topic_label:
                        inds.append(ind)
                num_total[topic] += len(inds)
                inds = np.array(inds)
                num_correct[topic] += sum(top_pred[inds] == y[inds])

            # for topic in topics.keys():
            #     inds = []
            #     for ind, topic_label in enumerate(topic_labels):
            #         if topic_label in topics[topic]:
            #             inds.append(ind)
            #     num_total[topic] += len(inds)
            #     inds = np.array(inds)
            #     num_correct[topic] += sum(top_pred[inds] == y[inds])
        f1 = dict()
        domain_list = ['bbc', 'guardian', 'usa_today', 'washington_post']
        for domain in domain_list:
            inds = [idx for idx, domain_name in enumerate(domain_labels_list) if domain_name == domain]
            f1[domain] = f1_score(np.concatenate(y_true_list)[inds], np.concatenate(y_pred_list)[inds], average='macro')
        print(f1)
        logger.info("Overall testing accuracy %.4f, bbc testing accuracy %.4f, guardian testing accuracy %.4f, "
                    "usa_today testing accuracy %.4f, washington_post testing accuracy %.4f, loss: %.4f" % (num_correct["all"] / num_total["all"],
                                                                    num_correct["bbc"] / num_total["bbc"],
                                                                    num_correct["guardian"] / num_total["guardian"],
                                                                    num_correct["usa_today"] / num_total["usa_today"],
                                                                    num_correct["washington_post"] / num_total["washington_post"],
                                                                    total_loss / num_total["all"]))
                
        # logger.info("Overall testing accuracy %.4f, loss: %.4f" % (num_correct["all"] / num_total["all"], total_loss / num_total["all"]))
        # for topic in topics.keys():
        #     logger.info(f"{topic} testing accuracy %.4f" % (num_correct[topic] / (num_total[topic] + 0.000001)))

    return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, required=True, help="batch size")
    p.add_argument("--epochs", type=int, required=True, help="number of training epochs")
    p.add_argument("--target_agency", type=str, required=True, help="{'bbc', 'guardian', 'usa_today', 'washington_post'}")
    # p.add_argument("--target_domain", type=str, required=True, help="topics")

    args = p.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    BATCH_SIZE = args.bs
    EPOCHS = args.epochs
    target_agency = args.target_agency

    # Set up device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    root_dir = '/import/network-temp/yimengg/data/'

    logger.info("Loading training data")
    train_iterator, train_len = get_dataloader_2(target_agency=target_agency, shuffle=True, batch_size=BATCH_SIZE, phase='train')
    # train_iterator, train_len = get_dataloader_2(target_agency=target_domain, shuffle=True, batch_size=BATCH_SIZE, phase='train')
    logger.info(f"Found {train_len} items in the training dataset")

    logger.info("Loading testing data")
    test_iterator, test_len = get_dataloader_2(target_agency=target_agency, shuffle=False, batch_size=BATCH_SIZE, phase='test')
    logger.info(f"Found {test_len} items in valid data")

    logger.info("Start training the model")

    net = train(train_iterator, test_iterator, device)