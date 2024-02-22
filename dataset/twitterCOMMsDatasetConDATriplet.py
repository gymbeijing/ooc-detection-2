from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler, RandomSampler


class TwitterCOMMsDatasetConDATriplet(Dataset):
    def __init__(self, triplet_feather_path, img_dir,
                 original_multimodal_embeds_path, positive_multimodal_embeds_path, negative_multimodal_embeds_path,
                 metadata_path, few_shot_topic=None, mode="train/val"):
        """
        Args:
            feather_path (string): Path to the {train|val}_completed_exist.feather file.
            img_dir (string): Directory containing the images
        """
        # self.df = pd.read_csv(csv_path, index_col=0)
        if few_shot_topic is None:
            few_shot_topic = []
        self.img_dir = img_dir
        self.original_multimodal_embeds = load_tensor(original_multimodal_embeds_path)
        self.positive_multimodal_embeds = load_tensor(positive_multimodal_embeds_path)
        self.negative_multimodal_embeds = load_tensor(negative_multimodal_embeds_path)
        self.metadata = load_json(metadata_path)

        self.df = pd.read_feather(triplet_feather_path)  # already drop the non-exists
        self.domain_map_to_idx = {"climate": 0, "covid": 1, "military": 2}
        self.mode = mode

        assert len(self.df) == self.original_multimodal_embeds.shape[0], \
            "The number of news in self.df doesn't equal to number of tensor"
        assert self.original_multimodal_embeds.shape[0] == self.positive_multimodal_embeds.shape[0], \
            "The number original news items doesn't equal to the number of positive news items"
        assert self.original_multimodal_embeds.shape[0] == self.negative_multimodal_embeds.shape[0], \
            "The number original news items doesn't equal to the number of negative news items"

        # if not excluding any topic
        self.row_kept = self.df.index

        # Exclude few shot topic news
        if 'military' in few_shot_topic:
            self.df['is_military'] = self.df['topic'].apply(lambda topic: 'military' in topic)
            row_excluded = self.df[self.df["is_military"] == True].index
            self.row_kept = self.row_kept.difference(row_excluded)

        if 'covid' in few_shot_topic:
            self.df['is_covid'] = self.df['topic'].apply(lambda topic: 'covid' in topic)
            row_excluded = self.df[self.df["is_covid"] == True].index
            self.row_kept = self.row_kept.difference(row_excluded)

        if 'climate' in few_shot_topic:
            self.df['is_climate'] = self.df['topic'].apply(lambda topic: 'climate' in topic)
            row_excluded = self.df[self.df["is_climate"] == True].index
            self.row_kept = self.row_kept.difference(row_excluded)

        # Exclude cross topic news
        self.df['is_cross'] = self.df['topic'].apply(lambda topic: 'cross' in topic)
        row_excluded = self.df[self.df["is_cross"] == True].index
        self.row_kept = self.row_kept.difference(row_excluded)

    def __len__(self):
        return len(self.row_kept)

    def __getitem__(self, idx):
        row_number = self.row_kept[idx]
        item = self.df.iloc[row_number]
        falsified = int(item['falsified'])
        original_label = np.array(falsified)

        img_filename = item['filename']   # img_filename corresponding to the original
        image_path = os.path.join(self.img_dir, img_filename)

        if self.mode != "toy":
            assert image_path == self.metadata[str(row_number)], "Image path does not match with the metadata"

        original_multimodal_emb = self.original_multimodal_embeds[row_number]
        positive_multimodal_emb = self.positive_multimodal_embeds[row_number]
        negative_multimodal_emb = self.negative_multimodal_embeds[row_number]

        return {"original_multimodal_emb": original_multimodal_emb,
                "positive_multimodal_emb": positive_multimodal_emb,
                "negative_multimodal_emb": negative_multimodal_emb,
                "original_label": original_label}


def get_dataloader(cfg, few_shot_topic, shuffle, phase='val'):
    root_dir = '/import/network-temp/yimengg/data/'
    if phase == 'train':
        # train_data = TwitterCOMMsDataset(feather_path='../raw_data/train_completed_exist.feather',
        #                                  img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
        #                                  multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_train.pt',
        #                                  metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_idx_to_image_path_train.json',
        #                                  few_shot_topic=[cfg.args.few_shot_topic])  # took ~one hour to construct the dataset
        # train_iterator = data.DataLoader(train_data,
        #                                  shuffle=shuffle,
        #                                  batch_size=cfg.args.batch_size)
        # return train_iterator, train_data.__len__()
        toy_dataset = TwitterCOMMsDatasetConDATriplet(triplet_feather_path='./raw_data/toy_completed_exist_triplet.feather',
                                               img_dir=root_dir + 'twitter-comms/train/images/train_image_ids',
                                               original_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_toy_original.pt',
                                            #    positive_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_toy_positive.pt',
                                               positive_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_toy_GaussianBlur.pt',
                                               negative_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_toy_negative.pt',
                                            #    augmented_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_mini_toy_rephrased.pt',
                                               metadata_path=root_dir + f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_multimodal_idx_to_image_path_toy_original.json',
                                               few_shot_topic=few_shot_topic,
                                               mode="toy"
                                               )
        sampler = RandomSampler(toy_dataset)   # randomly sampling, order determined by torch.manual_seed()
        # batch_sampler = BatchSampler(range(len(toy_dataset)), batch_size=cfg.args.batch_size, drop_last=True)   # not random, in its original order
        batch_sampler = BatchSampler(sampler, batch_size=cfg.args.batch_size, drop_last=True)
        # toy_iterator = data.DataLoader(toy_dataset,
        #                                shuffle=shuffle,
        #                                batch_size=cfg.args.batch_size)
        toy_iterator = data.DataLoader(toy_dataset, batch_sampler=batch_sampler)   # cannot be shuffled
        return toy_iterator, toy_dataset.__len__()
    else:  # phase=='val'
        val_dataset = TwitterCOMMsDatasetConDATriplet(triplet_feather_path='./raw_data/val_completed_exist.feather',
                                               img_dir=root_dir + 'twitter-comms/images/val_images/val_tweet_image_ids',
                                               original_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_valid.pt',
                                               positive_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_valid.pt',
                                               negative_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_valid.pt',
                                               metadata_path=root_dir + f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_multimodal_idx_to_image_path_valid.json',
                                               few_shot_topic=few_shot_topic,
                                               )
        val_iterator = data.DataLoader(val_dataset,
                                       shuffle=shuffle,
                                       batch_size=cfg.args.batch_size)
        return val_iterator, val_dataset.__len__()
