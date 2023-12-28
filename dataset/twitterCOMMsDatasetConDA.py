from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data


class TwitterCOMMsDatasetConDA(Dataset):
    def __init__(self, feather_path, augmented_feather_path, img_dir,
                 multimodal_embeds_path, augmented_multimodal_embeds_path,
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
        self.multimodal_embeds = load_tensor(multimodal_embeds_path)
        self.augmented_multimodal_embeds = load_tensor(augmented_multimodal_embeds_path)
        self.metadata = load_json(metadata_path)
        #
        # self.df['exists'] = self.df['filename'].apply(lambda filename: os.path.exists(os.path.join(img_dir, filename)))
        # delete_row = self.df[self.df["exists"] == False].index
        # self.df = self.df.drop(delete_row)
        # self.df = self.df.reset_index(drop=True)   # set index from 0 to len(df)-1, now index<->row number, i.e. df.iloc[row number]=df.iloc[index]

        self.df = pd.read_feather(feather_path)  # already drop the non-exists
        self.augmented_df = pd.read_feather(augmented_feather_path)  # already drop the non-exists?
        self.domain_map_to_idx = {"climate": 0, "covid": 1, "military": 2}
        self.mode = mode

        assert len(self.df) == self.multimodal_embeds.shape[0], \
            "The number of news in self.df isn't equal to number of tensor"
        assert len(self.augmented_df) == self.augmented_multimodal_embeds.shape[0], \
            "The number of news in self.augmented_df isn't equal to number of tensor"
        assert len(self.df) == len(self.augmented_df), \
            "The number of news in self.df isn't equal to that in self.augmented_df"

        # if not excluding any topic
        self.row_kept = self.df.index

        # Remove news of the few shot topic
        if 'military' in few_shot_topic:
            self.df['is_military'] = self.df['topic'].apply(lambda topic: 'military' in topic)
            row_excluded = self.df[self.df["is_military"] == True].index
            # row_all = self.df.index
            # self.row_kept = row_all.difference(row_excluded)
            self.row_kept = self.row_kept.difference(row_excluded)

        if 'covid' in few_shot_topic:
            self.df['is_covid'] = self.df['topic'].apply(lambda topic: 'covid' in topic)
            row_excluded = self.df[self.df["is_covid"] == True].index
            # row_all = self.df.index
            # self.row_kept = row_all.difference(row_excluded)
            self.row_kept = self.row_kept.difference(row_excluded)

        if 'climate' in few_shot_topic:
            self.df['is_climate'] = self.df['topic'].apply(lambda topic: 'climate' in topic)
            row_excluded = self.df[self.df["is_climate"] == True].index
            # row_all = self.df.index
            # self.row_kept = row_all.difference(row_excluded)
            self.row_kept = self.row_kept.difference(row_excluded)

        self.df['is_cross'] = self.df['topic'].apply(lambda topic: 'cross' in topic)
        row_excluded = self.df[self.df["is_cross"] == True].index
        self.row_kept = self.row_kept.difference(row_excluded)

    def __len__(self):
        return len(self.row_kept)

    def __getitem__(self, idx):
        row_number = self.row_kept[idx]
        item = self.df.iloc[row_number]
        falsified = int(item['falsified'])
        label = np.array(falsified)

        img_filename = item['filename']
        image_path = os.path.join(self.img_dir, img_filename)

        if self.mode != "toy":
            assert image_path == self.metadata[str(row_number)], "Image path does not match with the metadata"

        multimodal_emb = self.multimodal_embeds[row_number]
        augmented_multimodal_emb = self.augmented_multimodal_embeds[row_number]

        return {"multimodal_emb": multimodal_emb,
                "augmented_multimodal_emb": augmented_multimodal_emb,
                "label": label}


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
        toy_data = TwitterCOMMsDatasetConDA(feather_path='./raw_data/toy_completed_exist.feather',
                                            augmented_feather_path='./raw_data/toy_completed_exist_augmented.feather',
                                            img_dir=root_dir + 'twitter-comms/train/images/train_image_ids',
                                            multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_toy.pt',
                                            augmented_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_toy_augmented.pt',
                                            metadata_path=root_dir + f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_idx_to_image_path_train.json',
                                            few_shot_topic=few_shot_topic,
                                            mode="toy"
                                            )
        toy_iterator = data.DataLoader(toy_data,
                                       shuffle=shuffle,
                                       batch_size=cfg.args.batch_size)
        return toy_iterator, toy_data.__len__()
    else:  # phase=='val'
        val_data = TwitterCOMMsDatasetConDA(feather_path='./raw_data/val_completed_exist.feather',
                                            augmented_feather_path='./raw_data/val_completed_exist.feather',
                                            img_dir=root_dir + 'twitter-comms/images/val_images/val_tweet_image_ids',
                                            multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_valid.pt',
                                            augmented_multimodal_embeds_path=root_dir + f'twitter-comms/processed_data/tensor/{cfg.args.base_model}_multimodal_embeds_valid.pt',
                                            metadata_path=root_dir + f'twitter-comms/processed_data/metadata/{cfg.args.base_model}_multimodal_idx_to_image_path_valid.json',
                                            few_shot_topic=few_shot_topic,
                                            )
        val_iterator = data.DataLoader(val_data,
                                       shuffle=shuffle,
                                       batch_size=cfg.args.batch_size)
        return val_iterator, val_data.__len__()
