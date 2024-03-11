from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data
import torch

CUSTOM_TEMPLATES = {
    "Twitter-COMMs": "a piece of news in {}."   # {domain}
}

class TwitterCOMMsUnimodalDataset(Dataset):
    def __init__(self, feather_path, img_dir, text_embeds_path, image_embeds_path, metadata_path, few_shot_topic=[], mode="val"):
        """
        Args:
            feather_path (string): Path to the {train|val}_completed_exist.feather file.
            img_dir (string): Directory containing the images
        """
        # self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.text_embeds = load_tensor(text_embeds_path)
        self.image_embeds = load_tensor(image_embeds_path)
        self.metadata = load_json(metadata_path)
        #
        # self.df['exists'] = self.df['filename'].apply(lambda filename: os.path.exists(os.path.join(img_dir, filename)))
        # delete_row = self.df[self.df["exists"] == False].index
        # self.df = self.df.drop(delete_row)
        # self.df = self.df.reset_index(drop=True)   # set index from 0 to len(df)-1, now index<->row number, i.e. df.iloc[row number]=df.iloc[index]

        self.df = pd.read_feather(feather_path)   # already drop the non-exists
        self.domain_map_to_idx = {"climate": 0, "covid": 1, "military": 2, "cross": 3}
        self.mode = mode

        assert len(self.df) == self.text_embeds.shape[0], \
            f"The number of news in self.df doesn't equal to number of tensor: {len(self.df)} vs {self.text_embeds.shape[0]}"

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

        # self.df['is_cross'] = self.df['topic'].apply(lambda topic: 'cross' in topic)
        # row_excluded = self.df[self.df["is_cross"] == True].index
        # self.row_kept = self.row_kept.difference(row_excluded)

    def __len__(self):
        return len(self.row_kept)

    def __getitem__(self, idx):
        row_number = self.row_kept[idx]
        item = self.df.iloc[row_number]

        img_filename = item['filename']
        topic = item['topic']
        falsified = int(item['falsified'])
        not_falsified = float(not item['falsified'])
        label = np.array(falsified)
        domain_name = topic.split('_')[0]
        domain_id = self.domain_map_to_idx[domain_name]   # turn string to ordinal
        difficulty = topic.split('_')[1]

        image_path = os.path.join(self.img_dir, img_filename)

        assert image_path == self.metadata[str(row_number)], f"Image path does not match with the metadata. {image_path} vs {self.metadata[str(row_number)]}"

        return {"text_emb": self.text_embeds[row_number],
                "image_emb": self.image_embeds[row_number],
                "topic": topic,
                "label": label,
                "domain_id": domain_id,
                "domain_name": domain_name,
                "difficulty": difficulty}
