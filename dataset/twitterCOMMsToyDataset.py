from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data
import argparse
from tqdm.auto import tqdm


class TwitterCOMMsToyDataset(Dataset):
    def __init__(self, feather_path, img_dir, multimodal_embeds_path, metadata_path, size=30000):
        """
        Args:
            feather_path (string): Path to the {train|val}_completed_exist.feather file.
            img_dir (string): Directory containing the images
        """
        # self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.multimodal_embeds = load_tensor(multimodal_embeds_path)
        self.metadata = load_json(metadata_path)
        #
        # self.df['exists'] = self.df['filename'].apply(lambda filename: os.path.exists(os.path.join(img_dir, filename)))
        # delete_row = self.df[self.df["exists"] == False].index
        # self.df = self.df.drop(delete_row)
        # self.df = self.df.reset_index(drop=True)   # set index from 0 to len(df)-1, now index<->row number, i.e. df.iloc[row number]=df.iloc[index]

        self.df = pd.read_feather(feather_path)[:256]   # already drop the non-exists
        self.domain_map_to_idx = {"climate": 0, "covid": 1, "military": 2}

        assert len(self.df) == self.multimodal_embeds.shape[0], \
            "The number of news in self.df isn't equal to number of tensor"

        # if not excluding any topic
        self.row_kept = self.df.index

        # Randomly sample n=size number of news
        self.row_kept_sampled = self.row_kept.sample(n=size)

    def __len__(self):
        return len(self.row_kept_sampled)

    def __getitem__(self, idx):
        row_number = self.row_kept_sampled[idx]
        item = self.df.iloc[row_number]
        multimodal_emb = self.multimodal_embeds[row_number]

        return {"item": item,
                "multimodal_emb": multimodal_emb}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, required=True, help="batch size")
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
    p.add_argument("--toy_dataset_size", type=int, required=True, help="size of the toy dataset")

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    batch_size = args.bs
    base_model = args.base_model
    toy_dataset_size = args.toy_dataset_size

    root_dir = '/import/network-temp/yimengg/data/'
    train_data = TwitterCOMMsToyDataset(feather_path='../raw_data/train_completed_exist.feather',
                                     img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
                                     multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{base_model}_multimodal_embeds_train.pt',
                                     metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{base_model}_idx_to_image_path_train.json',
                                     size=toy_dataset_size)  # took ~one hour to construct the dataset
    train_iterator = data.DataLoader(train_data,
                                     shuffle=False,
                                     batch_size=batch_size)

    list_pd_series = []
    list_tensor = []
    for batch_idx, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
        multimodal_emb = batch["multimodal_emb"]
        item = batch["item"]
        print(type(item))
        print(type(multimodal_emb))

        list_pd_series += item
        list_tensor += multimodal_emb

    col_names = ['id', 'full_text', 'image_id', 'filename', 'falsified', 'topic', 'exists']
    toy_df = pd.DataFrame(list_pd_series, columns=col_names)
    print(toy_df.head(5))

