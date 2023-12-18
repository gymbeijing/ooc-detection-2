from torch.utils.data import Dataset
from utils.helper import save_tensor, load_tensor, load_json
import pandas as pd
import numpy as np
import os
import torch.utils.data as data
import argparse
from tqdm.auto import tqdm
import torch
from utils.helper import save_tensor


class TwitterCOMMsToyDataset(Dataset):
    def __init__(self, feather_path, img_dir, multimodal_embeds_path, metadata_path, ratio=0.015):
        """
        Args:
            feather_path (string): Path to the {train|val}_completed_exist.feather file.
            img_dir (string): Directory containing the images
        """
        # self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.multimodal_embeds = load_tensor(multimodal_embeds_path)
        self.metadata = load_json(metadata_path)

        self.df = pd.read_feather(feather_path)   # already drop the non-exists
        self.domain_map_to_idx = {"climate": 0, "covid": 1, "military": 2}

        # if not excluding any topic
        self.row_kept = self.df.index

        # Randomly sample n=size number of news
        df_sampled = self.df.sample(frac=ratio).sort_index()
        self.row_kept_sampled = df_sampled.index

    def __len__(self):
        return len(self.row_kept_sampled)

    def __getitem__(self, idx):
        row_number = self.row_kept_sampled[idx]
        item = self.df.iloc[row_number]
        multimodal_emb = self.multimodal_embeds[row_number]
        cols = {"id": str(item["id"]),
                "full_text": item["full_text"],
                "image_id": str(item["image_id"]),
                "filename": item["filename"],
                "falsified": item["falsified"],
                "topic": item["topic"]}
        image_path = os.path.join(self.img_dir, item["filename"])
        assert image_path == self.metadata[str(row_number)], "Image path does not match with the metadata"

        return {"cols": cols,
                "multimodal_emb": multimodal_emb}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, required=True, help="batch size")
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
    p.add_argument("--ratio", type=float, required=True, help="ratio of the toy dataset")

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    batch_size = args.bs
    base_model = args.base_model
    ratio = args.ratio

    root_dir = '/import/network-temp/yimengg/data/'
    train_data = TwitterCOMMsToyDataset(feather_path='./raw_data/train_completed_exist.feather',
                                     img_dir=root_dir+'twitter-comms/train/images/train_image_ids',
                                     multimodal_embeds_path=root_dir+f'twitter-comms/processed_data/tensor/{base_model}_multimodal_embeds_train.pt',
                                     metadata_path=root_dir+f'twitter-comms/processed_data/metadata/{base_model}_idx_to_image_path_train.json',
                                     ratio=ratio)  # took ~one hour to construct the dataset
    train_iterator = data.DataLoader(train_data,
                                     shuffle=False,
                                     batch_size=batch_size)

    list_tensor = []
    col_dict = {"id": [], "full_text": [], "image_id": [], "filename": [], "falsified": [], "topic": []}
    for batch_idx, batch in tqdm(enumerate(train_iterator, 0), desc='iterations'):
        multimodal_emb = batch["multimodal_emb"]
        cols = batch["cols"]
        # print(type(col_list)) # list
        # print(type(multimodal_emb)) # torch.tensor

        col_dict["id"] += cols["id"]
        col_dict["full_text"] += cols["full_text"]
        col_dict["image_id"] += cols["image_id"]
        col_dict["filename"] += cols["filename"]
        col_dict["falsified"] += cols["falsified"]
        col_dict["topic"] += cols["topic"]
        list_tensor += multimodal_emb

    col_dict["falsified"] = [bool(item) for item in col_dict["falsified"]]   # torch.bool -> bool

    # print(col_dict)
    toy_tensor = torch.stack(list_tensor, dim=0)
    toy_df = pd.DataFrame(col_dict)
    # print(toy_df.head(10))

    assert toy_tensor.shape[0] == len(toy_df), f"toy_tensor has shape {toy_tensor.shape}, toy_df has length {len(toy_df)}"

    toy_df.to_feather("./raw_data/toy_completed_exist.feather")
    save_tensor(toy_tensor, root_dir + f"twitter-comms/processed_data/tensor/{base_model}_multimodal_embeds_toy.pt")






