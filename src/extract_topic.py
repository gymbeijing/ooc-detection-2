import json
import os
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.helper import save_tensor, save_json
import logging
import argparse
from tqdm import tqdm


# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


class NewsDataset(Dataset):
    def __init__(self, img_dir, data_dict, metadata):
        self.img_dir = img_dir
        self.annotations = data_dict["annotations"]
        self.metadata = metadata

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # image_id = ann["image_id"]
        image_path = visual_news_data_mapping[ann["image_id"]]["image_path"]
        str_image_id = str(ann["image_id"])
        image_path_in_metadata = self.metadata[str_image_id]["image_path"]

        assert '/'.join(image_path.split('/')[1:]) == '/'.join(image_path_in_metadata.split('/')[2:]), f"image_path: {image_path}\n image_path_in_metadata: {image_path_in_metadata}"
        topic = self.metadata[str_image_id]["topic"]

        return topic
    

def get_img_dir_and_json(phase, split):
    
    if phase == 'train':
        train_img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
        train_data_path = f"/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/{split}/train.json"
        logger.info(f"Reading json file from {train_data_path}")
        train_data = json.load(open(train_data_path))

        # df_train['exists'] = df_train['filename'].apply(
        #     lambda filename: os.path.exists(os.path.join(train_img_dir, filename)))
        # delete_row = df_train[df_train["exists"] == False].index
        # df_train = df_train.drop(delete_row)

        return train_img_dir, train_data
    if phase == 'valid':
        val_img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
        val_data_path = f"/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/{split}/val.json"
        logger.info(f"Reading json file from {val_data_path}")
        val_data = json.load(open(val_data_path))

        return val_img_dir, val_data
    if phase == 'test':
        test_img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
        test_data_path = f"/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/{split}/test.json"
        logger.info(f"Reading json file from {test_data_path}")
        test_data = json.load(open(test_data_path))

        # df_train['exists'] = df_train['filename'].apply(
        #     lambda filename: os.path.exists(os.path.join(train_img_dir, filename)))
        # delete_row = df_train[df_train["exists"] == False].index
        # df_train = df_train.drop(delete_row)

        return test_img_dir, test_data
    

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, required=True, help="{train, valid, test}")
    p.add_argument("--split", type=str, required=True, help="{semantics_clip_text_image, semantics_clip_text_text, person_sbert_text_text, scene_resnet_place, merged_balanced}")

    args = p.parse_args()
    return args
    

if __name__ == "__main__":
    metadata_dir = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/metadata'
    test_file = 'test.json'
    train_file = 'train.json'
    topic_list = []
    test_metadata = json.load(open(os.path.join(metadata_dir, test_file)))
    for id, item in tqdm(test_metadata.items()):
        topic = item['topic']
        topic_list.append(topic)

    train_metadata = json.load(open(os.path.join(metadata_dir, train_file)))
    for id, item in tqdm(train_metadata.items()):
        topic = item['topic']
        topic_list.append(topic)

    cntr = Counter(topic_list)
    print(cntr)
    # normalized_cntr = [(i, cntr[i] / len(topic_list) * 100.0) for i in cntr]
    # print(normalized_cntr)

    # # Parse arguments
    # args = parse_args()
    # phase = args.phase
    # split = args.split
    # logger.info(f'phase: {phase}')
    # logger.info(f'split: {split}')

    # visual_news_data = json.load(open("/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin/data.json"))
    # visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}   # image_id: image_path

    # metadata = json.load(open(f'/import/network-temp/yimengg/NewsCLIPpings/news_clippings/metadata/{phase}.json'))

    # img_dir, dict_data = get_img_dir_and_json(phase, split)

    # logger.info("Preparing dataset and dataloader")
    # topic_dataset = NewsDataset(img_dir, dict_data, metadata)

    # topic_list = []

    # for idx in tqdm(range(len(topic_dataset))):
    #     topic_list.append(topic_dataset[idx])
    
    # topic_dict = {"topic": topic_list}
    # root_dir = '/import/network-temp/yimengg/NewsCLIPpings/processed_data'
    # logger.info(f"Saving dict to {root_dir}/topic/{split}_topic_{phase}_GaussianBlur.json")
    # save_json(topic_dict, f'{root_dir}/topic/{split}_topic_{phase}_GaussianBlur.json')
