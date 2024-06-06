#!/usr/bin/env python
# coding: utf-8

import json
import os
from utils.helper import save_json

"""
Obtains image paths of news contained in the merged_balanced split. 
The obtained image paths are then used to select images to be ssh copied to the hpc.
"""

if __name__ == "__main__":
    visual_news_data = json.load(open("/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin/data.json"))
    visual_news_data_mapping = {ann["id"]: ann for ann in visual_news_data}

    test_img_dir = '/import/network-temp/yimengg/NewsCLIPpings/visual_news/origin'
    test_data_path = f"/import/network-temp/yimengg/NewsCLIPpings/news_clippings/data/merged_balanced/test.json"
    test_data = json.load(open(test_data_path))

    annotations = test_data["annotations"]
    img_path_list = []
    for ann in annotations:
        image_path = visual_news_data_mapping[ann["image_id"]]["image_path"]
        image_path = '/'.join(image_path.split('/')[1:])
        image_path = os.path.join(test_img_dir, image_path)
        img_path_list.append(image_path)
    
    img_path_list_unique = list(set(img_path_list))
    print(len(img_path_list))   #7264

    save_json(img_path_list, "./output/img_path_merged_balance.json")

    


