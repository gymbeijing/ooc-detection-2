import json
import os

if __name__ == '__main__':
    metadata_folder_path = '/import/network-temp/yimengg/NewsCLIPpings/news_clippings/metadata'
    test_metadata = json.load(open(os.path.join(metadata_folder_path, 'test.json')))
    print(test_metadata["1208083"])