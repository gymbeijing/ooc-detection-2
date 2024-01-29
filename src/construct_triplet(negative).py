import pandas as pd
from tqdm import tqdm
from random import randrange
import numpy as np

if __name__ == '__main__':
    train_df = pd.read_feather('./raw_data/train_completed_exist.feather')
    toy_df = pd.read_feather('./raw_data/toy_completed_exist.feather')
    list_of_negatives = []
    nan_series = pd.Series(np.nan, index=['id', 'full_text', 'image_id', 'filename', 'falsified', 'topic', 'exists'], name=-1)
    print(nan_series)
    nan_cnt = 0

    for index, row in tqdm(toy_df.iterrows(), desc='iteration'):
        id = int(row['id'])
        falsified = row['falsified']
        topic = row['topic']
        difficulty = topic.split('_')[-1]

        if falsified:   # _random or _hard
            candidate_df = train_df[train_df['id']==id]
            result_df = candidate_df[candidate_df['falsified']==False]
            if len(result_df) == 0:
                list_of_negatives.append(nan_series)
                nan_cnt += 1
            else:
                rnd_idx = randrange(len(result_df))
                list_of_negatives.append(result_df.iloc[rnd_idx])   # the original pair is the negative to the falsified pair (both _random and _hard)
        else:   # original
            candidate_df = train_df[train_df['id']==id]
            result_df = candidate_df[(candidate_df['falsified']==True) & (candidate_df['topic'].str.contains('hard'))]
            if len(result_df) == 0:
                # result_df = candidate_df[(candidate_df['falsified']==True) & (candidate_df['topic'].str.contains('random'))]
                list_of_negatives.append(nan_series)
                nan_cnt += 1
            else:
                rnd_idx = randrange(len(result_df))
                list_of_negatives.append(result_df.iloc[rnd_idx])   # the hard negative is the negative to the original pair

    negative_df = pd.DataFrame(list_of_negatives)
    negative_df_reset = negative_df.reset_index()
    negative_df_reset.to_feather('./raw_data/toy_completed_exist_negative.feather')