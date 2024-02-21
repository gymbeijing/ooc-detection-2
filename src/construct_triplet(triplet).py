import pandas as pd

if __name__ == '__main__':
    negative_df = pd.read_feather('./raw_data/toy_completed_exist_negative.feather')
    random_crop_df = pd.read_feather('./raw_data/toy_completed_exist_random_crop.feather')
    random_swap_df = pd.read_feather('./raw_data/toy_completed_exist_random_swap.feather')
    toy_df = pd.read_feather('./raw_data/toy_completed_exist.feather')

    negative_df = negative_df.rename(columns={'filename': 'filename_negative'})

    random_crop_triplet_df = pd.concat([toy_df, negative_df.filename_negative, random_crop_df.full_text_random_crop], axis=1)
    random_crop_triplet_df = random_crop_triplet_df.dropna().reset_index()
    random_crop_triplet_df.to_feather('./raw_data/toy_completed_exist_random_crop_triplet.feather')
    print(len(random_crop_triplet_df))

    random_swap_triplet_df = pd.concat([toy_df, negative_df.filename_negative, random_swap_df.full_text_random_swap], axis=1)
    random_swap_triplet_df = random_swap_triplet_df.dropna().reset_index()
    random_swap_triplet_df.to_feather('./raw_data/toy_completed_exist_random_swap_triplet.feather')
    print(len(random_swap_triplet_df))

