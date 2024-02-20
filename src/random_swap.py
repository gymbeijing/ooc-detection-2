import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm

from utils.helper import remove_punc, remove_url
from nltk.tokenize import TweetTokenizer

aug = naw.RandomWordAug(action="swap", aug_max=5, aug_p=0.2)
tt = TweetTokenizer()


def random_swap(data_frame):
    augmented_dataset = []

    for article in tqdm(data_frame.itertuples()):
        body_texts = article.full_text
        # print(body_texts)

        if body_texts is None:
            continue
        # Below can be delayed to encoding stage
        body_texts = ' '.join(tt.tokenize(body_texts))  # tokenized caption
        body_texts = remove_punc(remove_url(body_texts))  # remove url & punctuation from the tokenized caption
        augmented_text = aug.augment(body_texts)
        # print(augmented_text)
        # print(len(augmented_text))   # 1

        augmented_dataset.append([article.full_text, augmented_text[0], article.image_id,
                                  article.filename, article.falsified, article.topic, article.exists])
    
    augmented_frame = pd.DataFrame(augmented_dataset, columns=["full_text", "full_text_random_swap", "image_id",
                                                               "filename", "falsified", "topic", "exists"])

    return augmented_frame


if __name__ == "__main__":
    feather_path = './raw_data/toy_completed_exist.feather'
    df = pd.read_feather(feather_path)
    augmented_df = random_swap(df)
    augmented_df.to_feather('./raw_data/toy_completed_exist_random_swap.feather')
    print(augmented_df.head(10))
    print(f"len: {len(augmented_df)}")