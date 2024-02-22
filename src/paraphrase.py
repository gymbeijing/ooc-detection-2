from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import pandas as pd

from tqdm import tqdm

from utils.helper import remove_punc, remove_url
from nltk.tokenize import TweetTokenizer

tt = TweetTokenizer()

# load pre-trained Pegasus Paraphrase model and tokenizer
tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")


def paraphrase(data_frame):
    augmented_dataset = []

    for article in tqdm(data_frame.itertuples()):
        body_texts = article.full_text
        # print(body_texts)

        if body_texts is None:
            continue

        # Below can be delayed to encoding stage
        body_texts = ' '.join(tt.tokenize(body_texts))  # tokenized caption
        body_texts = remove_punc(remove_url(body_texts))  # remove url & punctuation from the tokenized caption

        # Tokenize the input sentence
        input_ids = tokenizer.encode(body_texts, return_tensors='pt')
        print(input_ids.shape)

        # Generate paraphrased sentence
        paraphrase_ids = model.generate(input_ids, num_beams=5, max_length=100, early_stopping=True)

        # Decode and print the paraphrased sentence
        paraphrase = tokenizer.decode(paraphrase_ids[0], skip_special_tokens=True)


        augmented_dataset.append([article.full_text, paraphrase, article.image_id,
                                  article.filename, article.falsified, article.topic, article.exists])
    
    augmented_frame = pd.DataFrame(augmented_dataset, columns=["full_text", "full_text_paraphrase", "image_id",
                                                               "filename", "falsified", "topic", "exists"])

    return augmented_frame


if __name__ == "__main__":
    feather_path = './raw_data/toy_completed_exist.feather'
    df = pd.read_feather(feather_path)[:10]
    augmented_df = paraphrase(df)
    # augmented_df.to_feather('./raw_data/toy_completed_exist_paraphrase.feather')
    print(augmented_df.head(10))
    print(f"len: {len(augmented_df)}")