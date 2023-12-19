from tqdm import tqdm
import pandas as pd
import torch

import nltk
import re

# nltk.download('wordnet')
# nltk.download('universal_tagset')
# nltk.download('averaged_perceptron_tagger')

from nltk.corpus import wordnet as wn

import random

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

from utils.helper import remove_punc, remove_url

stop_words = set(stopwords.words('english'))
tt = TweetTokenizer()


def get_word_count(text):
    """
    Get the number of tokens in text
    """
    tokens = word_tokenize(text)
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    return len(filtered)


def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    if word[1] == "NOUN":
        param = wn.NOUN
    elif word[1] == "VERB":
        param = wn.VERB
    elif word[1] == "ADV":
        param = wn.ADV
    elif word[1] == "ADJ":
        param = wn.ADJ
    else:
        ## word not considered for syn rep
        return []
    for syn in wn.synsets(word[0], pos=param):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word[0] in synonyms:
        synonyms.remove(word[0])
    # print("INSIDE GET_SYNONYMS: ")
    # print(synonyms)
    return list(synonyms)


def synonym_replacement(words, n, stop_words):
    """
    Returns the synonym replaced sentence
    """
    fail_count = 0
    # words = words.split()

    new_words = words.copy()
    random_word_list = []
    for word in words:
        if word[1] in ["NOUN", "ADJ", "ADV", "VERB"]:
            random_word_list.append(word)
    # random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word[0] for word in words]
            num_replaced += 1

        else:
            new_words = [word[0] for word in words]  ##no possible synonyms, so just use old words as new

        if num_replaced >= n:  # only replace up to n words
            break

    try:
        sentence = ' '.join(new_words)
    except TypeError as e:
        print(e)
        print('new_words: ', new_words)

        old_words = [word[0] for word in words]
        sentence = ' '.join(old_words)
        fail_count += 1

    return sentence


def augment_dataset(data_frame, percentage=0.2):

    augmented_dataset = []

    for article in tqdm(data_frame.itertuples()):
        body_texts = article.full_text

        if body_texts is None:
            continue
        # Below can be delayed to encoding stage
        # body_texts = ' '.join(tt.tokenize(body_texts))  # tokenized caption
        # body_texts = remove_punc(remove_url(body_texts))  # remove url & punctuation from the tokenized caption
        body_texts = body_texts.split(". ")

        new_sent = []

        for sent in body_texts:

            word_count = get_word_count(sent)
            n = int(word_count*percentage)
            # sent_tokens = word_tokenize(sent)
            sent_tokens = tt.tokenize(sent)
            sent_with_pos = nltk.pos_tag(sent_tokens, tagset='universal')
            new_sent.append(synonym_replacement(sent_with_pos, n, stop_words))

        augmented_article = ". ".join(new_sent)

        augmented_dataset.append([article.full_text, augmented_article, article.image_id,
                                  article.filename, article.falsified, article.topic])

    augmented_frame = pd.DataFrame(augmented_dataset, columns=["full_text", "full_text_perturb", "image_id",
                                                               "filename", "falsified", "topic"])

    # jsonl_data = augmented_frame.to_json(orient='records', lines=True)

    # with open(output_filepath, "w") as text_file:
    #     text_file.write(jsonl_data)

    return augmented_frame


if __name__ == "__main__":
    feather_path = './raw_data/toy_completed_exist.feather'
    df = pd.read_feather(feather_path)
    augmented_df = augment_dataset(df, 0.1)
    augmented_df.to_feather('./raw_data/toy_completed_exist_augmented.feather')
    print(augmented_df.head(10))
    print(f"len: {len(augmented_df)}")