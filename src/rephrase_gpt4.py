import requests

from io import BytesIO
import re
import pandas as pd
import os
from tqdm import tqdm
import argparse

# OpenAI API Key
api_key = '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'

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}


def load_queries_and_image_paths(df):
    queries = []

    for idx, item in tqdm(df.iterrows(), desc='iterations'):
        # Prompt
        text = "Rephrase the following text with as little as its semantic meaning being changed: "
        text += item['full_text']  # original caption
        img_filename = item['filename']
        queries.append(text)

    return queries


def rephrase_query(df, top_n):
    queries = load_queries_and_image_paths(df[:top_n])
    preds = []
    for query in tqdm(queries, desc='iterations'):
        payload = {
            "model": "gpt-4-1106-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        preds.append(response.json()["choices"][0]['message']['content'])

    assert len(preds) == top_n, "Prediction length doesn't match the specified number of datapoints."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    toy_feather_path = './raw_data/toy_completed_exist.feather'
    toy_df = pd.read_feather(toy_feather_path)  # already drop the non-exists

    rephrased_texts = rephrase_query(toy_df, args.top_n)

    mini_toy_df = toy_df.head(args.top_n)
    mini_toy_df.insert(len(mini_toy_df.columns), "rephrased_gpt4", rephrased_texts)
    mini_toy_df.to_feather('./raw_data/mini_toy_completed_exist_rephrased.feather')

    