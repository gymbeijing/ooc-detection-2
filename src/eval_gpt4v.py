import base64
import requests

from PIL import Image
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


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_queries_and_image_paths(df):
    queries = []
    image_paths = []
    labels = []

    for idx, item in tqdm(df.iterrows(), desc='iterations'):
        # text = "Answer with yes or no. Does the image match the following text? "
        text = "Task: News Image-Text Matching Analysis."
        "Objective: To assess whether the attached image and the provided text description correspond with each other accurately,"
        "Instructions:"
        "1. Examine the attached image carefully."
        "2. Read the text description provided below the image."
        "3. Determine the degress of alignment between the image content and the text. Consider factors such as the main subjects, background elements, and overall context."
        "4. Provide your response in a clear 'yes' or 'no' format."
        "Prompt: Does the following text description accurately represent the content and context of the attached image? Please respond with 'yes' if there is a match and 'no' if there are discrepancies."
        " Does the image match the following text? "
        text += item['full_text']  # original caption
        img_filename = item['filename']
        image_path = os.path.join(val_img_dir, img_filename)
        label = item["falsified"]
        queries.append(text)
        image_paths.append(image_path)
        labels.append(label)

    return queries, image_paths, labels


def eval_model(top_n):
    queries, image_paths, labels = load_queries_and_image_paths(val_df[:top_n])
    preds = []
    for query, image_path in tqdm(zip(queries, image_paths), desc='iterations'):
        base64_image = encode_image(image_path)
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        preds.append(response.json()["choices"][0]['message']['content'])

    preds = [True if pred == "No" else False for pred in preds]
    num_correct = sum(x == y for x, y in zip(preds, labels))
    num_total = len(labels)
    print(f"num_correct: {num_correct}")
    print(f"num_total: {num_total}")
    print(float(num_correct / num_total))


# # Path to your image
# image_path = "/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids/78/1421763579028643843-1421763541678411778.jpg"
#
# # Getting the base64 string
# base64_image = encode_image(image_path)
#
# headers = {
#   "Content-Type": "application/json",
#   "Authorization": f"Bearer {api_key}"
# }
#
# payload = {
#     "model": "gpt-4-vision-preview",
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Task: News Image-Text Matching Analysis."
#                             "Objective: To assess whether the attached image and the provided text description correspond with each other accurately,"
#                             "Instructions:"
#                             "1. Examine the attached image carefully."
#                             "2. Read the text description provided below the image."
#                             "3. Determine the degress of alignment between the image content and the text. Consider factors such as the main subjects, background elements, and overall context."
#                             "4. Provide your response in a clear 'yes' or 'no' format."
#                             "Prompt: Does the following text description accurately represent the content and context of the attached image? Please respond with 'yes' if there is a match and 'no' if there are discrepancies."
#                             " Does the image match the following text? White House: Climate among 'root causes' of migration https://t.co/gcDVO2b7gD The new White House strategy for improving conditions in Central America to slow migration includes helping to build resilience to climate change. https://t.co/c2ONeB44x3"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}"
#                     }
#                 }
#             ]
#         }
#     ],
#     "max_tokens": 300
# }
#
# response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#
# print(response.json()["choices"][0]['message']['content'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    val_feather_path = './raw_data/val_completed_exist_with_llava_outputs.feather'
    val_df = pd.read_feather(val_feather_path)  # already drop the non-exists
    val_img_dir = '/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids'
    eval_model(args.top_n)
