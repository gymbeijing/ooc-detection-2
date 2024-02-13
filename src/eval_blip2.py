# activate venv_py38
'''
python src/eval_blip2.py
'''
from lavis.models import load_model_and_preprocess
from PIL import Image
import argparse
import logging
import pandas as pd
import torch
import requests
from io import BytesIO
from tqdm import tqdm
import os

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)



def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        raw_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raw_image = Image.open(image_file).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return image


def load_images(image_file):
    image = load_image(image_file)
    return image


def load_queries_and_image_paths(df):
    queries = []
    image_paths = []
    labels = []

    for idx, item in tqdm(df.iterrows(), desc='iterations'):
        # Prompt 1
        # text = "Answer with yes or no. Does the image match the following text? "

        # Prompt 2
        # text = "Task: News Image-Text Matching Analysis."
        # "Objective: To assess whether the attached image and the provided text description correspond with each other accurately,"
        # "Instructions:"
        # "1. Examine the attached image carefully."
        # "2. Read the text description provided below the image."
        # "3. Determine the degress of alignment between the image content and the text. Consider factors such as the main subjects, background elements, and overall context."
        # "4. Provide your response in a clear 'yes' or 'no' format."
        # "Prompt: Does the following text description accurately represent the content and context of the attached image? Please respond with 'yes' if there is a match and 'no' if there are discrepancies."
        # " Does the image match the following text? "

        # # Prompt 3
        # text = "Below is an instruction that describes the task. Write a response that appropriately completes the request. "
        # "[Instructions]:"
        # "1. Examine the query image (the third attached image) carefully"
        # "2. Read the caption provided carefully"
        # "3. Determine whether the image and the caption are semantically matched. Consider factors such as the main objects, background elements and overall context."
        # "4. Provide your response in a clear ‘yes’ or ‘no’ format."
        # "5. Look at two examples provided with label in the below and get an understanding of the task."
        # "[Example 1]: Take the first attached image and the following caption as the first example with ground true label. [Caption]: Speaker Pelosi and Senate Maj Leader Schumer talking about climate change action now on @cspan"
        # "@cspanwj https://t.co/9vw0qxdk7a [Label]: Yes"
        # "[Example 2]: Take the second attached image and the following caption as the second example with ground true label. [Caption]: Russia and China continue to unabashedly copy military vehicle designs. These pictures are *not* of a Black Hawk, CB90 or A400M..."
        # "(Z-20, Project 03160 and suspected Y-30) https://t.co/vRZtzwxHgn [Label]: No"
        # "[Prompt]: Does the following caption accurately represent the content and the context of the query image (the third attached image)? Please respond with ‘yes’ if there is a match and ‘no’ if there are discrepancies identified."
        # "[Caption]: "

        # Prompt 4
        text = "Below is an instruction that describes the task. Write a response that appropriately completes the request."
        "[Task]: Out-of-context Image Detection"
        "[Objective]: To learn to categorize image-text posts as pristine or falsified (out-of-context) by means of detecting semantic inconsistencies between images and text."
        "[Instructions]:"
        "1. Examine the attached image carefully."
        "2. Read the text description provided carefully."
        "3. Determine whether the image and the text are originally paired in the news post. Consider factors such as the main objects, background elements and overall context."
        "4. Provide your response in a clear 'yes' or 'no' format."
        "[Prompt]: Does the following text description and the attached image come from the same news post? Please respond with 'yes' if there is a semantic match and 'no' if there are semantic inconsistencies."

        text += item['full_text']  # original caption
        img_filename = item['filename']
        image_path = os.path.join(val_img_dir, img_filename)
        label = item["falsified"]
        queries.append(text)
        image_paths.append(image_path)
        labels.append(label)

    return queries, image_paths, labels


def eval_model():

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )

    queries, image_paths, labels = load_queries_and_image_paths(val_df)
    # qs = args.query   # can it be batched queries?
    preds = []

    for q, image_file in tqdm(zip(queries, image_paths), desc='iterations'):
        # print(image_file)
        image = load_images(image_file)
        output = model.generate({"image": image, "prompt": f"Question: {q} Answer:"})

        preds.append(output)

    preds = [True if pred == "No" else False for pred in preds]
    num_correct = sum(x == y for x, y in zip(preds, labels))
    num_total = len(labels)
    print(float(num_correct / num_total))


if __name__ == "__main__":

    val_df = pd.read_feather('raw_data/val_completed_exist.feather')
    val_img_dir = '/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'device: {device}')

    logger.info("Loading blip-2")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
    )

    eval_model()
