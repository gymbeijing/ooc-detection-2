# activate venv_llava
'''
python src/eval_llava.py --model-path ../LLaVA/llava-v1.5-7b --sep , --temperature 0 --num_beams 1 --max_new_tokens 512
'''
import sys
sys.path.append('../LLaVA')

import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import pandas as pd
import os
from tqdm import tqdm

# model_path = "../LLaVA/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_queries_and_image_paths(df):
    queries = []
    image_paths = []
    labels = []

    for idx, item in tqdm(df.iterrows(), desc='iterations'):
        # Prompt 1.1
        # Answered with No/Yes
        # text = "Answer with yes or no. Does the image match the following text? "

        # Prompt 1.2
        text = "Does the following text description and the attached image come from the same news post?"
        "Please respond with 'yes' if there is a semantic match and 'no' if there are semantic inconsistencies."
        "Text description: "

        # Prompt 2
        # Answered with text chunk
        # text = "Task: News Image-Text Matching Analysis."
        # "Objective: To assess whether the attached image and the provided text description correspond with each other accurately,"
        # "Instructions:"
        # "1. Examine the attached image carefully."
        # "2. Read the text description provided below the image."
        # "3. Determine the degress of alignment between the image content and the text. Consider factors such as the main subjects, background elements, and overall context."
        # "4. Provide your response in a clear 'yes' or 'no' format."
        # "Prompt: Does the following text description accurately represent the content and context of the attached image? Please respond with 'yes' if there is a match and 'no' if there are discrepancies."
        # " Does the image match the following text? "

        # Prompt 3
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
        # Answered with text chunk
        # text = "Below is an instruction that describes the task. Please write a response that appropriately completes the request."
        # "Task: Out-of-context Image Detection"
        # "Objective: To learn to categorize image-text posts as pristine or falsified (out-of-context) by means of detecting semantic inconsistencies between images and text."
        # "Instructions:"
        # "1. Examine the attached image carefully."
        # "2. Read the text description provided carefully."
        # "3. Determine whether the image and the text are originally paired in the news post. Consider factors such as the main objects, background elements and overall context."
        # "4. Provide your response in a clear 'yes' or 'no' format."
        # "Prompt: Does the following text description and the attached image come from the same news post? Please respond with 'yes' if there is a semantic match and 'no' if there are semantic inconsistencies."

        text += item['full_text']  # original caption
        img_filename = item['filename']
        image_path = os.path.join(val_img_dir, img_filename)
        label = item["falsified"]
        queries.append(text)
        image_paths.append(image_path)
        labels.append(label)

    return queries, image_paths, labels


def eval_model(args):
    # Model
    disable_torch_init()

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )

    queries, image_paths, labels = load_queries_and_image_paths(val_df)
    # qs = args.query   # can it be batched queries?
    preds = []

    for qs, image_files in tqdm(zip(queries, image_paths), desc='iterations'):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # image_files = image_parser(args)   # can it be batched image files? yes
        image_files = [image_files]
        images = load_images(image_files)
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print(outputs.split(",")[0])
        preds.append(outputs.split(",")[0])   # Added split only for the Prompt 1.2

    preds = [True if pred == "No" or pred == 'no' else False for pred in preds]
    num_correct = sum(x == y for x, y in zip(preds, labels))
    num_total = len(labels)
    print(float(num_correct / num_total))
    val_df.insert(len(val_df.columns), "llava_prompt_1.2", preds)
    val_df.to_feather("./raw_data/val_completed_exist_with_llava_outputs.feather")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)   # prompt
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    val_feather_path = './raw_data/val_completed_exist_with_llava_outputs.feather'
    val_df = pd.read_feather(val_feather_path)  # already drop the non-exists
    val_img_dir = '/import/network-temp/yimengg/data/twitter-comms/images/val_images/val_tweet_image_ids'

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    eval_model(args)
