#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
from pathlib import Path
from typing import List, cast
import logging
import os

import multiprocessing
import shutil
import sys
import time
from argparse import ArgumentParser
from functools import wraps
from multiprocessing.process import BaseProcess

import numpy as np
import requests
from tqdm import tqdm
import gc


# # Test download_image.py

# In[28]:


logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


# In[29]:


class LastTime:
    """
    Credit: Copied and modified this: https://gist.github.com/gregburek/1441055
    >>> import rate_limited as rt
    >>> a = rt.LastTime()
    >>> a.add_cnt()
    >>> a.get_cnt()
    1
    >>> a.add_cnt()
    >>> a.get_cnt()
    2
    """

    def __init__(self, name="LT"):
        # Init variables to None
        self.name = name
        self.ratelock = None
        self.cnt = None
        self.last_time_called = None

        # Instantiate control variables
        self.ratelock = multiprocessing.Lock()
        self.cnt = multiprocessing.Value("i", 0)
        self.last_time_called = multiprocessing.Value("d", 0.0)

        logging.debug("\t__init__: name=[{!s}]".format(self.name))

    def acquire(self):
        self.ratelock.acquire()

    def release(self):
        self.ratelock.release()

    def set_last_time_called(self):
        self.last_time_called.value = time.time()
        # self.debug('set_last_time_called')

    def get_last_time_called(self):
        return self.last_time_called.value

    def add_cnt(self):
        self.cnt.value += 1

    def get_cnt(self):
        return self.cnt.value

    def debug(self, debugname="LT"):
        now = time.time()
        logging.debug(
            "___Rate name:[{!s}] "
            "debug=[{!s}] "
            "\n\t        cnt:[{!s}] "
            "\n\tlast_called:{!s} "
            "\n\t  timenow():{!s} ".format(
                self.name,
                debugname,
                self.cnt.value,
                time.strftime(
                    "%T.{}".format(
                        str(self.last_time_called.value - int(self.last_time_called.value)).split(
                            "."
                        )[1][:3]
                    ),
                    time.localtime(self.last_time_called.value),
                ),
                time.strftime(
                    "%T.{}".format(str(now - int(now)).split(".")[1][:3]), time.localtime(now)
                ),
            )
        )


# In[2]:


def load_tweets(json_path: Path) -> pd.DataFrame:
    assert json_path.exists(), str(json_path) 
    df = pd.read_json(
        json_path,
        lines=True,
        precise_float=True,
        dtype={"id": int, "id_str":int},  # specify the datatype of id_str (int), otherwise will lose precision
    )
    # Note: id_str matches the tweet_id's that are passed into hydrator:
    df["tweet_id"] = df.id_str.astype(int)
#     df["tweet_id"] = df.id_str
    df = df.drop_duplicates("tweet_id")   # each tweet id tends to appear ~twice
    df.set_index(
        "tweet_id",
        drop=False,   # used to be False
        inplace=True,
        verify_integrity=True,
    )
    df["country"] = df.geo.apply(
        lambda x: x["country"] if isinstance(x, dict) and "country" in x else None
    )
    return cast(pd.DataFrame, df)


# In[3]:


def pre_json_normalize(
    row, parent_col_name: str, target_col_name: str, child_properties: list = None
) -> dict:
    """
    Prepares a column of a dataframe (`target_col_name`) to be run through
    `json_normalize()`. To achieve this it takes a row of a dataframe containing tweet
    data, and:
    1. Adds the `parent_col_name: row[parent_col_name]` as a key-value pair to the dict
        object contained in `row[target_col_name]`. This helps link the rows of the
        original DataFrame to those of the DataFrame that is output when the target
        column is run thru `json_normalize()`.
    2. Adds a `"media": []` key-value pair if no media key is in the target dict.
    3. If the target dict is null, a new one is created.
    """
    target_dict = row[target_col_name]
    if not isinstance(target_dict, dict):
        target_dict = {}
    if parent_col_name not in target_dict:
        target_dict[parent_col_name] = row[parent_col_name]
    if child_properties:
        for prop_name, prop_type in child_properties:
            if prop_name not in target_dict:
                target_dict[prop_name] = prop_type()
    return target_dict


# In[4]:


def get_photo_urls(json_path: Path, media_type: str = "photo") -> pd.DataFrame:
    """
    Returns a DataFrame of media info from the give json file. The input file should be a twitter
    json file that came from the Twitter v2 Search API.
    Example output:
        >       height              id_str   type                                              url  width  duration_ms preview_image_url  public_metrics.view_count alt_text             tweet_id             filename
        > 0       1920  3_1427530554480619521  photo  https://pbs.twimg.com/media/E8-bedXVcAEqkXA.jpg   1080          NaN               NaN                        NaN      NaN  1427531793771622400    xx/<filename>.jpg
        > 1        409  3_1427531788591771661  photo  https://pbs.twimg.com/media/E8-cmSyXMA0ETDr.jpg    615          NaN               NaN                        NaN      NaN  1427531789392883727    xx/<filename>.jpg
        > 2       1000  3_1427531783868878850  photo  https://pbs.twimg.com/media/E8-cmBMVkAIcD4a.jpg   1000          NaN               NaN                        NaN      NaN  1427531786746163202    xx/<filename>.jpg
        > 3        546  3_1427531482029903875  photo  https://pbs.twimg.com/media/E8-cUcwUUAMRefa.jpg    728          NaN               NaN                        NaN      NaN  1427531786477740034    xx/<filename>.jpg
        > 4        375  3_1427526863543578632  photo  https://pbs.twimg.com/media/E8-YHnjXsAgRpIu.jpg    500          NaN               NaN                        NaN      NaN  1427531785458659334    xx/<filename>.jpg
        > ...      ...                    ...    ...                                              ...    ...          ...               ...                        ...      ...                  ...                  ...
    """
    df = load_tweets(json_path)
    if "attachments" in df.columns:
        df["attachments"] = df.apply(
            lambda row: pre_json_normalize(row, "tweet_id", "attachments", [("media", list)]), axis=1
        )
        df_media = pd.json_normalize(df.attachments, record_path="media", meta=["tweet_id"])
    else:
        df["entities"] = df.apply(
            lambda row: pre_json_normalize(row, "tweet_id", "entities", [("media", list)]), axis=1
        )
        df_media = pd.json_normalize(df.entities, record_path="media", meta=["tweet_id"])
    df_media = df_media[(df_media.type == media_type) & ~df_media.media_url.isna()]   # might filtered out some rows
    df_media.reset_index(inplace=True)
    df_media.index.name = "id"

    def _get_filename(row):
        extension = Path(row["media_url"]).suffix if row["media_url"] and not pd.isna(row["media_url"]) else None
        filename = f"{row['id_str'][-2:]}/{row['tweet_id']}-{row['id_str']}{extension}"
        return filename

    df_media["filename"] = df_media.apply(lambda row: _get_filename(row), axis=1)
    df.index.names = ['index']
#     df = df[['tweet_id', 'full_text']]
#     df_media = df_media.merge(df, on='tweet_id')
    logger.info(f"Found {len(df_media)} media urls of type: '{media_type}'")
#     print(df_media)
    return cast(pd.DataFrame, df_media)


# In[6]:


def rate_limited(max_per_second):
    """
    Decorator to rate limit a python function.
    Example, limits to 10 requests per second, and works w/ multiprocessing as well:
        > @rate_limited(10)
        > def download_img(media_url):
        > ...
    Credit: Copied and modified this: https://gist.github.com/gregburek/1441055
    """
    min_interval = 1.0 / max_per_second
    LT = LastTime("rate_limited")

    def decorate(func):
        LT.acquire()
        if LT.get_last_time_called() == 0:
            LT.set_last_time_called()
        LT.debug("DECORATE")
        LT.release()

        @wraps(func)
        def rate_limited_function(*args, **kwargs):

            logging.debug(
                "___Rate_limited f():[{!s}]: "
                "Max_per_Second:[{!s}]".format(func.__name__, max_per_second)
            )

            try:
                LT.acquire()
                LT.add_cnt()
                xfrom = time.time()

                elapsed = xfrom - LT.get_last_time_called()
                left_to_wait = min_interval - elapsed
                logging.debug(
                    "___Rate f():[{!s}] "
                    "cnt:[{!s}] "
                    "\n\tlast_called:{!s} "
                    "\n\t time now():{!s} "
                    "elapsed:{:6.2f} "
                    "min:{!s} "
                    "to_wait:{:6.2f}".format(
                        func.__name__,
                        LT.get_cnt(),
                        time.strftime("%T", time.localtime(LT.get_last_time_called())),
                        time.strftime("%T", time.localtime(xfrom)),
                        elapsed,
                        min_interval,
                        left_to_wait,
                    )
                )
                if left_to_wait > 0:
                    time.sleep(left_to_wait)

                ret = func(*args, **kwargs)

                LT.debug("OVER")
                LT.set_last_time_called()
                LT.debug("NEXT")

            except Exception as ex:
                sys.stderr.write(
                    "+++000 " "Exception on rate_limited_function: [{!s}]\n".format(ex)
                )
                sys.stderr.flush()
                raise
            finally:
                LT.release()
            return ret

        return rate_limited_function

    return decorate


# In[ ]:


@rate_limited(30)
def save_image(idx, img_row, images_dir: Path, size="large"):
    """
    Download and save image to path.
    Args:
        image: The url of the image.
        path: The directory where the image will be saved.
        filename:
        size: Which size of images to download.
    """
    if img_row["media_url"]:
        save_dest: Path = images_dir / img_row["filename"]
        save_dest.parent.mkdir(exist_ok=True, parents=True)

        if not save_dest.exists():
            # logger.info(f"Saving image: {save_dest.name}")
            r = requests.get(img_row["media_url"] + ":" + size, stream=True)
            if r.status_code == 200:
                with open(save_dest, "wb") as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
            elif r.status_code in [403, 404]:
                pass
            else:
                print(f"Error on {idx}, tweet_id:{img_row['tweet_id']}, url:{img_row['media_url']}")
                print(r.headers)
                print(r.status_code, ", ", r.reason)
                print(r.text)
            if r.status_code in [429]:
                sleep_seconds = 15 * 60
                logger.error(f"Rate limit hit... Will retry in {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)
        else:
            # print(f"Skipping {save_dest}: already downloaded")
            pass


# In[8]:


def dl_images(df_media: pd.DataFrame, images_dir: Path) -> None:
    """
    Single process, just download one images at a time.
    """
    for idx, row in df_media.iterrows():
        save_image(idx, row, images_dir, size="orig")

'''
# # Prepare completed train set

# ## Step 1: load (text)tweet json lines

# In[13]:

print("Reading in train_text_ids.jsonl...")
df = load_tweets(Path('/import/network-temp/yimengg/data/twitter-comms/train/train_text_ids.jsonl'))


# In[14]:


# Only keep the <tweet id> and the <text>
df = df[['id_str', 'full_text']]


# ## Step 2: load train.csv and merge with df to append text

# In[15]:

print("Reading in train.csv...")
df_train = pd.read_csv('../data/train.csv', 
                     index_col=0,
                     dtype={"id": int, "id_str":int})


# In[16]:

print("Merging df_train and df...")
df_train = df_train.merge(df, left_on='id', right_on='id_str')

df_train = df_train[['id', 'full_text', 'image_id', 'falsified', 'topic']]


# In[17]:

print("Dropping NA in df_train...")
df_train = df_train.dropna()
print("Writing to train_full_text_completed.csv...")
df_train.to_csv('/import/network-temp/yimengg/data/twitter-comms/train/train_full_text_completed.csv')

del df
del df_train
gc.collect()
'''
# ## Step 3: load (image)tweet json files

# In[5]:


# Test get_photo_urls()
logger.info("Reading in train_images_ids.jsonl...")
df_media = get_photo_urls(Path('/import/network-temp/yimengg/data/twitter-comms/train/train_image_ids.jsonl'))


# In[ ]:

logger.info("Renaming columns in df_media...")
df_media = df_media.rename({'id': 'media_id', 'id_str': 'media_id_str'}, axis='columns')


# ## Step 4: load train.csv from step 2 and merge with df_media to append image filepath

# In[ ]:

logger.info("Reading train_full_text_completed.csv...")
df_train = pd.read_csv('/import/network-temp/yimengg/data/twitter-comms/train/train_full_text_completed.csv', 
                     index_col=0)
#                     dtype={"id": int, "id_str":int})


# In[ ]:

logger.info("Merging df_train and df_media...")
df_train = df_train.merge(df_media, left_on='image_id', right_on='tweet_id')


# In[ ]:

logger.info("Extracting columns from df_train")
df_train = df_train[['id', 'full_text', 'image_id', 'filename', 'falsified', 'topic']]


# In[ ]:

logger.info("Writing df_train to csv file...")
df_train.to_csv('../data/train_completed.csv')


# In[ ]:




