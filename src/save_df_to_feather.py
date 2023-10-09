import pandas as pd

import os
import logging
import argparse

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True, help="Path to the .csv file")
    p.add_argument("--img_dir", type=str, required=True, help="Directory to the images")
    p.add_argument("--feather_path", type=str, required=True, help="Path to the to-be-saved .feather file")

    args = p.parse_args()
    return args


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()
    csv_path = args.csv_path
    img_dir = args.img_dir
    feather_path = args.feather_path

    logger.info(f"Loading .csv from: {csv_path}")
    logger.info(f"Image directory set to: {img_dir}")

    # Reading from csv
    df = pd.read_csv(csv_path, index_col=0)
    logger.info(f"Length of the dataframe: {len(df)}")

    # Filtering out rows with non-exist images
    logger.info(f"Filtering out rows with non-exist images")
    df['exists'] = df['filename'].apply(lambda filename: os.path.exists(os.path.join(img_dir, filename)))
    delete_row = df[df["exists"] == False].index
    df = df.drop(delete_row)
    df = df.reset_index(drop=True)  # set index from 0 to len(df)-1, now index<->row number, i.e. df.iloc[row number]=df.iloc[index]
    logger.info(f"Length of the filtered dataframe: {len(df)}")

    logger.info(f"To save .feather to: {feather_path}")
    df.to_feather(feather_path)
