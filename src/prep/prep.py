# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse
from dotenv import load_dotenv
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")

    return parser.parse_args()

def main(args):
    '''Read, split, and save datasets'''

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    df = pd.read_csv((Path(args.raw_data)))

    # ------ Preprocess & Split Data ------- #
    # -------------------------------------- #
    df_prep = df.drop(columns=['Posted On', 'Area Locality'])
    df_prep['Floor'] = df_prep['Floor'].str.replace('Ground', '1')
    df_prep['Floor_num'] = df_prep['Floor'].apply(lambda x: x.split(' out of ')[0])
    df_prep['Floor_building'] = df_prep['Floor'].apply(lambda x: int(x.split(' out of ')[-1]))
    df_prep.loc[df_prep['Floor_num'].str.contains('Basement'), 'Floor_num'] = '0'
    df_prep['Floor_num'] = df_prep['Floor_num'].astype(int)
    df_prep = df_prep.drop(columns='Floor')

    train, test = train_test_split(df_prep, test_size=0.2, random_state=42)

    # Log dataset sizes to MLflow
    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    # Save the datasets as parquet files
    train.to_parquet(Path(args.train_data) / "train.parquet")
    test.to_parquet(Path(args.test_data) / "test.parquet")


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",

    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()

    
