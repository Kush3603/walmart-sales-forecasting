# src/data/loader.py

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def load_csv(file_path: str) -> pd.DataFrame:
    """Generic CSV loader"""
    return pd.read_csv(file_path)


def load_raw_data(data_dir: str = "../datasets/raw"):
    """Load all Walmart datasets"""
    data_path = Path(data_dir)

    train = pd.read_csv(data_path / "train.csv")
    features = pd.read_csv(data_path / "features.csv")
    stores = pd.read_csv(data_path / "stores.csv")

    return train, features, stores


def merge_datasets(train, features, stores):
    """Merge train, features, and stores datasets"""

    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on="Store", how="left")

    return df


def load_full_data(data_dir: str = "../datasets/raw"):
    """
    Main function: load + merge everything
    This is what notebooks should call
    """

    logging.info("Loading raw datasets...")

    train, features, stores = load_raw_data(data_dir)
    df = merge_datasets(train, features, stores)

    logging.info(f"Final dataset shape: {df.shape}")

    return df