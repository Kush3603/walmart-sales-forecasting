# src/data/loader.py

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def load_csv(file_path: str) -> pd.DataFrame:
    """Generic CSV loader."""
    return pd.read_csv(file_path)


def load_raw_data(data_dir: str = "../data/raw"):
    """Load all Walmart raw datasets."""
    data_path = Path(data_dir)

    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    features = pd.read_csv(data_path / "features.csv")
    stores = pd.read_csv(data_path / "stores.csv")

    return train, test, features, stores


def combine_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Combine train and test before preprocessing so both receive
    the same cleaning and feature engineering steps.
    """

    train = train.copy()
    test = test.copy()

    train["is_train"] = 1
    test["is_train"] = 0

    test["Weekly_Sales"] = pd.NA

    data = pd.concat([train, test], axis=0, ignore_index=True)

    return data


def merge_data(data: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Merge combined train/test data with features and stores."""

    data = data.copy()
    features = features.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    df = data.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    df = df.merge(
        stores,
        on="Store",
        how="left"
    )

    return df


def load_full_data(data_dir: str = "../data/raw") -> pd.DataFrame:
    """
    Load train/test/features/stores, combine train and test,
    then merge with external features and store metadata.
    """

    logging.info("Loading raw data...")

    train, test, features, stores = load_raw_data(data_dir)

    logging.info(f"Train shape: {train.shape}")
    logging.info(f"Test shape: {test.shape}")

    combined_data = combine_train_test(train, test)

    logging.info(f"Combined train/test shape: {combined_data.shape}")

    df = merge_data(combined_data, features, stores)

    logging.info(f"Final merged dataset shape: {df.shape}")

    return df