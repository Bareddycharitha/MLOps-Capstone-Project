# data ingestion

import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)

import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging


def load_params(params_path: str) -> dict:
    """Load parameters from YAML."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except Exception as e:
        logging.error("Error loading params: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sentiment column and keep only positive/negative."""
    try:

        logging.info("Preprocessing sentiment column")

        # clean sentiment column
        df["sentiment"] = (
            df["sentiment"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # keep only positive/negative rows
        final_df = df[df["sentiment"].isin(["positive", "negative"])].copy()

        # convert labels
        final_df["sentiment"] = final_df["sentiment"].map({
            "positive": 1,
            "negative": 0
        })

        logging.info("Filtered dataset shape: %s", final_df.shape)

        return final_df

    except Exception as e:
        logging.error("Error during preprocessing: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    """Save train and test datasets."""
    try:

        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logging.info("Train and test data saved to %s", raw_data_path)

    except Exception as e:
        logging.error("Error saving data: %s", e)
        raise


def main():

    try:

        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]

        df = load_data(
            "https://raw.githubusercontent.com/Bareddycharitha/Datasets/main/data%20.csv"
        )

        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        save_data(train_data, test_data, "./data")

    except Exception as e:
        logging.error("Failed to complete data ingestion: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()