# data preprocessing

import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging

nltk.download("wordnet")
nltk.download("stopwords")


def preprocess_dataframe(df, col="review"):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):

        if pd.isna(text):
            return ""

        text = str(text)

        # remove urls
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # lowercase
        text = text.lower()

        # remove punctuation
        text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)

        # remove extra spaces
        text = re.sub("\s+", " ", text).strip()

        words = text.split()

        # remove stopwords
        words = [word for word in words if word not in stop_words]

        # lemmatize
        words = [lemmatizer.lemmatize(word) for word in words]

        text = " ".join(words)

        return text

    df[col] = df[col].astype(str).apply(preprocess_text)

    # keep rows with at least 1 word
    df = df[df[col].str.split().str.len() > 0]

    logging.info("Data preprocessing completed")

    return df


def main():

    try:

        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")

        logging.info("Raw data loaded successfully")

        train_processed_data = preprocess_dataframe(train_data, "review")
        test_processed_data = preprocess_dataframe(test_data, "review")

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(data_path, "train_processed.csv"),
            index=False
        )

        test_processed_data.to_csv(
            os.path.join(data_path, "test_processed.csv"),
            index=False
        )

        logging.info("Processed data saved successfully")

    except Exception as e:

        logging.error("Data preprocessing failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()