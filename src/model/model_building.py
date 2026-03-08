import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data: %s', e)
        raise


def train_model(X_train, y_train) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(
            C=1,
            solver='liblinear',
            penalty='l2',
            class_weight='balanced',
            max_iter=2000
        )

        clf.fit(X_train, y_train)

        logging.info('Model training completed')
        return clf

    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

        logging.info('Model saved to %s', file_path)

    except Exception as e:
        logging.error('Error saving model: %s', e)
        raise


def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')

        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]

        clf = train_model(X_train, y_train)

        save_model(clf, 'models/model.pkl')

    except Exception as e:
        logging.error('Failed to complete model training: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()