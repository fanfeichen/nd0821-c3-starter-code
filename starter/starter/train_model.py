# Script to train machine learning model.

import pandas as pd
from ml.data import clean_data, process_data
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load and clean raw data
cur_path = str(Path(__file__).parent.absolute())
data = clean_data(cur_path + "/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
