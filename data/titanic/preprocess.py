"""Script to preprocess the kaggle titanic dataset.
"""
import pickle

import pandas as pd

import tools

def main():
    """Preprocess the data."""
    # Load the raw data
    raw_data_df = load_raw_data(path_raw_data="data/titanic/raw_data/data.csv")
    # Study data
    study_data(raw_data_df)
    # Transform the data
    data_df = process(raw_data_df)
    # Study transformed data
    study_data(data_df)
    # Format data
    labeled_data = format_data(data_df)
    # Store the data
    store(labeled_data, path_preprocessed_data="data/titanic/data.pkl")


def load_raw_data(path_raw_data):
    """Load the raw data."""
    raw_data_df = pd.read_csv(
        path_raw_data,
        nrows=10000,
    )
    return raw_data_df


def study_data(data_df):
    """Examine the data."""
    # Display shape
    print("- shape :\n{}\n".format(data_df.shape))
    # Display data dataframe (raws and columns)
    print("- dataframe :\n{}\n".format(data_df.head(10)))
    # Display types
    print("- types :\n{}\n".format(data_df.dtypes))
    # Missing values
    print("- missing values :\n{}\n".format(data_df.isnull().sum()))


def process(raw_data_df):
    """Clean the data.
    """
    # Select a subset of columns
    data_df = raw_data_df[[
        "Pclass",
        "Fare",
        "Age",
        "SibSp",
        "Parch",
        "Survived"]]
    # Convert to float type
    for attribute in data_df.columns:
        data_df[attribute] = raw_data_df[attribute].astype(float)
    data_df.dropna(inplace=True)
    return data_df


@tools.debug
def format_data(data_df):
    """Format the data.
    """
    labeled_data = [(list(row[1:-1]), row[-1]) for row in data_df.itertuples()]
    return labeled_data


def store(labeled_data, path_preprocessed_data):
    """Store the processed data."""
    with open(path_preprocessed_data, "wb") as handle:
        pickle.dump(labeled_data, handle)
