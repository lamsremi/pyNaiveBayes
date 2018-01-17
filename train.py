"""
Script to train models.
"""
import importlib
import pandas as pd

import tools

def main(data_source, model_type):
    """
    Main function for training.
    """
    # Load labaled data
    data_df = load_labaled_data(data_source)

    # Init the model
    model = init_model(model_type)

    # Train the model
    model = fit(model, data_df, data_source)

    # Store the model parameters.
    model.persist_parameters(path_pickle="library/{}/params/param_{}.pkl".format(
        model_type, data_source))

# @tools.debug
def load_labaled_data(data_source):
    """
    Load labeled data.
    """
    data_df = pd.read_pickle("data/{}/data.pkl".format(data_source))
    return data_df


def init_model(model_type):
    """
    Init a model.
    Args:
        model_type (str): type of the model to init.
    Return:
        model (object): loaded model
    """
    # Import the library
    model_class = importlib.import_module("library.{}.model".format(model_type))
    # Inits the model instance
    model = model_class.Model()
    return model


def fit(model, data_df, data_source):
    """
    Fit task.
    """
    if data_source == "us_election":
        model.fit(data_df, "vote")
    elif data_source == "titanic":
        model.fit(data_df, "Survived")
    return model

if __name__ == '__main__':
    for source in ["us_election", "titanic"]:
        for model_str in ["random", "doityourself", "scikit_learn"]:
            main(source, model_str)
