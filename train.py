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

    # Store the model
    model.persist(path_pickle="library/{}/params/param_{}.pkl".format(model_type, data_source))

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
    model_class = importlib.import_module("library.{}.model".format(model_type))
    model = model_class.Model()
    return model


def fit(model, data_df, data_source):
    """
    Fit task.
    """
    if data_source == "us_election":
        parameters = model.fit(data_df, "vote")
        tools.print_elegant(parameters)
    return model

if __name__ == '__main__':
    DATA_SOURCE = "us_election"
    MODEL_TYPE = "scikit_learn"
    main(DATA_SOURCE, MODEL_TYPE)
