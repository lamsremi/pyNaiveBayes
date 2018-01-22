"""
Script to train models.
"""
import importlib
import pandas as pd

import tools


def main(data_df=None,
         data_source=None,
         model_type=None):
    """
    Main function for training.
    """

    if data_df is None:
        # Load labaled data
        data_df = load_labaled_data(data_source)

    # Init the model
    model = init_model(model_type)

    # Train the model
    model.fit(data_df)

    # Store the model parameters.
    model.persist_parameters(model_version=data_source)


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


if __name__ == '__main__':
    for source in ["us_election", "titanic"]:
        for model_str in ["random", "doityourself", "scikit_learn"]:
            main(data_df=None,
                 data_source=source,
                 model_type=model_str)
