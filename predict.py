"""
Script for prediction.
"""
import pickle
import importlib

import tools


# @tools.debug
def main(inputs_data,
         model_type,
         model_version):
    """
    Main prediction function.
    Args:
        inputs_data (array-like): list of inputs to predict.
        model_type (str): type of the chosen model for the prediction
        model_version (str): version of model to use.
    """
    # Init a model if none
    model = init_model(model_type)
    # Load the model parameters
    model.load_parameters(model_version)
    # Predict
    predictions = model.predict(inputs_data)
    # Return
    return predictions


def init_model(model_type):
    """
    Init a model.
    Args:
        model_type (str): type of the model to init.
    Return:
        model (object): loaded model
    """
    # Import the good model
    model_class = importlib.import_module("library.{}.model".format(model_type))
    # Init the instance
    model = model_class.Model()
    return model

# @tools.debug
def load_labaled_data(data_source):
    """Load labeled data.
    """
    with open("data/{}/data.pkl".format(data_source), "rb") as handle:
        labeled_data = pickle.load(handle)
    return labeled_data


if __name__ == '__main__':
    for data_source in ["us_election", "titanic"]:
        inputs_data = [row[0] for row in load_labaled_data(data_source)][0:100]
        for model_str in ["pure_python", "scikit_learn"]:
            main(inputs_data=inputs_data,
                 model_type=model_str,
                 model_version=data_source)
