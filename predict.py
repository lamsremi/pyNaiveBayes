"""
Script for prediction.
"""
import importlib
import numpy as np
import pandas as pd

import tools


# @tools.debug
def main(x_input,
         model_type,
         model_version):
    """
    Main prediction function.
    Args:
        x_input (Serie or dict): input to predict
        model_type (str): type of the chosen model for the prediction
        model_version (str): version of model to use.
    """
    # Init a model if none
    model = init_model(model_type)
    # Load the model parameters
    model.load_parameters(model_version)
    # Predict
    prediction = model.predict(x_input)
    return prediction


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


if __name__ == '__main__':
    X_INPUT = pd.Series({
        "popul": 300,
        "TVnews": 3,
        "selfLR": 3,
        "ClinLR": 3,
        "DoleLR": 5,
        "PID": 1,
        "age": 45,
        "educ": 4,
        "income": 15
    })
    for model_str in ["random", "doityourself", "scikit_learn"]:
        main(x_input=X_INPUT,
             model_type=model_str,
             model_version="us_election")
