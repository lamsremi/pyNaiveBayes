"""
Script for prediction.
"""
import importlib
import numpy as np
import pandas as pd

import tools


# @tools.debug
def main(x_input, model_type, model=None):
    """
    Main prediction function.
    Args:
        x_input (Serie or dict): input to predict
        model_type (str): type of the chosen model for the prediction
        model (object): specific model to use
    """
    # Init a model if none
    if model is None:
        model = init_model(model_type)
    # Predict
    prediction = model.predict(x_input)
    return prediction


def init_model(model_type, data_source="us_election"):
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
    # Load the model parameters
    model.load(path_pickle="library/{}/params/param_{}.pkl".format(model_type, data_source))
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
    MODEL_TYPE = "scikit_learn"
    main(X_INPUT, MODEL_TYPE)
