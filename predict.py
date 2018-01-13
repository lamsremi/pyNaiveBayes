"""
Script for prediction.
"""
import importlib
import numpy as np



def main(x_input, model_type, model=None):
    """
    Main prediction function.
    Args:
        x_input (nd_array): input to predict
        model_type (str): type of the chosen model for the prediction
        model (object): specific model to use
    """
    # Init a model if none
    if model is None:
        model = init_model(model_type)
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
    model_class = importlib.import_module("library.{}.model".format(model_type))
    model = model_class.Model()
    return model


if __name__ == '__main__':
    X_INPUT = np.array([1, 2, 3])
    MODEL_TYPE = "random"
    main(X_INPUT, MODEL_TYPE)
