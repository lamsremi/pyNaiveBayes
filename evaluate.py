"""
Script to evaluate performances of model.
"""
import importlib
import pandas as pd

import predict
from performance import qualitative
from performance import quantitative
import tools


pd.set_option('display.width', 800)

def main(data_source, model_type):
    """
    Main evaluate functions.
    Args:
    Return:
    """
    output_names = {
        "us_election": "vote",
        "titanic": "Survived"
    }
    # Load labaled data
    data_df = load_labaled_data(data_source)
    # Init the model
    model = init_model(model_type, data_source)
    # Predict all the data
    predicted_data_df = predict_frame(data_df, model_type, model)
    # Assess qualitative performance
    # qualitative.main(predicted_data_df)
    # Assess quantitative performance
    result = quantitative.main(predicted_data_df, output_names[data_source])
    tools.print_elegant(result)


# @tools.debug
def load_labaled_data(data_source):
    """
    Load labeled data.
    """
    data_df = pd.read_pickle("data/{}/data.pkl".format(data_source))
    return data_df


def init_model(model_type, data_source):
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
    # Load the model
    model.load(path_pickle="library/{}/params/param_{}.pkl".format(model_type, data_source))
    return model


def predict_frame(data_df, model_type, model):
    """
    Perform the prediction of all the input of the DataFrame.
    """
    predicted_data_df = data_df.copy()
    for key, serie in data_df.iloc[:, :-1].iterrows():
        predicted_data_df.loc[key, "prediction"] = predict.main(
            serie,
            model_type,
            model
        )
    return predicted_data_df


if __name__ == '__main__':
    for source in ["us_election", "titanic"]:
        for model_str in ["random", "doityourself", "scikit_learn"]:
            main(source, model_str)
