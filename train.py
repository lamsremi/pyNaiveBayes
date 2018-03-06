"""Script to fit naive bayes models.
"""
import pickle
import importlib

import tools


def main(labeled_data=None,
         data_source=None,
         model_type=None,
         model_version=None):
    """Main function for fitting a model to labeled data.
    Args:
        labeled_data (list): table of labeled data.
        [
            ([ 0.0, 6.0, 6.0, 2.0], 1.0)
            ([ 8.0, 7.0, 4.0, 2.0], 1.0)
        ]
        data_source (str): source in case no data is given.
        model_type (str): type of model to use. (ie. pure_python or scikit_learn)
        model_version (str): version of model to use.
    """

    if labeled_data is None:
        # Load labaled data
        labeled_data = load_labaled_data(data_source)

    # Init the model
    model = init_model(model_type)

    # Train the model
    model.fit(labeled_data)

    # Store the model parameters.
    model.persist_parameters(model_version=model_version)


# @tools.debug
def load_labaled_data(data_source):
    """Load labeled data.
    """
    with open("data/{}/data.pkl".format(data_source), "rb") as handle:
        labeled_data = pickle.load(handle)
    return labeled_data


def init_model(model_type):
    """Instantiate a model.
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
    for source in ["us_election"]:
        for model_str in ["pure_python", "scikit_learn"]:
            main(labeled_data=None,
                 data_source=source,
                 model_type=model_str,
                 model_version="X")
