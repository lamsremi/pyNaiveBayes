"""
Random standard model.
"""
import os
import shutil
import random
import pickle

class Model():
    """
    Random class model.
    """
    def __init__(self):
        """Init the model."""
        self._parameters = None

    def predict(self, input):
        """Predict method."""
        prediction = random.randint(0, 1)
        return prediction

    def fit(self, data_df):
        """Fit method."""
        self._parameters = 0

    def persist_parameters(self, model_version):
        """
        Persist the model parameters..
        """
        # Set folder of the param
        folder_path = "library/random/params/{}/".format(model_version)

        # Create folder
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        # Save params
        with open(folder_path + "params.pkl", 'wb') as handle:
            pickle.dump(self._parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, model_version):
        """
        Load parameters model.
        """
        # Set folder of the param
        folder_path = "library/random/params/{}/".format(model_version)

        with open(folder_path + "params.pkl", 'rb') as handle:
            self._parameters = pickle.load(handle)
