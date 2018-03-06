"""Scikit learn implementation of Gaussian Naive Bayes.
"""
import os
import pickle

from sklearn.naive_bayes import GaussianNB


class Model():
    """Model object
    """
    def __init__(self):
        """Init the class."""
        self._gnb = GaussianNB()

    def predict(self, inputs_data):
        """Perform a prediction.
        Args:
            inputs_data (list): input to predict the class from.
        """
        predictions = list(self._gnb.predict(inputs_data))
        return predictions

    def fit(self, labeled_data):
        """Fit a model on labeled data.
        Args:
            labeled_data (list): labeled dataset.
        """
        # Format the data according to scikit-learn framework
        x_array = [row[0] for row in labeled_data]
        y_array = [row[1] for row in labeled_data]
        # Fit
        self._gnb.fit(x_array, y_array)

    def persist_parameters(self, model_version):
        """Persist the model parameters.
        """
        # Set folder of the param
        folder_path = "library/scikit_learn/params/{}/".format(model_version)
        # Create folder
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        # Store
        with open(folder_path + "params.pkl", 'wb') as handle:
            pickle.dump(self._gnb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, model_version):
        """Load parameters model.
        """
        # Set folder of the param
        folder_path = "library/scikit_learn/params/{}/".format(model_version)
        with open(folder_path + "params.pkl", 'rb') as handle:
            self._gnb = pickle.load(handle)
