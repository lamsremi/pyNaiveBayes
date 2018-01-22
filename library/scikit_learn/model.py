"""
Scikit learn implementation of gaussian naive bayes.
"""
import os
import pickle

import numpy as np
from sklearn.naive_bayes import GaussianNB


class Model():
    """
    class Model.
    """
    def __init__(self):
        """Init the class."""
        self._gnb = GaussianNB()

    def fit(self, data_df):
        """
        Train the model.
        Args:
            data_df (DataFrame): training dataset
            label_column (str): name of the label column
        """
        # Set the label columns as the last one
        label_column = list(data_df.columns)[-1]
        x_array = np.array(data_df.loc[:, data_df.columns != label_column])
        y_array = np.array(data_df[label_column])
        self._gnb.fit(x_array, y_array)

    def predict(self, input_var):
        """
        Predict method.
        Args:
            input_var (Serie or dict): input to predict the class from.
        """
        x_array = np.array(input_var)
        if x_array.ndim == 1: # If only one sample
            x_array = x_array.reshape(1, -1)
        prediction = self._gnb.predict(x_array)
        return prediction

    def persist_parameters(self, model_version):
        """
        Persist the model parameters.
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
        """
        Load parameters model.
        """
        # Set folder of the param
        folder_path = "library/scikit_learn/params/{}/".format(model_version)

        with open(folder_path + "params.pkl", 'rb') as handle:
            self._gnb = pickle.load(handle)
