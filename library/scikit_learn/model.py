"""
Scikit learn implementation of gaussian naive bayes.
"""
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB


class Model():
    """
    class Model.
    """
    def __init__(self):
        """Init the class."""
        self.gnb = GaussianNB()

    def fit(self, data_df, label_column):
        """
        Train the model.
        Args:
            data_df (DataFrame): training dataset
            label_column (str): name of the label column
        """
        x_array = np.array(data_df.loc[:, data_df.columns != label_column])
        y_array = np.array(data_df[label_column])
        self.gnb.fit(x_array, y_array)

    def predict(self, input_var):
        """
        Predict method.
        Args:
            input_var (Serie or dict): input to predict the class from.
        """
        x_array = np.array(input_var)
        if x_array.ndim == 1: # If only one sample
            x_array = x_array.reshape(1, -1)
        prediction = self.gnb.predict(x_array)
        return prediction

    def persist_parameters(self, path_pickle="library/scikit_learn/params/param_0.pkl"):
        """
        Persist the model parameters.
        """
        with open(path_pickle, 'wb') as handle:
            pickle.dump(self.gnb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, path_pickle="library/scikit_learn/params/param_0.pkl"):
        """
        Load parameters model.
        """
        with open(path_pickle, 'rb') as handle:
            self.gnb = pickle.load(handle)
