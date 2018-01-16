"""
Random standard model.
"""
import random
import pickle

class Model():
    """
    Random class model.
    """
    def __init__(self):
        """Init the model."""
        self.parameters = None

    def predict(self, input):
        """Predict method."""
        prediction = random.randint(0, 1)
        return prediction

    def fit(self, data_df, label_column):
        """Fit method."""
        self.parameters = 0

    def persist(self, path_pickle="library/doityourself/params/param_1.pkl"):
        """
        Persist the model parameters..
        """
        with open(path_pickle, 'wb') as handle:
            pickle.dump(self.parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path_pickle="library/doityourself/params/param_1.pkl"):
        """
        Load parameters model.
        """
        with open(path_pickle, 'rb') as handle:
            self.parameters = pickle.load(handle)
