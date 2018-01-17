"""
Do it yourself Naive Bayes model.
"""
import pickle
import math


class Model():
    """
    Do it yourself class model.
    """
    def __init__(self):
        """Init the model."""
        self.parameters = {}

    def predict(self, input_var):
        """
        Predict method.
        Args:
            input_var (Serie or dict): input to predict the class from.
        """
        # Copy the model parameters for code readebility
        param = self.parameters
        # Compute the probability fo each of class/label given the input
        probabilities = {}
        for label, param_label in param.items():
            prior = param_label["probability"]
            likelihood = prior
            for attribute, param_attribute in param_label.items():
                if attribute != "probability":
                    var_likelihood = self.gaussian_distribution(
                        input_var[attribute],
                        param_attribute["mean"],
                        param_attribute["variance"]
                    )
                    likelihood *= var_likelihood
            probabilities[label] = likelihood
        # Categorize
        return max(probabilities, key=probabilities.get)


    @staticmethod
    def gaussian_distribution(value, mean, variance):
        """
        Gaussian distribution for numerical variables.
        """
        var_likelihood = 1/(2*math.pi*variance)**0.5
        var_likelihood *= math.exp(
            -(value - mean)**2/(2*variance**2)
        )
        return var_likelihood

    def fit(self, data_df, label_column=None):
        """
        Train the model.
        Args:
            data_df (DataFrame): training dataset
            label_column (str): name of the label column
        """
        # Get the list of classes
        labels = data_df[label_column].unique()
        parameters = {}
        data_counts = data_df[label_column].value_counts()
        for label in labels:
            data_df_label = data_df[data_df[label_column] == label]
            parameters[label] = {}
            parameters[label]["probability"] = round(data_counts[label]/len(data_df), 2)
            for column in [col for col in list(data_df_label.columns) if col != label_column]:
                parameters[label][column] = {
                    "mean": data_df_label[column].mean(),
                    "variance": data_df_label[column].var()
                }
        self.parameters = parameters

    def persist_parameters(self, path_pickle="library/doityourself/params/param_0.pkl"):
        """
        Persist the model parameters..
        """
        with open(path_pickle, 'wb') as handle:
            pickle.dump(self.parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, path_pickle="library/doityourself/params/param_0.pkl"):
        """
        Load parameters model.
        """
        with open(path_pickle, 'rb') as handle:
            self.parameters = pickle.load(handle)
