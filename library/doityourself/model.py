"""
Do it yourself Naive Bayes model.
"""
import os
import pickle
import math


class Model():
    """
    Do it yourself class model.
    """
    def __init__(self):
        """Init the model."""
        self._parameters = {}

    def predict(self, input_var):
        """
        Predict method.
        Args:
            input_var (Serie or dict): input to predict the class from.
        """
        # Copy the model parameters for code readebility
        param = self._parameters
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

    def fit(self, data_df):
        """
        Train the model.
        Args:
            data_df (DataFrame): training dataset
        """
        # Set the label columns as the last one
        label_column = list(data_df.columns)[-1]
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
        self._parameters = parameters

    def persist_parameters(self, model_version):
        """
        Persist the model parameters..
        """
        # Set folder of the param
        folder_path = "library/doityourself/params/{}/".format(model_version)

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
        folder_path = "library/doityourself/params/{}/".format(model_version)

        with open(folder_path + "params.pkl", 'rb') as handle:
            self._parameters = pickle.load(handle)
