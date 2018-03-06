"""Pure python implementation of Gaussian Naive Bayes.
"""
import os
import pickle
import math


class Model():
    """Model object.
    """
    def __init__(self):
        """Init the model."""
        self._parameters = {}

    def predict(self, inputs_data):
        """
        Predict method.
        Args:
            inputs_data (list): list of inputs.
        Returns:
            outputs_data (list): list of predicted outputs.
        """
        outputs_data = []
        # For each input row
        for input_data in inputs_data:
            probs = []
            # For each labels
            for lab_tuple in self._parameters:
                # Extract the label, the prior and the params
                label, prior, gauss_params = lab_tuple[0], lab_tuple[1], lab_tuple[2]
                # Initiate the probability with prior
                prob = prior
                # For each value
                for col, val in enumerate(input_data):
                    # Update the probability
                    prob *= gaussian_distribution(val,
                                                  gauss_params[col][0],
                                                  gauss_params[col][1])
                # Append it
                probs.append((label, prob))
            # Get the label for which the probability is the greatest
            output = max(probs, key=lambda x:x[1])[0]
            outputs_data.append(output)
        return outputs_data

    def fit(self, labeled_data):
        """Fit a model.
        Args:
            labeled_data (list): list of tuples of data.
        Note:
            self._parameters (list)
            [
                (
                    1,
                    prob(1),
                    [
                        (mean, variance),
                        (mean, variance)
                    ]
                ),
                (
                    0,
                    prob(0),
                    [
                        (mean, variance),
                        (mean, variance)
                    ]
                )
            ]
        """
        # Get the list of unique labels
        unique_labels = list(set([row[1] for row in labeled_data]))
        self._parameters = []
        # Compute the dimension
        dimension = len(labeled_data[0][0])
        # For each labels
        for label in unique_labels:
            params = []
            # Compute the prior
            prior = labels.count(label)/len(labeled_data)
            # For each column
            for col in range(dimension):
                # Get all the value
                vals = [row[0][col] for row in labeled_data if row[1] == label]
                # Compute the mean
                mean = sum(vals)/len(vals)
                # Get all the values for variance
                var_vals = [(row[0][col] - mean)**2 for row in labeled_data if row[1] == label]
                # Compute the variance
                var = sum(var_vals)/len(var_vals)
                params.append((mean, var))
            self._parameters.append((label, prior, params))
        print(self._parameters)

    def persist_parameters(self, model_version):
        """Persist the parameters of the model.
        """
        # Set the folder path.
        folder_path = "library/pure_python/params/{}/".format(model_version)

        # Create a folder if non existant
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        # Perist the parameters.
        with open(folder_path + "params.pkl", 'wb') as handle:
            pickle.dump(self._parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, model_version):
        """Load the parameters of the model.
        """
        # Set folder of the param
        folder_path = "library/pure_python/params/{}/".format(model_version)
        # Load the parameters.
        with open(folder_path + "params.pkl", 'rb') as handle:
            self._parameters = pickle.load(handle)


def gaussian_distribution(value, mean, var):
    """Gaussian distribution for numerical variables.
    Args:
        value (float): value
        mean (float): average
        var (float): variance
    """
    likelihood = 1/(2*math.pi*var)**0.5
    likelihood *= math.exp(
        -(value - mean)**2/(2*var**2)
    )
    return likelihood
