"""
Do it yourself Naive Bayes model.
"""

class Model():
    """
    Do it yourself class model.
    """
    def __init__(self):
        """Init the model."""
        self.parameters = []

    def predict(self, input):
        """Predict method."""
        # Compute the probability for each of class
        for label in self.parameters.keys():

        # Categorize

    def fit(self, data_df, label_column):
        """
        Train the model.
        """
        # Get the list of classes
        labels = data_df[label_column].unique()
        parameters = []
        for label in labels:
            data_df_label = data_df[data_df[label_column] == label]
            param_label = {
                label: {}
            }
            for column in [col for col in list(data_df_label.columns) if col != label_column]:
                param_label[label][column] = {
                    "mean": data_df_label[column].mean(),
                    "variance": data_df_label[column].var()
                }
            parameters.append(param_label)
        self.parameters = parameters
        return parameters
