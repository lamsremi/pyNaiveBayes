"""Evaluation module.
"""
import pickle
import importlib

import predict
from benchmark import performance
import tools


@tools.debug
def main(data_source, model_type, model_version):
    """Evaluate the performance of a model.
    Args:
        data_source (str)
        model_type (str)
        model_version (str)
    """
    # Load labaled data
    labeled_data = load_labaled_data(data_source)

    # Extract inputs and outputs data
    inputs_data = [row[0] for row in labeled_data]
    outputs_data = [row[1] for row in labeled_data]

    # Perform a prediction
    predictions = predict.main(inputs_data, model_type, model_version)

    # Assess quantitative performance
    result = performance.confusion_matrix(predictions, outputs_data)
    return result


# @tools.debug
def load_labaled_data(data_source):
    """Load labeled data.
    """
    with open("data/{}/data.pkl".format(data_source), "rb") as handle:
        labeled_data = pickle.load(handle)
    return labeled_data


if __name__ == '__main__':
    for source in ["us_election"]:
        for model_str in ["pure_python", "scikit_learn"]:
            main(source, model_str, "X")
