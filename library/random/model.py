"""
Random standard model.
"""
import random

class Model():
    """
    Random class model.
    """
    def __init__(self):
        """Init the model."""
        self.model = None

    def predict(self, input):
        """Predict method."""
        prediction = random.randint(0, 1)
        return prediction
