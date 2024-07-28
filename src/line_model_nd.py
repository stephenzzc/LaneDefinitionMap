import numpy as np

class LineModelND:
    def __init__(self):
        self.params = None

    def estimate(self, data):
        # The linear model is defined as y = ax + b
        # We use np.polyfit to find the best fit line
        x, y = data.T
        a, b = np.polyfit(x, y, deg=1)
        self.params = np.array([a, b])

    def predict_xy(self, data):
        x, _ = data.T
        return np.array([x, self.params[0] * x + self.params[1]]).T

    def residuals(self, data):
        # Calculate the residuals using the fitted line
        x, y = data.T
        return y - (self.params[0] * x + self.params[1])

    def get_params(self):
        return self.params