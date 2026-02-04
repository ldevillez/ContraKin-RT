"""
Module to define all the logistic functions used in the estimators
"""

import numpy as np
import matplotlib.pyplot as plt


def logistic(steep, median, x):
    return 1 / (1 + np.exp(steep * (x - median)))


class Logistic:
    """
    Class representing a logistic function.
    """

    steep: float
    median: float
    callback: callable
    inverse: bool

    def __init__(self, steep, median, callback, inverse=False) -> None:
        """
        Initialize a logistic function with a steepness and a median.
        """
        self.steep = steep
        self.median = median
        self.callback = callback
        self.inverse = inverse

    def __call__(self) -> float:
        """
        Compute the value of the logistic function at x.
        """
        if self.inverse:
            return 1 - logistic(self.steep, self.median, self.callback())
        return logistic(self.steep, self.median, self.callback())

    def compute(self, input_data):
        """
        Compute the value of the logistic function at x.

        Parameters
        ----------
        input_data : array
            Input data to compute the logistic function
        """

        if self.inverse:
            return 1 - logistic(self.steep, self.median, input_data)
        return logistic(self.steep, self.median, input_data)


class FuseFonction:
    """
    Class representing the multiplication of multiple logistic functions.
    """

    functions: list[Logistic]

    def __init__(self) -> None:
        pass

    def add(self, logistic: Logistic):
        self.functions.append(logistic)

    def __call__(self) -> float:
        """
        Compute the value of the logistic function at x.
        """
        return np.prod([f() for f in self.functions])


class TimeLogistic(Logistic):
    """
    Class representing a logistic function of time.
    """

    time: float

    def __init__(self, steep, time_half, inverse=True) -> None:
        """
        Initialize a logistic function with a steepness and a median.
        """
        super().__init__(steep, time_half, None, inverse)

        self.time = 0

    def reset_time(self, time):
        self.time = time

    def compute(self, current_time):
        """
        Compute the value of the logistic function for the current_time.

        Parameters
        ----------
            Input data to compute the logistic function
        """

        return super().compute(current_time - self.time)


if __name__ == "__main__":
    x = np.linspace(0, 20, 200)
    time_log = TimeLogistic(steep=1, time_half=3, inverse=True)
    y = np.zeros(x.shape)

    for idx, xi in enumerate(x):
        if xi < 5:
            time_log.reset_time(xi)

        y[idx] = time_log.compute(xi)

    plt.figure()
    plt.plot(x, y)
    plt.show()
