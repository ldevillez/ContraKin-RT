"""
Module implementing different types of filters

Description
-----------
This module provides a generic Filter class and specific implementations such as ExponentialFilter
and AsymExpFilter for filtering input data.

Example
-------
> exp_filter = ExponentialFilter(alpha=0.1) # Creates an exponential filter with
> filtered_value = exp_filter(input_value) # Filters a single input value
> filtered_array = exp_filter.apply(input_array) # Applies the filter to an array of input
>
> asym_filter = AsymExpFilter(alpha_smaller=0.9) # Creates an asymmetrical exponential filter
> filtered_value = asym_filter(input_value) # Filters a single input value
> filtered_array = asym_filter.apply(input_array) # Applies the filter to an array of input
>
> bidir_filter = BidirExpFilter(alpha_greater=0.95, alpha_smaller=0.85) # Creates a bidirectional exponential filter
> filtered_value = bidir_filter(input_value) # Filters a single input value
> filtered_array = bidir_filter.apply(input_array) # Applies the filter to an array of
"""

from numpy import array, zeros

class Filter:
    """
    Generic class to represent a filter.
    """

    output_value: float

    def __init__(self, base_value: float = 0.0) -> None:
        """
        Initialize a Filter.

        Parameters
        ----------
        base_value : float
            The initial value of the filter output.

        """

        self.output_value = base_value

    def __call__(self, input_value: float) -> float:
        """
        Filtering the input value.

        Parameters
        ----------
        input_value : float
            The input value to be filtered.

        Returns
        -------
        float
            The filtered output value.
        """

        self.output_value = input_value

        return self.get()

    def get(self) -> float:
        """
        Get last computed value

        Returns
        -------
        float
            The last output value of the filter.
        """
        return self.output_value

    def apply(self, input_values: array) -> array:
        """
        Apply the filter to an array of input values.

        Parameters
        ----------
        input_values : array
            The array of input values to be filtered.

        Returns
        -------
        array
            The array of filtered output values.
        """

        output_values = zeros(input_values.shape)

        for i, input_value in enumerate(input_values):
            output_values[i] = self(input_value)

        return output_values


class BidirExpFilter(Filter):
    """
    Class representing an bidirectionnal exponential filter.

    If the input_value is greater than the previous value then the current output_value is equal to the input value
    else the output_value is equal to the previous value multiplied by the decay factor.
    """

    alpha_greater: float
    alpha_smaller: float


    def __init__(self, alpha_greater: float = 0.95, alpha_smaller: float = 0.85, base_value=0) -> None:
        """
        Initialize a BidirExpFilter

        Parameters
        ----------
        alpha_greater : float
            Decay factor when input value is greater than output value.
        alpha_smaller : float
            Decay factor when input value is smaller than output value.
        base_value : float
            The initial value of the filter output.
        """

        super().__init__(base_value)
        self.alpha_greater = alpha_greater
        self.alpha_smaller = alpha_smaller



    def __call__(self, input_value) -> float:
        """
        Filtering the input value with decay

        Parameters
        ----------
        input_value : float
            The input value to be filtered.
        Returns
        -------
        float
            The filtered output value.
        """

        if input_value > self.output_value:
            self.output_value = self.alpha_greater * self.output_value + (1 - self.alpha_greater) * input_value
        else:
            self.output_value = self.alpha_smaller * self.output_value + (1 - self.alpha_smaller) * input_value

        return self.get()

class AsymExpFilter(BidirExpFilter):
    """
    Class representing an asymmetrical exponential filter.

    if the input value is greater than the previous value then the output is not changed,
    else the output value is computed with a exponential filter.
    """

    def __init__(self, alpha_smaller: float = 0.9, base_value = 0) -> None:
        """
        Initialize an AsymExpFilter
        """

        super().__init__(0.0, alpha_smaller, base_value)

class ExponentialFilter(Filter):
    """
    Class representing an exponential filter.

    The output value is computed as a weighted average of the input value and the previous output value.
    """

    alpha: float

    def __init__(self, alpha: float = 0.1, base_value = 0) -> None:
        """
        Initialize an ExponentialFilter

        Parameters
        ----------
        alpha : float
            The smoothing factor for the exponential filter.
        base_value : float
            The initial value of the filter output.
        """

        super().__init__(base_value)
        self.alpha = alpha

    def __call__(self, input_value) -> float:
        """
        Filtering the input value with exponential smoothing

        Parameters
        ----------
        input_value : float


        Returns
        -------
            The filtered output value.
        """

        self.output_value = self.alpha * input_value + (1 - self.alpha) * self.output_value

        return self.get()

if __name__ == "__main__":
    # Example usage of the filters
    exp_filter = ExponentialFilter(alpha=0.1)
    print("Exponential Filter:")
    for i in range(10):
        filtered_value = exp_filter(i)
        print(f"Input: {i}, Filtered: {filtered_value}")

    asym_filter = AsymExpFilter(alpha_smaller=0.9)
    print("\nAsymmetrical Exponential Filter:")
    for i in range(10):
        filtered_value = asym_filter(i)
        print(f"Input: {i}, Filtered: {filtered_value}")

    bidir_filter = BidirExpFilter(alpha_greater=0.95, alpha_smaller=0.85)
    print("\nBidirectional Exponential Filter:")
    for i in range(10):
        filtered_value = bidir_filter(i)
        print(f"Input: {i}, Filtered: {filtered_value}")
