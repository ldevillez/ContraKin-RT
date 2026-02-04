"""
Support functions for various utilities

Description
-----------
This module provides utility functions for reading YAML files, computing moving averages,
and swapping words in strings.

Example
-------
> read_yaml(Path("config.yml")) # Reads a YAML file and returns its content as a dictionary
> moving_average(data_array, n=5) # Computes the moving average of data_array with a window of 5
> moving_average_and_extend(data_array, n=2) # Computes the moving average and extends the result by padding
> swap_words("hello world", "hello", "hi") # Swaps "hello" with "hi" in the given string
>


"""

import yaml
from pathlib import Path
from numpy import array, cumsum, concatenate, full, arange


def read_yaml(path: Path) -> dict:
    """
    Read the yaml file and return it

    parameters
    ----------
    path : Path
        Path to the yaml file
    """

    if not path.is_file():
        print(f"File {path} not found")
        return {}

    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return {}


def moving_average(a: array, n: float = 3) -> array:
    """
    Compute the moving average of an array

    Parameters
    ----------
    a : array
        The array to compute the moving average
    n : int
        The number of elements to average

    Returns
    -------
    array
        The moving average of the array
    """

    ret = cumsum(a, dtype=float).reshape(-1, 1)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def moving_average_and_extend(a: array, n: float = 3) -> array:
    """
    Compute the moving average of an array and pad the beginning and the end with the first and last value

    Parameters
    ----------
    a : array
        The array to compute the moving average
    n : int
        The number of elements to average is 2n+1
        The padding is n

    Returns
    -------
    array
        The moving average of the array
    """

    ma = moving_average(a, 2 * n + 1)
    return concatenate((full(n, a[0]), ma[:, 0], full(n, a[-1]))).reshape(-1, 1)


def swap_words(s: str, x: str, y: str) -> str:
    """
    Swap in a string s, the word x by the word y and vice versa

    Parameters
    ----------
    s : str
        The string to swap words in
    x : str
        The first word to swap
    y : str
        The second word to swap

    Returns
    -------
    str
        The string with the words swapped
    """

    return y.join(part.replace(y, x) for part in s.split(x))


if __name__ == "__main__":
    # Example usage
    data = arange(10)
    print("Original data:", data.flatten())
    print("Moving average (n=3):", moving_average(data, n=3).flatten())
    print(
        "Moving average and extend (n=2):",
        moving_average_and_extend(data, n=2).flatten(),
    )

    s = "hello world, welcome to the universe"
    print("Original string:", s)
    print("Swapped string:", swap_words(s, "world", "universe"))
