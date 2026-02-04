"""
Module to set parameters of certains objects with respect to the subject

Description
-----------
This module provides functionality to set parameters for specific objects based on
the subject being analyzed. It reads parameters from a YAML configuration file and
applies them to the object if applicable.

Example
-------
> set_parameters(clme_estimator)
"""

from pathlib import Path
from clme_estimator import CLME
from support import read_yaml

parameters = read_yaml(Path("parameters.yml"))


def set_parameters(obj) -> None:
    """
    Set parameters of the object with respect to the subject

    Parameters
    ----------
    obj : object
        Object to set parameters
    """

    if type(obj) is CLME:
        # Get clme parameters
        if "CLME" not in parameters:
            return

        clme_params = parameters["CLME"]
        dm = obj.data_manager

        # Get related database parameters
        if dm.ds_name.value not in clme_params:
            return

        db_params = clme_params[dm.ds_name.value]

        if dm.subject is None or dm.subject not in db_params:
            return

        subject_params = db_params[dm.subject]

        # Set parameters
        for key, value in subject_params.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
            else:
                print(f"Warning: {key} is not an attribute of {type(obj)}")

    else:
        print("Settings parameters for", type(obj), " is not implemented")


if __name__ == "__main__":
    from data_manager import DataManager
    from dataset_manager import ds

    # Example usage
    dm = DataManager()
    dm.load_data("P01", "exp_transition_03-S3_01", ds.DEVILLEZ)
    clme = CLME(dm)

    print("Before setting parameters:")
    print("offset 0:", clme.offset[0, 0])
    print("offset 1:", clme.offset[1, 0])

    set_parameters(clme)
    print("")
    print("After setting parameters:")
    print("offset 0:", clme.offset[0, 0])
    print("offset 1:", clme.offset[1, 0])
