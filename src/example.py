"""
Example script demonstrating on how to use the StandWalk estimator.

Description
-----------
This script shows how to use the StandWalk estimator to combine standing and walking estimations.
It includes functions to demonstrate the estimator on a single trial and to compute errors over multiple trials.

Example
-------
> single_SW_plot()
> multiple_SW_errors()
"""

from matplotlib import pyplot as plt
import numpy as np

from data_manager import DataManager, DS_TYPE, get_populated_dmi

from ao_estimator import AO
from clme_estimator import CLME
from cao_estimator import ClmeAO
from standing_estimator import Standing
from standwalk_estimator import StandWalk

from parameter_manager import set_parameters

def single_SW_plot() -> None:
    """
    Function to demonstrate the StandWalk estimator on a trial.
    """

    # Define a data manager object and load data
    dm = DataManager()
    res = dm.load_data("P01", "exp_transition_03-S3_01", DS_TYPE.DEVILLEZ)

    # Check if data is loaded
    if not res:
        print("Failed to load data")
        exit(1)

    # define estimator object
    clme = CLME(dm)
    ao = AO(dm)
    clme_ao = ClmeAO(dm, clme, ao)
    standing = Standing(dm)
    sw = StandWalk(dm, standing, clme_ao)

    # Set parameters
    set_parameters(clme) # Only the CLME require parameter setting

    # Compute estimator
    sw.compute()

    # Plot results
    sw.complete_plot()
    plt.show()


def multiple_SW_errors() -> None:
    """
    Function to compute the StandWalk estimator errors over multiple trials.
    """

    # Get iterator object to go over the dataset

    dmi = get_populated_dmi()
    errors = np.zeros((len(dmi),2))
    for idx, (ds_name, subject, trial) in enumerate(dmi):
        dm = DataManager()
        res = dm.load_data(subject, trial, ds_name)
        if not res:
            print(f"Failed to load data for {subject} {trial}")
            continue
        
        # define estimator object
        clme = CLME(dm)
        ao = AO(dm)
        clme_ao = ClmeAO(dm, clme, ao)
        standing = Standing(dm)
        sw = StandWalk(dm, standing, clme_ao)

        # Set parameters
        set_parameters(clme)

        # Compute estimator
        sw.compute()
        # Compute errors
        sw.errors()
        errors[idx, :] = np.mean(np.abs(sw.error_data), axis=0)
    
    # Mean error over all trials
    mean_errors = np.mean(errors, axis=0)
    print(f"Mean Position Error: {mean_errors[0]:.4f} deg")
    print(f"Mean Velocity Error: {mean_errors[1]:.4f} deg/s")




if __name__ == "__main__":
    single_SW_plot()
    multiple_SW_errors()