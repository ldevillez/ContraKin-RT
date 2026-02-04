"""
Utility module to manipulate datasets

Description
-----------

This module provides functions to list datasets, subjects, and trials,
convert column names, and load data from specified datasets.

Datasets supported:
    - Devillez
The module is easily extendable to support additional datasets.

Example
-------
> list_ds() # Lists available datasets
> list_subject(ds.DEVILLEZ) # Lists subjects in the Devillez dataset
> list_trial("P01", ds.DEVILLEZ) # Lists trials for P01 in the Devillez dataset
> load_devillez("P01", "exp_transition_03-S3_01") # Loads data for trial exp_transition_03-S3_01 of subject P01 in the Devillez dataset
> convert_col("q_r_hip", ds.DEVILLEZ) # Converts column name from ContraKin specification to Devillez specification

"""

import os
from pathlib import Path
from enum import Enum

from numpy import loadtxt, array, pi, logical_and


from support import read_yaml


# Edit the path for your own datasets location
PATH_TO_DS = Path("../datasets")

contrakin_to_devillez = {
    "q_r_hip": "joint_q_q_15",
    "qd_r_hip": "joint_qd_qdd_qd_15",
    "q_r_knee": "joint_q_q_18",
    "qd_r_knee": "joint_qd_qdd_qd_18",
    "q_r_ankle": "joint_q_q_21",
    "qd_r_ankle": "joint_qd_qdd_qd_21",
    "q_l_hip": "joint_q_q_6",
    "qd_l_hip": "joint_qd_qdd_qd_6",
    "q_l_knee": "joint_q_q_9",
    "qd_l_knee": "joint_qd_qdd_qd_9",
    "q_l_ankle": "joint_q_q_12",
    "qd_l_ankle": "joint_qd_qdd_qd_12",
    "q_pelvis_yaw": "joint_q_q_2",
    "qd_pelvis_yaw": "joint_qd_qdd_qd_2",
    "q_pelvis_pitch": "joint_q_q_3",
    "qd_pelvis_pitch": "joint_qd_qdd_qd_3",
    "q_pelvis_roll": "joint_q_q_4",
    "qd_pelvis_roll": "joint_qd_qdd_qd_4",
        }


class ds(Enum):
    """
    Enum of the different datasets
    """

    NONE = "NONE"
    DEVILLEZ = "DEVILLEZ"

def read_config_devillez(user: str) -> dict:
    """
    Read the config file of the user.
    Valid only for the devillez dataset

    parameters
    ----------
    user : str
        User ID

    returns
    -------
    config : dict
    """
    ds_user_path = PATH_TO_DS / "devillez" / user
    config_path = ds_user_path / "config.yaml"
    return read_yaml(config_path)


def list_ds() -> list[ds]:
    """
    List all the datasets available
    """

    return [ ds.DEVILLEZ ]


def list_subject(ds_name: ds) -> list[str]:
    """
    List the subjects in the given ds

    parameters
    ----------
    ds_name : ds
        Dataset name
    """

    if ds_name == ds.DEVILLEZ:
        dir_path = PATH_TO_DS
        ds_path = os.path.join(dir_path, "devillez")
        listdir = []

        for test_path in os.listdir(ds_path):
            if os.path.isdir(os.path.join(ds_path, test_path)):
                listdir.append(test_path)

        listdir.sort()
        return listdir


    print(f"{ds_name} is not a recognised ds")
    return []


def list_trial(user: str, ds_name: ds, full_trial=False) -> list[str]:
    """
    List all the trial from a given user of a given ds

    parameters
    ----------
    user : str
        User ID
    ds_name : ds
        Dataset name
    full_trial : bool
        If True, list the trials with full trial data

    returns
    -------
    list_of_trials : list[str]
        List of the trials
    """

    if ds_name == ds.DEVILLEZ:
        ds_user_path = PATH_TO_DS / "devillez" / user

        list_of_trials = []

        # Full trials are the txt files in the folder
        if full_trial:
            for trials in ds_user_path.iterdir():
                if trial.endswith(".txt"):
                    list_of_trial.append(trial.split(".")[0])
            return list_of_trials

        # The sub trials are defined in the config file

        config_path = ds_user_path / "config.yaml"
        config = read_yaml(config_path)

        for main_trials in config:
            for sub_trials in config[main_trials]["trials"]:
                list_of_trials.append(f"{main_trials}-{sub_trials}")
        return list_of_trials

    print(f"{ds_name} is not a recognised ds")
    return []


def convert_col(colname: str, ds_name: ds) -> str | None:
    """
    Convert cols names from ContraKin to ds specification

    parameters
    ----------
    colname : str
        Column name in the ds
    ds_name : ds
        Dataset name

    returns
    -------
    converted_colname : str | None
    """
    try:
        if ds_name == ds.DEVILLEZ:
                return contrakin_to_devillez[colname]
        else:
            print(f"{ds_name} is not a recognised ds")
            return None
    except KeyError:
        print(f"Column {colname} not found in {ds_name} to contrakin mapping")
        return None


def load(subject: str, trial: str, ds_name: ds, full_trial:bool = False) -> tuple[array, array, array, list, dict]:
    """
    Load data from the specified ds, subject and trial

    Load angular position and velocity as well as the columns name and the idx of the columns. If the trial is not found, return empty arrays

    parameters
    ----------
    subject : str
        Subject ID
    trial : str
        Trial ID
    ds_name : ds
        Dataset name
    full_trial : bool
        If True, load the full trial data

    Returns
    -------
    time : array
        Time vector
    angle : array
        Angular position
    velocity : array
        Angular velocity
    cols : list
        List of the columns
    idx_of_cols : dict
        Dict of the idx of the columns
    """
    if ds_name == ds.DEVILLEZ:
        return load_devillez(subject, trial, full_trial)
    else:
        print(f"{ds_name} is not a recognised ds")
        return array([]), array([]), array([]), [], {}, None


def load_devillez(subject, trial, full_trial=False):
    """
    Retun Time, angle, velocities, col and idx_of_cols
    """

    # Read the full trial in both case
    # For a non full trial strip the end
    if full_trial:
        filename = os.path.join(PATH_TO_DS, "devillez", subject, f"{trial}.txt")
    else:
        splitted_trial = trial.rsplit("-", 1)
        main_trial = splitted_trial[0]
        sub_trial = splitted_trial[1]
        filename = os.path.join(PATH_TO_DS, "devillez", subject, f"{main_trial}.txt")

    raw_data = loadtxt(filename, dtype=object, delimiter=" ", skiprows=1)

    # Get all the cols from the filename
    with open(filename) as f:
        cols = f.readline().replace("\n","").split(" ")

    # Get idx in all thoses cols
    idx_of_cols = {}
    start_ang = -1
    end_ang = -1
    start_vel = -1
    end_vel = -1

    for idx, col in enumerate(cols):
        if "_q_" in col:
            end_ang = idx + 1
            if start_ang < 0:
                start_ang = idx

            idx_of_cols[col] = idx - start_ang

        if "_qd_" in col:
            end_vel = idx + 1
            if start_vel < 0:
                start_vel = idx
            idx_of_cols[col] = idx - start_vel

    time = raw_data[:, 0].astype(float)
    angle = -raw_data[:, start_ang:end_ang].astype(float) * 180 / pi
    velocity = -raw_data[:, start_vel:end_vel].astype(float) * 180 / pi

    leading = "Right"
    if not full_trial:
        config = read_config_devillez(subject)

        if main_trial not in config:
            print(f"trial {main_trial} not found in config")
            return array([]), array([]), array([]), [], {}, None

        trial_config = config[main_trial]
        if sub_trial not in trial_config["trials"]:
            print(f"Sub trial {sub_trial} not found in config")
            return array([]), array([]), array([]), [], {}, None


        if "leading" in trial_config:
            leading = trial_config["leading"]


        trial_config = trial_config["trials"][sub_trial]
        if isinstance(trial_config, list):
            trial_config_time = trial_config
        else:
            trial_config_time = trial_config["time"]
            if "leading" in trial_config:
                leading = trial_config["leading"]

        idx_bool = logical_and(time > trial_config_time[0], time < trial_config_time[1])

        time = time[idx_bool]
        angle = angle[idx_bool]
        velocity = velocity[idx_bool]

    return time, angle, velocity, cols, idx_of_cols, leading

if __name__ == "__main__":
    subjects = list_subject(ds.DEVILLEZ)
    for subject in subjects:
        trials = list_trial(subject, ds.DEVILLEZ)
        print(f"Subject {subject} has the following trials:")
        for trial in trials:
            print(f"- {trial}")
        print()
