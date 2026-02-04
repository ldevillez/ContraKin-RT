"""
This module regroups the utility functions related to intervals.

Description
-----------
This module provides functions and classes to manage and manipulate time intervals
and tasks within a dataset. It includes functionalities to grow intervals, extract
intervals from boolean masks boolean arrays, and manage tasks with associated
masks.

Example
-------
> tm = TaskManager(data_manager)
> intervals_standing = tm.get_interval(Task.Standing.value)
> data_standing = tm.apply_masks_to_data(data, [Task.Standing.value])
> tm.plot(ax, time)
"""

import enum

from numpy import (
    zeros,
    array,
    nonzero,
    logical_xor,
    logical_and,
    asarray,
    logical_or,
)
from numpy import abs as npabs
from numpy import max as npmax

from matplotlib.pyplot import figure, show

from support import moving_average_and_extend

import colors


def get_interval(mask: array) -> array:
    """
    Get the intervals of the mask
    Intervals of size 1 are removed.

    Parameters
    ----------
    mask : array
        Mask to get the intervals from

    Returns
    -------
    array
        Intervals of the mask

    Raises
    ------
    ValueError
        If the function cannot find the start and end of all intervals
    """

    intervals = []
    if mask[0] and mask[1]:
        intervals.append(0)

    intervals.extend(
        nonzero(
            logical_and(
                logical_xor(
                    mask[1:-1] != mask[:-2], mask[1:-1] != mask[2:]
                ),  # remove one size
                mask[1:-1],  # keep only the ones that are True
            )
        )[0]
    )

    if mask[-1] and mask[-2]:
        intervals.append(len(mask) - 1)

    if len(intervals) % 2 != 0:
        raise ValueError("The sum of start and end shouldbe an even number of elements")

    intervals = array(intervals)
    intervals = intervals.reshape(-1, 2)

    return intervals


def get_first_and_last_threshold(
    mask: array, time: array, threshold: float = 0.5, max_time_lookout: float = 15
) -> tuple:
    """
    Get the Idx of the first and last threshold crossing in the mask.
    The threshold is defined as a percentage of the maximum value of the mask.

    Parameters
    ----------
    mask : array
        Mask to get the threshold crossings from
    time : array
        Time array corresponding to the mask
    threshold : float
        Threshold percentage to use

    Returns
    -------
    tuple
        Tuple of the first and last threshold crossing Idx.
    """

    if len(mask) == 0:
        return None, None

    mask = npabs(asarray(mask, dtype=float))

    max_value_first = npmax(mask[time < time[0] + max_time_lookout])
    max_value_last = npmax(mask[time[-1] - max_time_lookout < time])

    if max_value_first == 0 or max_value_last == 0:
        return None, None

    threshold_value_first = threshold * max_value_first
    threshold_value_last = threshold * max_value_last

    idxs_first = nonzero(mask >= threshold_value_first)[0]
    idxs_last = nonzero(mask >= threshold_value_last)[0]

    return idxs_first[0], idxs_last[-1]


class Task(enum.Enum):
    """
    Enum to define the tasks managed by the TaskManager.
    """

    Transition_IN = "Transition_IN"
    Transition_OUT = "Transition_OUT"
    Transition = "Transition"
    Standing = "Standing"
    Walking = "Walking"

    @classmethod
    def get_all_tasks(cls):
        """
        Get all tasks defined in the Task enum.

        Returns
        -------
        list
            List of all task names
        """
        return [task.value for task in cls]


class TaskManager:
    """
    Class to manage tasks in a DataManager.

    Attributes
    ----------

    """

    dm: object  # To fix cyclic import issue
    threshold_transition: float

    task_masks: dict
    n_elem: int

    def __init__(self, data_manager: object, threshold_transition: float = 5):
        """
        Initialize a TaskManager.

        Parameters
        ----------
        data_manager : DataManager
            DataManager to manage tasks for
        threshold_transition : float
            Threshold for the transition between standing and walking
        """
        self.dm = data_manager
        self.threshold_transition = threshold_transition

        self.task_masks = {}

        self.get_tasks_mask()

    @property
    def task_names(self) -> list:
        """
        Get the names of the tasks managed by this TaskManager.

        Returns
        -------
        list
            List of task names
        """
        return list(self.task_masks.keys())

    def get_tasks_mask(self) -> dict:
        """
        Divide the tasks into a dictionary of tasks and their respective mask.

        The tasks extracted are:
        - "Transition": between standing and walking
        - "Standing": standing task
        - "Walking": walking task

        Returns
        -------
        dict
            Dictionary of tasks with the task name as key and the time mask
        """

        # Get q hip leading
        data_source = self.dm.get_data(["time", "qd_r_hip", "qd_l_hip"])

        self.n_elements = len(data_source)

        data_filtered_leading = moving_average_and_extend(
            data_source[:, 1], 15
        ).reshape(-1)
        data_filtered_following = moving_average_and_extend(
            data_source[:, 2], 15
        ).reshape(-1)

        # Maybe taking directly the max value is too harsh if there is one outlier
        first_transition_start_leading, last_transition_end_leading = (
            get_first_and_last_threshold(data_filtered_leading, data_source[:, 0], 0.1)
        )
        first_transition_end_leading, last_transition_start_leading = (
            get_first_and_last_threshold(data_filtered_leading, data_source[:, 0], 0.75)
        )

        first_transition_start_following, last_transition_end_following = (
            get_first_and_last_threshold(
                data_filtered_following, data_source[:, 0], 0.1
            )
        )
        first_transition_end_following, last_transition_start_following = (
            get_first_and_last_threshold(
                data_filtered_following, data_source[:, 0], 0.75
            )
        )

        if (
            first_transition_start_leading is None
            or last_transition_end_leading is None
        ):
            raise ValueError(
                "Not enough data to define tasks. Check the data source or the threshold."
            )

        if (
            first_transition_end_leading is None
            or last_transition_start_leading is None
        ):
            raise ValueError(
                "Not enough data to define tasks. Check the data source or the threshold."
            )

        if (
            first_transition_start_following is None
            or last_transition_end_following is None
        ):
            raise ValueError(
                "Not enough data to define tasks. Check the data source or the threshold."
            )

        if (
            first_transition_end_following is None
            or last_transition_start_following is None
        ):
            raise ValueError(
                "Not enough data to define tasks. Check the data source or the threshold."
            )

        first_transition_start = min(
            first_transition_start_leading, first_transition_start_following
        )
        first_transition_end = max(
            first_transition_end_leading, first_transition_end_following
        )

        last_transition_start = min(
            last_transition_start_leading, last_transition_start_following
        )
        last_transition_end = max(
            last_transition_end_leading, last_transition_end_following
        )

        mask_standing = zeros(len(data_filtered_leading), dtype=bool)
        mask_transition_in = zeros(len(data_filtered_leading), dtype=bool)
        mask_transition_out = zeros(len(data_filtered_leading), dtype=bool)
        mask_walking = zeros(len(data_filtered_leading), dtype=bool)

        mask_standing[:first_transition_start] = True
        mask_transition_in[first_transition_start:first_transition_end] = True
        mask_walking[first_transition_end:last_transition_start] = True
        mask_transition_out[last_transition_start:last_transition_end] = True
        mask_standing[last_transition_end:] = True

        mask_transition = logical_or(mask_transition_in, mask_transition_out)

        for task_name in Task.get_all_tasks():
            self.task_masks[task_name] = zeros(len(mask_standing), dtype=bool)

        self.task_masks[Task.Transition_IN.value] = mask_transition_in
        self.task_masks[Task.Transition_OUT.value] = mask_transition_out
        self.task_masks[Task.Standing.value] = mask_standing
        self.task_masks[Task.Walking.value] = mask_walking
        self.task_masks[Task.Transition.value] = mask_transition

    def get_interval(self, name_task) -> array:
        """
        Get the intervals of the task
        Parameters
        ----------
        name_task : str
            Name of the task to get the intervals for

        Returns
        -------
        array
            Intervals of the task
        """

        if name_task not in self.task_masks:
            raise ValueError(f"Task {name_task} not found")

        mask = self.task_masks[name_task]

        return get_interval(mask)

    def get_flattened_mask(self) -> array:
        """
        Get a flattened mask of all tasks.

        Returns
        -------
        array
            Flattened mask of all tasks
        """

        mask_flattened = zeros(len(self.task_masks[Task.Transition.value]))

        for idx, task_name in enumerate(Task.get_all_tasks()):
            mask_flattened[self.task_masks[task_name]] = (
                idx + 1
            )  # Start from 1 to avoid confusion with False (0)

        return mask_flattened

    def get_flattened_names(self) -> list:
        """
        Get the names of the tasks in the flattened mask.

        Returns
        -------
        list
            List of task names in the flattened mask
        """

        names = [""] * len(self.task_masks[Task.Transition.value])
        flattened = self.get_flattened_mask()

        all_tasks = Task.get_all_tasks()
        for idx, value in enumerate(flattened):
            if value > 0:
                names[idx] = all_tasks[int(value - 1)]

        return names

    def apply_masks_to_data(self, data: array, masks: list = []) -> array:
        """
        Apply the task masks to the data.

        Parameters
        ----------
        data : array
            Data to apply the masks to
        masks : list
            List of task names to apply the masks to. If empty, all masks are applied.

        Returns
        -------
        array
            Data with the masks applied
        """

        if self.n_elements != len(data):
            raise ValueError(
                "Data length does not match the number of elements in the TaskManager"
            )

        empty_mask = zeros(len(data), dtype=bool)

        for mask in masks:
            if isinstance(mask, Task):
                mask = mask.value
            empty_mask |= self.task_masks[mask]

        return data[empty_mask]

    def plot(self, axs, time):
        """
        Plot the tasks on the given axes.
        """

        color_cycler = colors.get_color_cycler()

        for task_name in self.task_names:
            if task_name == Task.Transition.value:
                continue
            intervals = self.get_interval(task_name)
            for i in range(len(intervals)):
                axs.axvspan(
                    time[intervals[i, 0]],
                    time[intervals[i, 1]],
                    **color_cycler[task_name],
                    alpha=0.2,
                    label="tsk " + task_name,
                )

    def add_legend(self, ax):
        """
        Add a legend to the given axes.
        """

        handles_raw, labels_raw = ax.get_legend_handles_labels()

        handles = []
        labels = []
        for label, handel in zip(labels_raw, handles_raw):
            if "tsk" not in label:
                continue
            label_processed = label.replace("tsk ", "")
            if label_processed not in labels:
                labels.append(label_processed)
                handles.append(handel)

        lg = ax.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 1.1),
            loc="lower center",
            ncol=len(handles),
        )
        ax.add_artist(lg)


if __name__ == "__main__":
    subject_name = "P01"
    trial_name = "exp_transition_03-S3_02"

    from data_manager import DataManager, DS_TYPE

    dm = DataManager()
    res = dm.load_data(subject_name, trial_name, DS_TYPE.DEVILLEZ)

    if res:
        tm = TaskManager(dm)

        # List tasks start and end
        for task_name in tm.task_names:
            intervals = tm.get_interval(task_name)
            print(f"Task: {task_name}")
            for i in range(len(intervals)):
                print(
                    f"  Start: {dm.time[intervals[i, 0]]:.3f} s, End: {dm.time[intervals[i, 1]]:.3f} s"
                )

        # Plot tasks
        fig = figure()
        ax = fig.add_subplot(111)
        tm.plot(ax, dm.time)
        tm.add_legend(ax)
        show()

    else:
        print("Data not found")
