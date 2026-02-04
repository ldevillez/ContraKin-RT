"""
Module to decompose gait data into cycles based on hip joint angle.

Description
-----------

This module defines a Cycler class that processes gait data to identify and extract individual gait cycles.
Each cycle is determined by detecting peaks in the hip joint angle data, allowing for analysis on a cycle-by-cycle basis.

Example
-------
> dm = DataManager()
> dm.load_data("P01", "exp_transition_03-S3_01", DS
> tm = TaskManager(dm)
> cycler = Cycler(dm)
> walking_task = tm.get_interval("Walking")[0]
> cycles = cycler.get_cycles(start_index=walking_task[0], end_index=walking_task[1])

"""

from data_manager import DataManager, DS_TYPE
from tasks import TaskManager
from numpy import min as npmin
from numpy import max as npmax
from numpy import where, argmax


class Cycler:
    """
    Class to decompose a gait data on a cycle by cycle basis
    """

    data_manager: DataManager
    cycles: list

    def __init__(self, data_manager):
        """
        Initialize the Cycler with a DataManager instance.

        Parameters:
        data_manager (DataManager): An instance of DataManager containing gait data.
        """

        self.data_manager = data_manager

    def get_cycles(self, start_index=0, end_index=-1):
        """
        Decomposes the gait data into cycles.

        Returns:
        list: A list of cycles, where each cycle is a list of data points.
        """

        data = self.data_manager.get_data(["q_r_hip"])

        max_val = npmax(data[start_index:end_index])
        min_val = npmin(data[start_index:end_index])
        threshold = (max_val + min_val) / 2

        searchable_data = data > threshold

        searchable_data = searchable_data.astype(int)
        edges = searchable_data[1:] - searchable_data[:-1]
        edges_idx = where(edges > 0)[0] + 1

        idx_max_list = []
        for i, j in zip(edges_idx[:-1], edges_idx[1:]):
            if not searchable_data[i]:
                continue

            potential_max = argmax(data[i:j]) + i

            if potential_max < start_index or (
                end_index != -1 and potential_max > end_index
            ):
                continue

            idx_max_list.append(potential_max)

        self.cycles = idx_max_list

        return self.cycles

    def get_cycle(self, idx_cycle):
        """
        Get a specific cycle by its index.
        """
        if idx_cycle >= len(self.cycles) - 1:
            raise IndexError("Cycle index out of range")

        start = self.cycles[idx_cycle]
        end = self.cycles[idx_cycle + 1]
        return (start, end)

    def cycle_idxs_generator(self, use_all_idxs: bool = False):
        """
        Generator to yield cycle indexes.
        """

        if use_all_idxs:
            return [(0, len(self.data_manager.time))]

        for i in range(len(self.cycles) - 1):
            start = self.cycles[i]
            end = self.cycles[i + 1]
            yield (start, end)


if __name__ == "__main__":
    # Example usage
    dm = DataManager()
    dm.load_data("P01", "exp_transition_03-S3_01", DS_TYPE.DEVILLEZ)
    tm = TaskManager(dm)

    cycler = Cycler(dm)
    walking_task = tm.get_interval("Walking")[0]
    cycles = cycler.get_cycles(start_index=walking_task[0], end_index=walking_task[1])

    print(f"{dm.subject} - {dm.trial} - {dm.ds_name}")
    for idx in range(len(cycles) - 1):
        start, end = cycler.get_cycle(idx)
        t_s, t_e = dm.time[start], dm.time[end]

        print(
            f"Cycle {idx}: from {t_s:.3f} ({start}) to {t_e:.3f} ({end}) | Duration: {t_e - t_s:.3f} s"
        )
