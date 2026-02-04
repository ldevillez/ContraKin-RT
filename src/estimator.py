"""
This module contains the Estimator class. To be extended by sub-classes.

Description
-----------
The Estimator class is a base class for all estimators. It provides methods to compute the
estimation, compute errors, and plot the results. Sub-classes should implement the compute
method to perform the actual estimation.

Example
-------
> estimator = Estimator("MyEstimator", data_manager)
> estimator.compute()
> estimator.errors()
> fig, axs = estimator.complete_plot()
"""

from numpy import array
from data_manager import DataManager
from matplotlib.pyplot import Figure, figure, Axes
from tasks import TaskManager
from cycler_decomposition import Cycler
import colors


class Estimator:
    """
    Class representing an estimator

    Attributes
    ----------
    name : str
        Name of the estimator
    data_manager : DataManager
        Data manager to load the data
    output_data : array
        Output data of the estimator
    output_data_col : list
        List of output data columns
    """

    name: str
    data_manager: DataManager
    output_data: array
    error_data: array
    task: TaskManager
    cycler: Cycler
    color_cycler: dict

    def __init__(self, name: str, source_of_data: DataManager) -> None:
        """
        Initialize an estimator with a name.

        Parameters
        ----------
        name : str

        source_of_data : DataManager
        """
        self.name = name

        self.data_manager = source_of_data

        self.output_data = array([])
        self.output_data_col = []

        self.error_data = array([])

    @property
    def sub_estimators(self) -> list:
        """ Sub-estimators of the estimator

        Returns
        -------
        list
            List of sub-estimators
        """
        return []

    def compute(self) -> None:
        """ Compute the resulting estimation.

        Store the result in the `output_data` attribute
        """

    def errors(self) -> None:
        """
        Compute the errors of the estimator.

        The errors are computed as the difference between the output data and the input data.
        The result is stored in the `error_data` attribute.

        """

        data_in = self.data_manager.get_data(
                ["q_l_hip", "qd_l_hip"]
                )

        self.error_data = data_in[:,:2] - self.output_data[:,:2]

    def plot(self, axs: None | Axes = None, options: dict = {}) -> Axes:
        """ Plot the output data
        Parameters
        ----------
        axs : None | Axes
            Axes to plot the data. If none a new axes is created
        options : dict
            Dictionary of options to plot the data
        """

        return axs if axs is not None else Axes()

    def get_help_option_plot(self) -> str:
        """ Get the help option for the plot function

        Returns
        -------
        str
            Help string for the plot function
        """

        return "No help available"

    def _overide_plot(self, ax, x, y, use_cycler = False, **args):
        """
        Override the plot function to customize the plot.
        """

        if "color" not in args:
            style = self.color_cycler[args["label"] + self.data_manager.subject + self.data_manager.trial]
            args.update(style)


        if use_cycler:
            cycler = None
            if hasattr(self, "cycler"):
                cycler = self.cycler
            elif hasattr(self.data_manager, "cycler"):
                cycler = self.data_manager.cycler

            if cycler is None:
                raise ValueError("No cycler available")

            for idx_a, idx_b in cycler.cycle_idxs_generator():
                t = x[idx_a:idx_b]
                t = t - t[0]  # Normalize time to start at 0
                t = 100 * t / t[-1]  # Normalize time to end at 1
                ax.plot(t, y[idx_a:idx_b], **args)
        else:
            ax.plot(x, y, **args)

        return ax

    def get_help_option_complete_plot(self) -> str:
        """ Get the help option for the complete_plot function

        Returns
        -------
        str
            Help string for the plot function
        """

        return "No help available"

    def complete_plot(self, fig: None | Figure = None, options: dict = {}) -> tuple[Figure, list[Axes]]:
        """ Complete plot of the estimator
        Each estimator will redefine the sub function _complete_plot

        Parameters
        ----------
        fig : None | Figure
            Figure to plot the data. If none a new figure is created
        options : dict
            Dictionary of options to plot the data
        """

        if fig is None:
            fig = figure()

        self.task = TaskManager(self.data_manager)
        self.cycler = Cycler(self.data_manager)

        walking_task = self.task.get_interval("Walking")[0]
        self.cycler.get_cycles(start_index=walking_task[0], end_index=walking_task[1])

        if "cycles" not in options:
            options["cycles"] = False

        self.color_cycler = colors.get_color_cycler()

        fig, axs = self._complete_plot(fig, options)
        fig.subplots_adjust(right=0.8)

        for idx, ax in enumerate(axs):
            if not options["cycles"]:
                self.task.plot(ax, self.data_manager.time)
                if idx == 0:
                    self.task.add_legend(axs[0])

            handles = []
            labels = []
            handles_raw, labels_raw = ax.get_legend_handles_labels()
            for label, handel in zip(labels_raw, handles_raw):
                if "tsk" in label:
                    continue
                if label not in labels:
                    labels.append(label)
                    handles.append(handel)

            ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left")



            if idx > 0:
                ax.sharex(axs[0])


        fig.suptitle(f"{self.name} - {self.data_manager.subject} - {self.data_manager.trial}")

        self._post_plot(fig, axs, options)

        return fig, axs

    def _complete_plot(self, fig: None | Figure = None, options: dict = {}) -> tuple[Figure, list[Axes]]:
        """ Complete plot of the estimator

        Parameters
        ----------
        fig : None | Figure
            Figure to plot the data. If none a new figure is created
        options : dict
            Dictionary of options to plot the data
        """

        return fig, []

    def _post_plot(self, fig: None | Figure = None, axs: list[Axes] = [], options: dict = {}) -> tuple[Figure, list[Axes]]:
        """ Post plot of the estimator

        Parameters
        ----------
        fig : None | Figure
            Figure to plot the data. If none a new figure is created
        axs : list[Axes]
            List of axes to plot the data
        options : dict
            Dictionary of options to plot the data
        """

        return fig, axs

if __name__ == "__main__":
    print("This is a base class for estimators. It should not be run directly.")
