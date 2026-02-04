"""
Module defining a StandWalk estimator to manage standing and walking.

Description
-----------
The StandWalk class is an estimator that combines a standing estimator and a walking estimator using a logistic function to smoothly transition between the two based on the leading hip velocity.

Example
-------
> dm = DataManager()
> stand_estimator = Standing(dm)
> walk_estimator = Walking(dm)
> stand_walk = StandWalk(dm, stand_estimator, walk_estimator)
> stand_walk.compute()
> fig, axs, = stand_walk.complete_plot()
"""

from matplotlib.pyplot import Axes, axes, Figure, setp, show
from numpy import array
from numpy import abs as npabs

from estimator import Estimator
from clme_estimator import CLME
from ao_estimator import AO
from cao_estimator import ClmeAO
from standing_estimator import Standing
from data_manager import DataManager, DS_TYPE

from logistic import Logistic
from support import moving_average_and_extend
from filter import AsymExpFilter, ExponentialFilter


class StandWalk(Estimator):
    """
    Class representing a estimator to manage standing and walking.

    There is a standing estimator and a walking estimator

    Attributes
    ----------
    estimator_stand : Estimator
        The estimator for standing
    estimator_walk : Estimator
        The estimator for walking
    logistic : Logistic
        Logistic function to smoothly transition between the two estimators based on the leading hip velocity
    output_data : array
        The output data of the estimator, which is a combination of the standing and walking estimators
    logistic_values : array
        The values of the logistic function used to combine the two estimators
    filtered_input_values : array
        The filtered input values used to compute the logistic function
    """

    estimator_stand: Estimator
    estimator_walk: Estimator
    logistic: Logistic
    output_data: array

    logistic_values: array
    filtered_input_values: array
    filter_velocity_log: AsymExpFilter

    def __init__(
        self,
        source_of_data: DataManager,
        estimator_stand: Estimator,
        estimator_walk: Estimator,
    ) -> None:
        """
        Initialize a StandWalk estimator with a data manager

        Parameters
        ----------
        source_of_data : DataManager
            Data manager to load the data
        estimator_stand : Estimator
            The estimator for standing
        estimator_walk : Estimator
            The estimator for walking
        """

        super().__init__("StandWalk", source_of_data)

        self.estimator_stand = estimator_stand
        self.estimator_walk = estimator_walk

        self.logistic = Logistic(0.4, 15, None, inverse=False)

        self.output_data_col = [
            "Following Position SW",
            "Following Velocity SW",
        ]

        # Initialize the filter
        self.filter_velocity_log = AsymExpFilter(0.96)

    @property
    def sub_estimators(self):
        """Sub-estimators of the estimator

        Returns
        -------
        list
            List of sub-estimators
        """
        return [self.estimator_stand, self.estimator_walk]

    def compute(self) -> None:
        """Compute the resulting estimation."""

        input_data = self.data_manager.get_data(["qd_r_hip"])
        input_data = npabs(input_data)
        input_data = self.filter_velocity_log.apply(input_data)

        w_values = 1 - self.logistic.compute(input_data)

        self.filtered_input_values = input_data
        self.logistic_values = w_values

        # Check if the person is walking
        self.estimator_stand.compute()
        self.estimator_walk.compute()

        self.output_data = (
            w_values * self.estimator_walk.output_data
            + (1 - w_values) * self.estimator_stand.output_data
        )

    def plot(self, axs: None | Axes = None, options: dict = {}) -> Axes:
        """
        Plot the output data

        Parameters
        ----------
        axs : None | plt.Axes
            Axes to plot the data. If none a new axes is created
        options : dict
            Dictionary of options to plot the data

        """

        if axs is None:
            axs = axes()

        if "type" in options:
            options_type = options["type"]
        else:
            options_type = "position"

        offset = 0
        is_velocity = False
        if options_type == "velocity":
            offset = 1
            is_velocity = True

        # For logistic and debug
        data_source = self.data_manager.get_data(["qd_r_hip"])
        before_log = npabs(data_source[:, 0])
        filt = ExponentialFilter(0.1, base_value=0)
        input_data = filt.apply(before_log)

        if options_type in ["position", "velocity"]:
            data_source = self.data_manager.get_data(["time", "q_l_hip", "qd_l_hip"])

            axs.plot(
                data_source[:, 0],
                data_source[:, 1 + offset],
                label=f"Following {'Velocity' if is_velocity else 'Position'} True",
                linewidth=4,
            )
            axs.plot(
                data_source[:, 0],
                self.output_data[:, 0 + offset],
                label=self.output_data_col[offset],
                linewidth=3,
            )
            axs.plot(
                data_source[:, 0],
                self.estimator_walk.output_data[:, 0 + offset],
                label=self.estimator_walk.output_data_col[offset],
                linewidth=2,
                linestyle="--",
            )
            axs.plot(
                data_source[:, 0],
                self.estimator_stand.output_data[:, 0 + offset],
                label=self.estimator_stand.output_data_col[offset],
                linewidth=1,
            )

            axs.set_ylabel("Position (deg)")

        elif options_type == "logistic":
            data_source = self.data_manager.get_data(["time"])
            raw_logistic = 1 - self.logistic.compute(input_data)
            axs.plot(data_source[:, 0], raw_logistic, label="Logistic Raw", linewidth=4)
            axs.plot(
                data_source[:, 0],
                self.logistic_values,
                label="Logistic with exp filter",
                linewidth=4,
            )

            axs.set_ylabel("Gain")

        elif options_type == "debug":
            data_source = self.data_manager.get_data(["time", "q_r_hip"])

            filter_positon = moving_average_and_extend(data_source[:, 1], 10)
            deriv = (filter_positon[1:, 0] - filter_positon[:-1, 0]) / (
                data_source[1:, 0] - data_source[:-1, 0]
            )
            deriv = npabs(deriv)

            axs.plot(
                data_source[:, 0], before_log, label="Leading hip velocity", linewidth=4
            )
            axs.plot(
                data_source[:, 0],
                self.filtered_input_values,
                label="Filtered velocity (log)",
                linewidth=3,
            )
            axs.plot(
                data_source[:-1, 0],
                deriv,
                label="Abs deriv of Leading hip position",
                linewidth=1,
            )
            axs.plot(
                [data_source[0, 0], data_source[-1, 0]],
                [self.logistic.median, self.logistic.median],
                label="threshold logistic function",
                linewidth=1,
                linestyle="--",
            )

            axs.set_ylabel("Velocity (deg/s)")

        return axs

    def get_help_option_plot(self) -> str:
        """Get the help option for the plot function

        Returns
        -------
        str
            Help string for the plot function
        """

        return """
        Options
        -------
        type : str
            Type of plot to display
            - position : Position of the leg
            - velocity : Velocity of the leg
            - logistic : Logistic value
            - debug : Debug information
        """

    def _complete_plot(
        self, fig: None | Figure = None, options: dict = {}
    ) -> tuple[Figure, list[Axes]]:
        """Complete plot of the estimator

        Parameters
        ----------
        fig : None | Figure
            Figure to plot the data. If none a new figure is created
        options : dict
            Dictionary of options to plot the data
        """

        axs = fig.subplots(3, 1)

        self.plot(
            axs[0],
            options={
                "type": "position",
            },
        )
        self.plot(
            axs[1],
            options={
                "type": "logistic",
            },
        )
        self.plot(
            axs[2],
            options={
                "type": "debug",
            },
        )

        return fig, axs

    def _post_plot(
        self, fig: None | Figure = None, axs: list[Axes] = [], options: dict = {}
    ) -> tuple[Figure, list[Axes]]:
        """Post plot of the estimator

        Parameters
        ----------
        fig : None | Figure
            Figure to plot the data. If none a new figure is created
        axs : list[Axes]
            List of axes to plot the data
        options : dict
            Dictionary of options to plot the data
        """

        for i in range(len(axs) - 1):
            setp(axs[i].get_xticklabels(), visible=False)

        return fig, axs


if __name__ == "__main__":
    dm = DataManager()
    res = dm.load_data("P01", "exp_transition_03-S3_01", DS_TYPE.DEVILLEZ)

    if not res:
        print("Failed to load data")
        exit(1)

    clme = CLME(dm)
    ao = AO(dm)
    clme_ao = ClmeAO(dm, clme, ao)

    standing = Standing(dm)

    sw = StandWalk(dm, standing, clme_ao)
    sw.compute()
    fig, axs = sw.complete_plot()
    show()
