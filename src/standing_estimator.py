"""
This module contains the class Standing, which is a copy estimator with an exponential filter to smooth the data (low-pass filter).

Description
-----------
The Standing class is an estimator that copies the input data and applies an exponential filter to smooth it


Example
-------
> dm = DataManager()
> res = dm.load_data("P01", "exp_transition_03-S3_01
> standing = Standing(dm)
> standing.compute()
> fig, axs, = standing.complete_plot()
"""

from estimator import Estimator
from data_manager import DataManager, DS_TYPE
from matplotlib.pyplot import Axes, axes, Figure, show
from numpy import zeros

from filter import ExponentialFilter


class Standing(Estimator):
    """
    Class representing a copy estimator.

    An exponential filter is added to the copy estimator to smooth the data (low-pass filter)


    Attributes
    ----------
    filter_output : ExponentialFilter
        Exponential filter to smooth the output data
    """

    filter_output: ExponentialFilter

    def __init__(self, source_of_data: DataManager) -> None:
        """
        Initialize a copy estimator with a data manager

        Parameters
        ----------
        source_of_data : DataManager
        """

        super().__init__("Standing", source_of_data)

        self.filter_output = ExponentialFilter(0.3)

        self.output_data_col = [
                "Following Position Standing",
                "Following Velocity Standing",
                ]


    def compute(self) -> None:
        """ Compute the resulting estimation.

        Store the result in the `output_data` attribute
        """
        data_in = self.data_manager.get_data(["q_r_hip", "qd_r_hip"])
        self.output_data = zeros(data_in.shape)

        self.output_data = self.filter_output.apply(data_in)



    def plot(self, axs: None | Axes = None, options: dict = {}) -> Axes:
        """ Plot the output data

        Parameters
        ----------
        axs : None | plt.Axes
            Axes to plot the data. If none a new axes is created
        options : dict
            Dictionary of options to plot the data
        """

        if axs is None:
            axs = axes()

        # Processing the options

        ## Position or velocity
        if "type" in options:
            options_type = options["type"]
        else:
            options_type = "position"

        offset = 0
        is_velocity = False
        if options_type == "velocity":
            offset += 1
            is_velocity = True


        # Display position/velocity for the estimator or the true leg
        if options_type in ["position", "velocity"]:


            data_source = self.data_manager.get_data([
                "time",
                f"q{'d' if is_velocity else ''}_l_hip",
                f"q{'d' if is_velocity else ''}_r_hip",
                ])

            axs.plot(data_source[:,0], data_source[:,1], label="Following Position True")
            axs.plot(data_source[:,0], data_source[:,2], label="Leading Position True")
            axs.plot(data_source[:,0], self.output_data[:,offset], label=self.output_data_col[offset], linestyle="--")
            axs.set_xlabel("Time (s)")

            if not is_velocity:
                axs.set_ylabel("Position (deg)")
            else:
                axs.set_ylabel("Velocity (deg/s)")

        if options_type == "error":
            self.errors()
            axs.plot(self.data_manager.get_data(["time"])[:,0], self.error_data[:,0], label="Position Error")
            # axs.plot(self.error_data[:,0],self.error_data[:,1], label="Position Error")

        return axs


    def get_help_option_plot(self) -> str:
        """ Get the help option for the plot function

        Returns
        -------
        str
            Help string for the plot function
        """

        return """
        Options
        -------
        type : str
            Type of the plot. Can be "position" or "velocity". Default is "position"
        """

    def _complete_plot(self, fig: None | Figure = None, options: dict = {}) -> tuple[Figure, list[Axes]]:
        """ Complete plot of the estimator

        Parameters
        ----------
        fig : None | Figure
            Figure to plot the data. If none a new figure is created
        options : dict
            Dictionary of options to plot the data
        """

        axs = fig.subplots(2, 1)

        self.plot(axs[0], options={
            "type": "position",
            })
        self.plot(axs[1], options={
            "type": "velocity",
            })

        return fig, axs

if __name__ == "__main__":

    dm = DataManager()
    res = dm.load_data("P01", "exp_transition_03-S3_01", DS_TYPE.DEVILLEZ)

    if not res:
        print("Failed to load data")
        exit(1)

    standing = Standing(dm)
    standing.compute()

    fig, axs, = standing.complete_plot(None, {"cycles": False})
    show()
