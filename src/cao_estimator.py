"""
Module defining the combined CLME-AO estimator

Description
-----------
The ClmeAO class is an estimator that combines the CLME and AO estimators using logistic
functions to smoothly transition between the two based on convergence metrics.

Example
-------
> dm = DataManager()
> clme = CLME(dm)
> ao = AO(dm)
> clme_ao = ClmeAO(dm, clme, ao)
> clme_ao.compute()
> fig, axs, = clme_ao.complete_plot()
"""


from matplotlib.pyplot import Axes, axes, Figure, show
from numpy import array, zeros
from numpy import abs as npabs

from estimator import Estimator
from clme_estimator import CLME
from ao_estimator import AO
from data_manager import DataManager, DS_TYPE

from logistic import Logistic, TimeLogistic
from filter import AsymExpFilter
import colors

class ClmeAO(Estimator):
    """
    Class representing the combined estimator between a CLME and AO

    The estimators are merged with a Logistic function

    Attributes
    ----------
    clme : CLME
        The CLME estimator
    ao : AO
        The AO estimator
    logistics : list[Logistic]
        List of logistic functions to smoothly transition between the two estimators based on convergence metrics and time
    output_data : array
        The output data of the estimator, which is a combination of the CLME and AO estim
    logistic_values : array
        The values of the logistic functions used to combine the two estimators
    convergence_filter : AsymExpFilter
        Asymmetrical exponential filter to smooth the convergence metrics used to compute the logistic functions
    convergences_filtered : array
        The filtered convergence metrics used to compute the logistic functions

    """

    clme: CLME
    ao: AO
    logistics: list[Logistic]

    output_data: array
    logistic_values : array

    convergence_filter: AsymExpFilter
    convergences_filtered: array


    def __init__(self, source_of_data: DataManager, clme: CLME, ao: AO) -> None:
        """
        Initialize a ClmeAO estimator with a data manager

        Parameters
        ----------
        source_of_data : DataManager
        """

        super().__init__("Clme-AO", source_of_data)

        self.clme = clme
        self.ao = ao

        self.logistics = [
                Logistic(0.8422, 6.591, None, inverse=False),
                TimeLogistic(4, 1.5),
                Logistic(0.83299, 4.597, None, inverse=False),
                TimeLogistic(3, 1.25),
                ]

        self.output_data_col = [
                "Following Position CLME-AO",
                "Following Velocity CLME-AO",
                ]

        self.convergence_filter = AsymExpFilter(0.992)

    @property
    def sub_estimators(self):
        """ Sub-estimators of the estimator

        Returns
        -------
        list
            List of sub-estimators
        """
        return [self.clme, self.ao]

    def compute(self) -> None:
        """ Compute the resulting estimation."""

        self.clme.compute()
        self.ao.compute()

        # Compute the logistic function

        convergence = self.ao.get_convergences()
        self.convergences_filtered = self.convergence_filter.apply(npabs(convergence))

        self.output_data = zeros((self.clme.output_data.shape[0], 2))
        self.logistic_values = zeros((self.clme.output_data.shape[0], len(self.logistics)+2))

        # Time and position_logistic
        self.logistics[1].reset_time(self.data_manager.time[0])
        self.logistics[3].reset_time(self.data_manager.time[0])

        for idx_time, t in enumerate(self.data_manager.time):

            for i in range(2):
                # Compute the logistic function
                idx_log = 2*i
                idx_time_log = 2*i+1

                self.logistic_values[idx_time,idx_log] = self.logistics[idx_log].compute(self.convergences_filtered[idx_time])
                # Time logistic
                if self.logistic_values[idx_time,idx_log] < 0.5:
                    self.logistics[idx_time_log].reset_time(t)

                self.logistic_values[idx_time,idx_time_log] = self.logistics[idx_time_log].compute(t)

                log_val = self.logistic_values[idx_time,idx_log] * self.logistic_values[idx_time,idx_time_log]

                self.logistic_values[idx_time,4+i] =log_val

                self.output_data[idx_time,i] = log_val * self.ao.output_data[idx_time,i] + (1 - log_val) * self.clme.output_data[idx_time,i]



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


        if options_type in ["position", "velocity"]:
            data_source = self.data_manager.get_data(["time", "q_l_hip", "qd_l_hip"])

            axs.plot(data_source[:,0], data_source[:,1+offset], label=f"Following {'Velocity' if is_velocity else 'Position'}", linewidth=4)
            axs.plot(data_source[:,0], self.output_data[:, 0+offset], label=self.output_data_col[offset], linewidth = 3)
            axs.plot(data_source[:,0], self.clme.output_data[:,0 + offset], label=self.clme.output_data_col[offset], linewidth = 2, linestyle="--")
            axs.plot(data_source[:,0], self.ao.output_data[:,0 + offset], label=self.ao.output_data_col[offset],linewidth = 1)


            if not is_velocity:
                axs.set_ylabel("Position (deg)")
            else:
                axs.set_ylabel("Velocity (deg/s)")

        elif options_type == "logistic":

            data_source = self.data_manager.get_data(["time"])
            axs.plot(data_source[:,0], self.logistic_values[:,0], label="Logistic Position", linewidth = 4)
            axs.plot(data_source[:,0], self.logistic_values[:,1], label="Logistic Time", linewidth = 3, linestyle="--")
            axs.plot(data_source[:,0], self.logistic_values[:,4], label="Logistic Tot", linewidth = 3, linestyle="-.")

            # Set lim between 0 and 1
            axs.set_ylim(0, 1)
            axs.set_ylabel("Logistic value (·)")

        elif options_type == "convergence":

            data_source = self.data_manager.get_data(["time"])
            axs.plot(data_source[:,0], self.ao.get_convergences(), label="Convergence non-filtered", linewidth = 4)
            axs.plot(data_source[:,0], self.convergences_filtered, label="Convergence filtered", linewidth = 3, linestyle="--")

            axs.set_ylabel("Convergence (deg)")

        axs.set_xlabel("Time (s)")

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
            Type of plot to display
            - position : Position of the leg
            - velocity : Velocity of the leg
            - logistic : Logistic value
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

        axs = fig.subplots(4, 1)

        self.color_cycler = colors.get_color_cycler()
        self.plot(axs[0], options={
            "type": "position",
            })
        self.plot(axs[1], options={
            "type": "logistic",
            })
        self.plot(axs[2], options={
            "type": "convergence",
            })
        self.color_cycler = colors.get_color_cycler()
        self.ao.color_cycler = self.color_cycler
        self.ao.plot(axs[3], options={
            "leg": "leading",
            })


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
    clme_ao.compute()
    fig, axs = clme_ao.complete_plot()
    show()
