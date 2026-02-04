"""
This module contains the AO class representing a AO estimator

Description
-----------
The AO estimator is designed to estimate the position and the velocity of a hip
using adaptive oscillators.

Example
-------
> dm = DataManager()
> res = dm.load_data("P01", "exp_transition_03-S3_01
> ao = AO(dm)
> ao.compute()
> fig, axs, = ao.complete_plot()
"""

from numpy import array, zeros, arange, pi, sin, cos
from numpy import sum as npsum
from numpy import abs as npabs

from matplotlib.pyplot import Axes, axes, Figure, show
from math import sqrt

from estimator import Estimator
from data_manager import DataManager, DS_TYPE

import colors


# Frequency bound for the AO estimator
LOW_BOUND = 0.5
HIGH_BOUND = 5e3

class AO(Estimator):
    """
    Class representing a AO estimator.

    The AO estimator is designed to estimate the position and the velocity of a hip

    Attributes
    ----------
    n_osc: int
        Number of oscillators
    omega: float
        Frequency of the oscillators
    alphas: array
        Amplitude of the oscillators
    phis: array
        Phase of the oscillators
    nu_phi: float
        Learning gain of the phase
    nu_omega: float
        Learning gain of the frequency
    eta: float
        Learning gain of the amplitude
    convergence: float
        Convergence of the estimator
    """

    n_osc: int

    omega: float
    alphas: array
    phis: array

    nu_phi: float
    nu_omega: float
    eta: float
    convergence: float

    __old_eval: array
    _omegas_data: array


    def __init__(self, source_of_data: DataManager, n_osc: float = 2) -> None:
        """
        Initialize a AO estimator with a data manager

        Parameters
        ----------
        source_of_data : DataManager
        """

        super().__init__("AO", source_of_data)

        self.n_osc = n_osc
        self.reset()

        # Someting a bit longer than one cycle
        period = 2
        self.set_gains_period(period)

        self.output_data_col = [
                "Prosthetic Position AO",
                "Prosthetic Velocity AO",
                "Intact Position AO",
                "Intact Velocity AO",
                "Convergence"
                ]


    def set_gains_period(self, period: float) -> None:
        """
        Set the gains of the AO estimator

        Parameters
        ----------
        period : float
            Period of learing for the oscillators
        """
        self.nu_omega = 20 / (0.5 * period) **2
        self.nu_phi = sqrt(24.2 * self.nu_omega)
        self.eta = 2 / (0.5 * period)

    def set_gains(self, nu_omega: float, nu_phi: float, eta: float) -> None:
        """
        Set the gains of the AO estimator

        Parameters
        ----------
        nu_omega : float
            Learning gain of the frequency
        nu_phi : float
            Learning gain of the phase
        eta : float
            Learning gain of the amplitude
        """

        self.nu_omega = nu_omega
        self.nu_phi = nu_phi
        self.eta = eta

    def reset(self, frequency:float=LOW_BOUND, amplitude:float=1) -> None:
        """
        Reset the AO estimator

        Parameters
        ----------
        frequency : float
            Frequency of the oscillator
        amplitude : float
            Amplitude of the oscillators
        """

        self.alphas = zeros(self.n_osc+1)
        self.alphas[0] = amplitude
        self.omega = frequency

        self.phis = zeros(self.n_osc)
        self.convergence = 0

        self.__old_eval = zeros(2)

    def evaluate(self, offset:float = 0) -> array:
        """
        Evaluate the AO estimator

        Parameters
        ----------
        offset : float
            Offset of the estimator

        Returns
        -------
        array
            Evaluation of the AO estimator
        """

        suite = arange(1, self.n_osc + 1)
        phase = self.phis
        if offset:
            phase = self.phis + pi * suite

        alpha_bis = self.omega * self.alphas[1:] * suite

        self.__old_eval[0] = self.alphas[0] + (self.alphas[1:].T @ sin(phase))
        self.__old_eval[1] = (alpha_bis.T @ cos(phase))

        return self.__old_eval

    def step(self, theta, delta_t) -> None:
        """
        Do a learning step of the AO estimator

        Parameters
        ----------
        theta : float
            Angle of the hip
        delta_t : float
            Time step
        """

        # Compute the convergence / error
        theta_est = self.evaluate()
        epsilon = theta - theta_est[0]


        # Compute the frequency of each OAs
        freq = arange(1, self.n_osc + 1) * self.omega

        # Precompute the sum of the alphas
        sum_alpha = npsum(self.alphas)

        # get variation of variable
        phi_dot = freq + self.nu_phi * epsilon * cos(self.phis) / sum_alpha
        omega_dot = self.nu_omega * epsilon * cos(self.phis[0]) / sum_alpha
        alpha_dot = self.eta * epsilon * sin(self.phis)
        alpha_0_dot = self.eta * epsilon

        # Avoid large change for small alphas
        if abs(sum_alpha) < 3e-2:
            phi_dot = freq
            omega_dot = 0

        # Learn
        self.phis += delta_t * phi_dot
        self.omega += delta_t * omega_dot
        self.alphas[1:] += delta_t * alpha_dot
        self.alphas[0] += delta_t * alpha_0_dot

        # Limit the frequency

        self.omega = max(self.omega, LOW_BOUND)
        self.omega = min(self.omega, HIGH_BOUND)

        return epsilon

    def get_convergences(self) -> array:
        """
        Get the convergence of the estimator

        Returns
        -------
        array
            Convergence of the estimator
        """

        return npabs(self.output_data[:, 4])

    def estimate(self, input_data: array, dt: float) -> None:
        """
        Estimate the output data from the input data

        Parameters
        ----------
        input_data : array
            Input data
        dt : float
            Time step

        Returns
        -------
        array
            Estimate data from the input data
        """

        # Preallocate output
        n_data = len(input_data)
        data_out = zeros((n_data, 5))

        # Save the first step without stepping
        data_out[0, 2:4] = self.evaluate(offset=0)
        data_out[0, :2] = self.evaluate(offset=0.5)
        data_out[0, 4] = self.convergence

        self.omegas_data = zeros((n_data, 1))
        self.omegas_data[0, 0] = self.omega
        for i in range(1, n_data):
            # Compute the learning step
            data_out[i, 4] = self.step(input_data[i-1, 0], dt)

            # Get the estimation of the leading leg
            data_out[i, 2:4] = self.__old_eval

            # Compute the estimation of the following leg
            data_out[i, :2] = self.evaluate(offset=0.5)

            # Get Omega
            self.omegas_data[i, 0] = self.omega


        return data_out

    def compute(self) -> None:
        """
        Compute the resulting estimation.

        Store the result in the `output_data` attribute
        """

        # Reset the estimator
        self.reset()

        # Load the data from the data manager
        data_in = self.data_manager.get_data(
                ["q_r_hip"]
                )


        # Estimate the data
        self.output_data = self.estimate(data_in, self.data_manager.get_dt())

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


        ## Leg, following or leading
        if "leg" in options:
            options_leg = options["leg"]
        else:
            options_leg = "following"

        is_leading = False
        if options_leg == "leading":
            offset += 2
            is_leading = True

        if "cycles" not in options:
            options["cycles"] = False

        # Display position/velocity for the estimator or the true leg
        if options_type in ["position", "velocity"]:


            data_source = self.data_manager.get_data([
                "time",
                f"q{'d' if is_velocity else ''}_{'r' if is_leading else 'l'}_hip"
                ])

            self._overide_plot(axs, data_source[:,0], data_source[:,1], use_cycler=options["cycles"], label=f"{'Intact' if is_leading else 'Prosthetic'} {'velocity' if is_velocity else 'Position'} truth")
            self._overide_plot(axs, data_source[:,0], self.output_data[:,offset], use_cycler=options["cycles"], label=self.output_data_col[offset], linestyle="--")

            axs.set_xlabel("Time (s)")

            if not is_velocity:
                axs.set_ylabel("Position (deg)")
            else:
                axs.set_ylabel("Velocity (deg/s)")
        elif options_type == "convergence":

            data_source = self.data_manager.get_data([
                "time",
                ])

            self._overide_plot(axs, data_source[:,0], self.output_data[:,4], use_cycler=options["cycles"], label=self.output_data_col[4])
            axs.set_xlabel("Time (s)")

            axs.set_ylabel("Error/Convergence (deg)")
        elif options_type == "omega":
            time = self.data_manager.time

            self._overide_plot(axs, time, self.omegas_data[:,0], use_cycler=options["cycles"], label="Omega")
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Omega (rad/s)")

        else:
            raise ValueError(f"Unknown type {options_type}. Must be one of ['position', 'velocity', 'convergence']")

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
            - convergence : Convergence of the estimator
        leg : str
            Leg to plot. Can be "leading" or "following". Default is "following"
        """

    def get_help_option_complete_plot(self) -> str:
        """ Get the help option for the complete_plot function

        Returns
        -------
        str
            Help string for the complete_plot function
        """

        return """
        Options
        -------
        cycles : bool
            If True, plot each cycle separately
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
            **options
            })
        self.plot(axs[0], options={
            "type": "position",
            "leg": "leading",
            **options
            })
        self.color_cycler = colors.get_color_cycler()
        self.plot(axs[1], options={
            "type": "velocity",
            **options
            })
        self.color_cycler = colors.get_color_cycler()
        self.plot(axs[2], options={
            "type": "omega",
            **options
            })
        self.color_cycler = colors.get_color_cycler()
        self.plot(axs[3], options={
            "type": "convergence",
            **options
            })



        return fig, axs

if __name__ == "__main__":

    dm = DataManager()
    res = dm.load_data("P01", "exp_transition_03-S3_01", DS_TYPE.DEVILLEZ)

    if not res:
        print("Failed to load data")
        exit(1)

    ao = AO(dm)
    ao.compute()

    fig, axs, = ao.complete_plot(None, {"cycles": False})
    show()
