"""
This module contains the CLME class representing a CLME estimator

Description
-----------
The Complementary Limb Motion Estimator (CLME) is designed to estimate the position and velocity
of a hip prosthetic based on the motion of the intact leg. It uses predefined matrices and
Kalman filtering to provide accurate estimations.

Example
-------
> clme = CLME(data_manager)
> clme.set_offset(array([[0.0], [0.0]]))
> clme.compute()
> fig, axs = clme.complete_plot()
"""

from estimator import Estimator
from data_manager import DataManager
from numpy import array, zeros
from matplotlib.pyplot import Axes, axes, Figure, show
from filter import ExponentialFilter


class CLME(Estimator):
    """
    Class representing a CLME estimator.

    The CLME estimator is designed to estimate the position and the velocity of a hip

    Attributes
    ----------
    a_matrix : array
        A matrix of the CLME
    b_vector : array
        B vector of the CLME
    offset : array
        Offset of the CLME estimator
    k_matrix : array
        Kalman gain matrix
    _k_f_fix : array
        Kalman fix part of the F matrix
    _k_f_var : array
        Kalman variable part of the F matrix
    filter_input : ExponentialFilter
        Exponential filter to smooth the input data
    """

    a_matrix: array
    b_vector: array

    offset: array

    k_matrix: array
    _k_f_fix: array
    _k_f_var: array

    filter_input: ExponentialFilter

    def __init__(self, source_of_data: DataManager) -> None:
        """
        Initialize a CLME estimator with a data manager

        Parameters
        ----------
        source_of_data : DataManager
        """

        super().__init__("CLME-generic", source_of_data)

        # Use generic gains for the matrices
        self.a_matrix = array(
            [
                [-0.98676767, -0.06262901, -0.43863667, 0.00340058],
                [2.46809444, -1.14976942, -1.78950297, -0.47262067],
            ]
        )
        self.b_vector = array([[-4.95165438], [-63.33245674]])

        # Kalman gain matrix
        self.kalman_gain = array([[0.04885665, 0.00370226], [0.17868418, 0.44637239]])
        self.kalman_state = zeros((2, 2))

        self._k_f_fix = array([[1, 0], [0, 1]])
        self._k_f_var = array([[0, 1], [0, 0]])

        # Participant dependant -> parameter_manager.py
        self.offset = zeros((2, 1))

        self.output_data_col = [
            "Prosthetic Position CLME",
            "Prosthetic Velocity CLME",
        ]

        # Initialize the filter
        self.filter_input = ExponentialFilter(0.3)

    def set_offset(self, offset: array) -> None:
        """
        Set the offset of the CLME estimator

        Parameters
        ----------
        offset : array
            Offset of the estimator
        """

        self.offset = offset

    @property
    def offset_pos(self) -> float:
        """
        Get the offset position of the CLME estimator

        Returns
        -------
        float
            Offset position of the estimator
        """

        return self.offset[0]

    @offset_pos.setter
    def offset_pos(self, value: float) -> None:
        """
        Set the offset position of the CLME estimator

        Parameters
        ----------
        value : float
            Offset position of the estimator
        """

        self.offset[0] = value

    @property
    def offset_vel(self) -> float:
        """
        Get the offset velocity of the CLME estimator

        Returns
        -------
        float
            Offset velocity of the estimator
        """

        return self.offset[1]

    @offset_vel.setter
    def offset_vel(self, value: float) -> None:
        """
        Set the offset velocity of the CLME estimator

        Parameters
        ----------
        value : float
            Offset velocity of the estimator
        """

        self.offset[1] = value

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

        # Compute the Kalman state
        estimate = (
            self.a_matrix @ input_data.T + self.b_vector
        ).T + self.offset.reshape(-1)

        k_state = estimate[0, :]

        # Preallocate output
        n_data = len(input_data)
        data_out = zeros((n_data, 2))

        # Save the first step without kalman
        data_out[0, :] = k_state[:]

        # Compute the Kalman state for each time step
        jac_f = self._k_f_fix + self._k_f_var * dt
        for i in range(n_data - 1):
            data_out[i + 1, :] = (jac_f - self.kalman_gain) @ data_out[
                i, :
            ] + self.kalman_gain @ estimate[i + 1, :]

        data_out[:, :] = estimate[:, :]

        return data_out

    def compute(self) -> None:
        """
        Compute the resulting estimation.

        Store the result in the `output_data` attribute
        """

        # Load the data from the data manager
        data_in = self.data_manager.get_data(
            ["q_r_hip", "qd_r_hip", "q_r_knee", "qd_r_knee"]
        )

        for i in [0, 1, 2, 3]:
            data_in[:, i] = self.filter_input.apply(data_in[:, i])

        # Estimate the data
        self.output_data = self.estimate(data_in, self.data_manager.get_dt())

    def plot(self, axs: None | Axes = None, options: dict = {}) -> Axes:
        """Plot the output data

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
            data_source = self.data_manager.get_data(
                [
                    "time",
                    f"q{'d' if is_velocity else ''}_l_hip",
                    f"q{'d' if is_velocity else ''}_r_hip",
                ]
            )

            self._overide_plot(
                axs,
                data_source[:, 0],
                data_source[:, 1],
                use_cycler=options["cycles"],
                label=f"Prosthetic {options_type} True",
            )
            self._overide_plot(
                axs,
                data_source[:, 0],
                self.output_data[:, offset],
                use_cycler=options["cycles"],
                label=f"Prosthetic {options_type} True CLME",
            )
            self._overide_plot(
                axs,
                data_source[:, 0],
                data_source[:, 2],
                use_cycler=options["cycles"],
                label=f"Intact {options_type} True CLME",
                linestyle="--",
            )

            axs.set_xlabel("Time (s)")

            if not is_velocity:
                axs.set_ylabel("Position (deg)")
            else:
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
            Type of the plot. Can be "position" or "velocity". Default is "position"
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

        axs = fig.subplots(2, 1)

        self.plot(axs[0], options={"type": "position", **options})
        self.plot(axs[1], options={"type": "velocity", **options})

        return fig, axs


if __name__ == "__main__":
    from data_manager import DataManager
    from dataset_manager import ds

    # Example usage
    dm = DataManager()
    dm.load_data("P01", "exp_transition_03-S3_01", ds.DEVILLEZ)
    clme = CLME(dm)

    clme.compute()

    fig, axs = clme.complete_plot(None, options={"cycles": False})

    show()
