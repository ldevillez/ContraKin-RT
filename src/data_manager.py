"""
Wrapper to manage data from datasets

Description
-----------
This module define
- the DataManager class for:
  - Loading data
  - Getting data
  - Plotting data
- the DataManagersIterator class
  - for iterating over multiple datasets/subjects/trials
  - with whitelist/blacklist functionality
- A function to get a populated DataManagersIterator from a YML file
- A function to compare two DataManager instances

Example
-------
> dmi = get_populated_dmi()
> for ds_name, subject, trial in dmi:
>     print(f"ds: {ds_name}, Subject: {subject}, Trial: {trial}")
>
> dm = DataManager()
> dm.load_data("P01", "exp_transition_03-S3_01", DS_TYPE.DEVILLEZ)
> data = dm.get_data(["time", "q_r_hip", "qd_r_hip"])
> dm.plot(options={"type": "position", "leg": "leading", "art": "hip"})
"""

import sys
from pathlib import Path

from numpy import array, zeros, where, savetxt


from matplotlib.pyplot import Axes, axes, Figure, figure, show

from tasks import TaskManager

from support import swap_words, read_yaml
from filter import ExponentialFilter

import colors

import dataset_manager as ds


DS_TYPE = ds.ds
# Path to the white/blacklist YML file
PATH_TRIAL_LIST = "ds_user_trial_lists.yml"


class DataManager:
    """Class representing a data manager

    Attributes
    ----------
    ds_name : ds.ds
        Dataset name
    subject : str
        Subject name
    trial : str
        Trial name
    time : array
        Time array
    _angle : array
        Angular position array
    _velocity : array
        Angular velocity array
    cols : list
        List of columns name
    idx_of_cols : dict
        Dictionary of index of columns
    leading : str
        Leading leg
    reverse : bool
        Reverse the data to match the leading leg

    """

    ds_name: ds.ds
    subject: str
    trial: str

    time: array
    _angle: array
    _velocity: array

    cols: list
    idx_of_cols: dict

    leading: str
    reverse: bool

    task: TaskManager
    color_cycler: dict

    def __init__(self) -> None:
        """
        Initialize a data manager.
        """

        self.ds_name = ds.ds.NONE
        self.subject = ""
        self.trial = ""

        self.time = array([])
        self._angle = array([])
        self._velocity = array([])

        self.cols = []
        self.idx_of_cols = {}

        self.color_cycler = colors.get_color_cycler()

    def load_data(self, subject, trial, ds_name, full_trial=False) -> None:
        """ Load the data from the ds into the data manager

        Parameters
        ----------
        subject : str
            Subject to load
        trial : str
            Trial to load
        ds_name : ds
            Dataset to load
        full_trial : bool
            If the trial is "full"

        Returns
        -------
        bool
            True if the data is loaded, False if the data is empty
        """

        self.subject = subject
        self.trial = trial
        self.ds_name = ds_name

        self.time, self._angle, self._velocity, self.cols, self.idx_of_cols, leading = ds.load(subject, trial, ds_name, full_trial)

        if len(self.time) == 0:
            self.reverse = False
            return False



        self.leading = "Right"
        self.reverse = False

        if leading is not None and leading == "Left":
            self.leading = "Left"
            self.reverse = True

        return True


    def get_data(self, cols: list) -> array:
        """ Fetch the data from data manager

        Parameters
        ----------
        cols : list
            List of columns to fetch

        Returns
        -------
        array
            Data fetched from the data manager
        """

        data_out = zeros((len(self.time), len(cols)))

        for i, col in enumerate(cols):
            if "time" in col:
                data_out[:, i] = self.time
                continue

            if self.reverse:
                col = swap_words(col, "_r_", "_l_")

            col_converted = ds.convert_col(col, self.ds_name)

            if "qd" in col:
                data_source = self._velocity
            else:
                data_source = self._angle


            data_out[:, i] = data_source[:, self.idx_of_cols[col_converted]]

        return data_out

    def get_dt(self) -> float:
        """ Get the time step

        Returns
        -------
        float
            Time step
        """

        return self.time[1] - self.time[0]


    def get_dts(self) -> array:
        """ Get the time steps

        Returns
        -------
        array
            Time steps
        """

        return self.time[1:] - self.time[:-1]

    def _overide_plot(self, ax, x, y, **args):
        """
        Override the plot function to customize the plot.
        """

        if "color" not in args:
            style = self.color_cycler[args["label"] + self.ds_name.value + self.subject + self.trial]
            args.update(style)

        use_cycler = False
        if "use_cycler" in args:
            use_cycler = args["use_cycler"]
            del args["use_cycler"]

        if not hasattr(self, "cycler"):
            use_cycler = False


        if use_cycler:
            for idx_a, idx_b in self.cycler.cycle_idxs_generator():
                t = x[idx_a:idx_b]
                t = t - t[0]  # Normalize time to start at 0
                t = 100 * t / t[-1]  # Normalize time to end at 1
                ax.plot(t, y[idx_a:idx_b], **args)
        else:
            ax.plot(x, y, **args)

        return ax

    def export_data(
            self,
            hip=True,
            knee=False,
            position = True,
            velocity = False,
            leading = True,
            following = False,
            reduce_factor = 1
            ) -> None:
        """
        Get the data to export

        Parameters
        ----------
        hip : bool
            If True, include hip data
        knee : bool
            If True, include knee data

        position : bool
            If True, include position data
        velocity : bool
            If True, include velocity data

        leading : bool
            If True, include leading leg data
        following : bool
            If True, include following leg data

        Returns
        -------
        tuple (ndarray, list)
            Data and columns names

        """

        cols = ["time"]

        for art in ["hip", "knee"]:
            if (art == "hip" and not hip) or (art == "knee" and not knee):
                continue

            for typ in ["q", "qd"]:
                if (typ == "q" and not position) or (typ == "qd" and not velocity):
                    continue

                for leg in ["l", "r"]:
                    if (leg == "l" and not following) or (leg == "r" and not leading):
                        continue

                    cols.append(f"{typ}_{leg}_{art}")

        data = self.get_data(cols)
        data = data[::reduce_factor, :]

        savetxt(f"data/dm_{self.subject}_{self.trial}.txt", data, header=" ".join(cols), comments="")



    def plot(self, axs: None | Axes = None, options: dict = {}) -> Axes:
        """ Plot the output data
        Parameters
        ----------
        axs : None | Axes
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
            is_leading = True

        ## Articulation, hip or knee
        if "art" in options:
            options_art = options["art"]
        else:
            options_art = "hip"

        is_hip = True
        if options_art == "knee":
            is_hip = False

        ## limited display
        if "duration" in options:
            options_duration = options["duration"]
        else:
            options_duration = 1e6

        use_cycles = False
        if "cycles" in options:
            use_cycles = options["cycles"]




        # Display position/velocity for the estimator or the true leg
        if options_type in ["position", "velocity"]:

            art = 'hip' if is_hip else 'knee'
            data_source = self.get_data([
                "time",
                f"q{'d' if is_velocity else ''}_{'r' if is_leading else 'l'}_{art}"
                ])

            data = data_source

            filt = ExponentialFilter(0.8)
            data[:, 1] = filt.apply(data_source[:, 1])

            idx_to_plot = where(data[:,0] - data[0, 0] < options_duration)[0]

            self._overide_plot(
                    axs,
                    data[idx_to_plot,0], data[idx_to_plot,1],
                    use_cycler=use_cycles,
                    label=f"{'Leading' if is_leading else 'Following'} {art} ",
                    )
            axs.set_xlabel("Time (s)")

            if not is_velocity:
                axs.set_ylabel("Position (deg)")
            else:
                axs.set_ylabel("Velocity (deg/s)")

        if options_type in ["task_mask"]:
            color_cycler = colors.get_color_cycler()


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
            Type of data to plot. Can be "position" or "velocity"
        leg : str
            Leg to plot. Can be "leading" or "following"
        art : str
            Articulation to plot. Can be "hip" or "knee"
        """

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

        if "type" not in options:
            options["type"] = "Side"

        is_side_separated = False
        if "Art" != options["type"]:
            is_side_separated = True

        use_cycles = False
        if "cycles" in options:
            use_cycles = options["cycles"]

        self.color_cycler = colors.get_color_cycler()


        axs = fig.subplots(2, 2)

        self.task = TaskManager(self)

        if not use_cycles:
            for axs_x in axs:
                for axs_y in axs_x:
                    self.plot(axs_y, options={
                        "type": "task_mask",
                        })
        type_items = ["position", "velocity"]
        leg_items = ["leading", "following"]
        art_items = ["hip", "knee"]

        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                # axs[i][j].clear()
                self.task.plot(axs[i][j], self.time)

        for i, type_item in enumerate(type_items):
            for j, leg_item in enumerate(leg_items):
                for k, art_item in enumerate(art_items):

                    idx_col = k
                    if is_side_separated:
                        idx_col = j

                    # Change j based on options
                    self.plot(axs[i][idx_col], options={
                        "type": type_item,
                        "leg": leg_item,
                        "art": art_item,
                        "cycles": use_cycles
                    })

                    if type_item == "velocity":
                        # Change j based on options
                        self.plot(axs[i][idx_col], options={
                            "type": type_item,
                            "leg": leg_item,
                            "art": art_item,
                            "derivative": True,
                            "cycles": use_cycles
                        })


        labels_tsk = []
        handels_tsk = []
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):

                handels_raw, labels_raw = axs[i][j].get_legend_handles_labels()

                labels = []
                handels = []

                for label, handel in zip(labels_raw, handels_raw):
                    if label not in labels and "tsk " not in label:
                        labels.append(label)
                        handels.append(handel)
                    if "tsk " in label:
                        label_shortened = label.split("tsk ")[-1]
                        if label_shortened not in labels_tsk:
                            labels_tsk.append(label_shortened)
                            handels_tsk.append(handel)

                axs[i][j].legend(handels, labels, loc="upper right")



                if i != 0 or j != 0:
                    axs[i][j].sharex(axs[0][0])

                if j != 0:
                    axs[i][j].sharey(axs[i][0])

        fig.suptitle(f"{self.subject} - {self.trial}")


        fig.legend(handels_tsk, labels_tsk, loc="lower center", ncol=len(labels_tsk))


        return fig, axs



class DataManagersIterator():
    """
    Iterator to manage the ds/subject/trials structrue
    """

    whitelist: dict
    blacklist: dict
    internal_len: int

    def __init__(self, whitelist: dict = {}, blacklist: dict = {}) -> None:
        """
        Initialize the iterator

        Parameters
        ----------
        ds_name : ds.ds
            Dataset name
        subject : str
            Subject name
        trials : list[str]
            List of trials to iterate over
        """

        self.whitelist = whitelist
        self.blacklist = blacklist

        self.internal_len = -1


    def _parselist(self, raw_data, level: int = 0) -> dict:
        """ Parse a list of trials to a dictionary

        Parameters
        ----------
        raw_data : list or dict or str or None
            List of trials to parse
        level : int
            Level of the list

        Returns
        -------
        yield tuple (ds.ds, str, str)
            ds name, subject name, trial name
        """

        if level == 0:
            if raw_data is None:
                return []
            if isinstance(raw_data, str):
                return [ raw_data ]
            if isinstance(raw_data, list):
                for item in raw_data:
                    if not isinstance(item, str):
                        raise ValueError(f"Final level of the list should be str")
                return raw_data
            if isinstance(raw_data, dict) and len(raw_data) == 0:
                return []

            raise ValueError(f"Invalid type for raw_data: {type(raw_data)}")

        if raw_data is None:
            return {}

        if isinstance(raw_data, str):
            return {raw_data: self._parselist(None, level - 1)}

        out_data = {}
        if isinstance(raw_data, list):
            for item in raw_data:
                out_data[item] = self._parselist(None, level - 1)
        elif isinstance(raw_data, dict):
            for key, value in raw_data.items():
                out_data[key] = self._parselist(value, level - 1)
        else:
            raise ValueError(f"Invalid item in list: {raw_data}")

        return out_data

    def load_list(self, filename) -> None:
        """ Load the list of trials from a file

        Parameters
        ----------
        filename : str
            Name of the file to load
            YML format
        """

        data = read_yaml(Path(filename))

        if "whitelist" not in data:
            self.whitelist = {}
        else:
            self.whitelist = self._parselist(data["whitelist"], 3)

        if "blacklist" not in data:
            self.blacklist = {}
        else:
            self.blacklist = self._parselist(data["blacklist"], 3)


    def __iter__(self):
        """
        Iterate
        """

        for ds_name in ds.list_ds():

            # If the whitelist is not empty and the ds is not in it
            if ds_name.value not in self.whitelist and len(self.whitelist) > 0:
                continue


            # If the blacklist is not empty and the ds is in it
            if ds_name.value in self.blacklist and len(self.blacklist[ds_name.value]) == 0:
                continue

            for user_name in ds.list_subject(ds_name):
                # If the whitelist is not empty and the user is not in it
                if ds_name.value in self.whitelist and user_name not in self.whitelist[ds_name.value] and len(self.whitelist[ds_name.value]) > 0:
                    continue

                # If the blacklist is not empty and the user is in it
                if ds_name.value in self.blacklist and user_name in self.blacklist[ds_name.value] and len(self.blacklist[ds_name.value][user_name]) == 0:
                    continue


                for trial_name in ds.list_trial(user_name, ds_name):

                    # If the whitelist is not empty and the trial is not in it
                    should_keep = False
                    if ds_name.value in self.whitelist and user_name in self.whitelist[ds_name.value] and trial_name not in self.whitelist[ds_name.value][user_name] and len(self.whitelist[ds_name.value][user_name]) > 0:

                        # We look for all the trial in the whitelist to see if they match the prefix
                        for prefix_name in self.whitelist[ds_name.value][user_name]:
                            if prefix_name in trial_name:

                                # If it match and the list is empty we keep it
                                if len (self.whitelist[ds_name.value][user_name][prefix_name]) == 0:
                                    should_keep = True
                                    break

                                # If it match and the list is not empty we look for the suffix
                                for suffix_name in self.whitelist[ds_name.value][user_name][prefix_name]:
                                    total_name = prefix_name + suffix_name
                                    # if the total name match we keep it
                                    if total_name == trial_name:
                                        should_keep = True
                                        break

                            if should_keep:
                                break
                        if not should_keep:
                            continue

                    # If the blacklist is not empty and the trial is in it
                    if ds_name.value in self.blacklist and user_name in self.blacklist[ds_name.value]:
                        if trial_name in self.blacklist[ds_name.value][user_name]:
                            continue

                        # We look for all the trial in the whitelist to see if they match the prefix
                        should_keep = True
                        for prefix_name in self.blacklist[ds_name.value][user_name]:
                            if prefix_name in trial_name:

                                # If it match and the list is empty we do not keep it
                                if len (self.whitelist[ds_name.value][user_name][prefix_name]) == 0:
                                    should_keep = False
                                    break

                                # If it match and the list is not empty we look for the suffix
                                for suffix_name in self.whitelist[ds_name.value][user_name][prefix_name]:
                                    total_name = prefix_name + suffix_name
                                    # if the total name match we do not keep it
                                    if total_name == trial_name:
                                        should_keep = False
                                        break
                            if not should_keep:
                                break
                        if not should_keep:
                            continue




                    yield ds_name, user_name, trial_name

    def get_number_of_elements(self) -> int:
        """ Get the number of elements in the iterator

        Returns
        -------
        int
            Number of elements in the iterator
        """

        count = 0
        for _ in self:
            count += 1

        self.internal_len = count

        return count

    def __len__(self) -> int:
        """ Get the length of the iterator

        Returns
        -------
        int
            Length of the iterator
        """

        if self.internal_len == -1:
            self.get_number_of_elements()

        return self.internal_len

def get_populated_dmi() -> DataManagersIterator:
    """ Get a populated DataManagersIterator

    Returns
    -------
    DataManagersIterator
        Populated DataManagersIterator
    """

    dmi = DataManagersIterator()
    dmi.load_list(PATH_TRIAL_LIST)

    return dmi


def compare_datamanagers(dm1, dm2):
    """
    Compare the cycles of two data managers

    Parameters
    ----------
    dm1 : DataManager
        First data manager
    dm2 : DataManager
        Second data manager

    Returns
    -------
    fig : Figure
        Figure of the comparison
    axs : list[Axes]
        List of axes of the comparison
    """

    dm1.color_cycler = dm2.color_cycler

    fig = plt.figure()
    axs = fig.subplots(2,2)
    dm1.plot(axs[0, 0], options={
        "leg": "leading",
        "art": "hip",
        "cycles": True,
    })
    dm2.plot(axs[0,0], options={
        "leg": "leading",
        "art": "hip",
        "cycles": True,
    })

    dm1.plot(axs[0, 1], options={
        "leg": "leading",
        "art": "knee",
        "cycles": True,
    })
    dm2.plot(axs[0,1], options={
        "leg": "leading",
        "art": "knee",
        "cycles": True,
    })

    dm1.plot(axs[1, 0], options={
        "leg": "leading",
        "type": "velocity",
        "art": "hip",
        "cycles": True,
    })
    dm2.plot(axs[1,0], options={
        "leg": "leading",
        "type": "velocity",
        "art": "hip",
        "cycles": True,
    })

    dm1.plot(axs[1, 1], options={
        "leg": "leading",
        "type": "velocity",
        "art": "knee",
        "cycles": True,
    })
    dm2.plot(axs[1,1], options={
        "leg": "leading",
        "type": "velocity",
        "art": "knee",
        "cycles": True,
    })

    return fig, axs

if __name__ == "__main__":

    # Iterate over the data managers iterator using the dmi with rules from PATH_TRIAL_LIST applied
    dmi = get_populated_dmi()
    for ds_name, subject, trial in dmi:
        print(f"ds: {ds_name}, Subject: {subject}, Trial: {trial}")

    dm = DataManager()
    dm.load_data("P05", "exp_transition_03-S3_08", DS_TYPE.DEVILLEZ)
    data = dm.get_data(["time", "q_r_hip", "qd_r_hip"])
    dm.complete_plot()
    show()
