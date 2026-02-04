"""
Module to generate and handle colors cycler and colormaps for visualizing data.

Description
-----------
This module provides functions to lighten colors and classes to manage colormaps and color cyclers.
It includes:
- `lighten_color`: Function to lighten a given color.
- `ColormapCycler`: Class to create and manage custom colormaps.
- `get_color_cycler`: Function to generate a color cycler with optional linestyles

Example
-------
> lighten_color('g', 0.3) # Lightens green color
> colormap_cycler = ColormapCycler() # Creates a colormap cycler
> cmap = colormap_cycler.get_colormap("example") # Retrieves or creates a colormap
> color_cycler = get_color_cycler(linestyle=True) # Generates a color cycler with linestyles

"""


import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.pyplot import rcParams
import matplotlib.colors as mc

import colorsys
from cycler import cycler
from collections import defaultdict

DS_STYLE = {
    "color": "#2D4E9C",
}

GRAY_STYLE = {
    "color": "gray",
}

LINESTYLE_CYCLE = ["solid", "dashed", "dashdot", "dotted", ":"]

def lighten_color(color: str | tuple[float, float, float], amount:float =0.5) -> tuple[float, float, float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    parameters
    ----------
    color : str | float
        Matplotlib color string, hex string, or RGB tuple.
    amount : float
        Amount to lighten the color. 0 returns black, 1 returns the original color.

    returns
    -------
    tuple(float, float, float)
        Lightened RGB color.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color

    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


class ColormapCycler:
    """
    Class to manage and cycle through a set of colormaps for plotting.

    Attributes
    ----------
    colormaps : dict
        A dictionary to store custom colormaps.
    """

    colormaps: dict
    current_color_cycle: any

    def __init__(self):
        self.colormaps = {}

        # Get the current color cycle from matplotlib
        self.current_color_cycle = (plt.rcParams["axes.prop_cycle"])()

    def get_colormap(self, name: str) -> clr.LinearSegmentedColormap:
        """
        Returns a custom colormap based on the provided name.
        If the colormap does not exist, it creates a new one with a background color
        and a color from the current color cycle.

        parameters
        ----------
        name : str
            The name of the colormap to retrieve or create.

        returns
        -------
        clr.LinearSegmentedColormap
        """

        if name not in self.colormaps:
            color_background = "#FFFFFF"
            if name == "total":
                color_cycler = DS_STYLE["color"]
            else:
                color_cycler = next(self.current_color_cycle)["color"]

            cmap_custom = clr.LinearSegmentedColormap.from_list(
                    name,
                    [
                        color_background,
                        color_cycler
                    ])
            self.colormaps[name] = cmap_custom


        return self.colormaps[name]

def get_color_cycler(linestyle: bool = False) -> defaultdict:
    """
    Generate a color cycler.

    parameters
    ----------
    linestyle : bool
        If True, include linestyles in the cycler.

    returns
    -------
    defaultdict

    """
    # For the cycle
    current_color_cycle = rcParams["axes.prop_cycle"]

    if linestyle:
        # If linestyle is True, add linestyle to the cycle
        linestyle_cycler = cycler("linestyle", LINESTYLE_CYCLE)

        current_color_cycle =  linestyle_cycler * current_color_cycle

    current_color_cycle = current_color_cycle()

    dd_loop = defaultdict(lambda: next(current_color_cycle))

    return dd_loop


if __name__ == "__main__":
    color_cycler = get_color_cycler(linestyle=True)
    a = color_cycler["test"]
    print(color_cycler)


    cmap = ColormapCycler().get_colormap("example")
    print(cmap)
