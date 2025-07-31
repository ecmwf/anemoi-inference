# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import FloatArray
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


def fix(lons: FloatArray) -> FloatArray:
    """Fix longitudes greater than 180 degrees.

    Parameters
    ----------
    lons : FloatArray
        Array of longitudes.

    Returns
    -------
    FloatArray
        Fixed array of longitudes.
    """
    return np.where(lons > 180, lons - 360, lons)


@output_registry.register("plot")
class PlotOutput(Output):
    """Plot output class."""

    def __init__(
        self,
        context: Context,
        path: str,
        strftime: str = "%Y%m%d%H%M%S",
        template: str = "plot_{variable}_{date}.{format}",
        dpi: int = 300,
        format: str = "png",
        variables: list[str] | None = None,
        missing_value: float | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ) -> None:
        """Initialize the PlotOutput.

        Parameters
        ----------
        context : Context
            The context.
        path : str
            The path to save the plots.
        variables : list, optional
            The list of variables to plot, by default all.
        strftime : str, optional
            The date format string, by default "%Y%m%d%H%M%S".
        template : str, optional
            The template for plot filenames, by default "plot_{variable}_{date}.{format}".
        dpi : int, optional
            The resolution of the plot, by default 300.
        format : str, optional
            The format of the plot, by default "png".
        missing_value : float, optional
            The value to use for missing data, by default None.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        """

        super().__init__(
            context,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
        self.path = path
        self.format = format
        self.variables = variables
        self.strftime = strftime
        self.template = template
        self.dpi = dpi
        self.missing_value = missing_value

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        os.makedirs(self.path, exist_ok=True)

        longitudes = state["longitudes"]
        latitudes = state["latitudes"]
        triangulation = tri.Triangulation(fix(longitudes), latitudes)

        for name, values in state["fields"].items():
            if self.skip_variable(name):
                continue

            _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")

            missing_values = np.isnan(values)
            missing_value = self.missing_value
            if missing_value is None:
                min = np.nanmin(values)
                missing_value = min - np.abs(min) * 0.001

            values = np.where(missing_values, self.missing_value, values).astype(np.float32)

            _ = ax.tricontourf(triangulation, values, levels=10, transform=ccrs.PlateCarree())

            ax.tricontour(
                triangulation,
                values,
                levels=10,
                colors="black",
                linewidths=0.5,
                transform=ccrs.PlateCarree(),
            )

            date = state["date"].strftime("%Y-%m-%d %H:%M:%S")
            ax.set_title(f"{name} at {date}")

            date = state["date"].strftime(self.strftime)
            fname = self.template.format(date=date, variable=name, format=self.format)
            fname = os.path.join(self.path, fname)

            plt.savefig(fname, dpi=self.dpi, bbox_inches="tight")
            plt.close()
