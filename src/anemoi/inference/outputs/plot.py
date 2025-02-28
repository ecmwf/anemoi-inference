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

from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


def fix(lons):
    return np.where(lons > 180, lons - 360, lons)


@output_registry.register("plot")
class PlotOutput(Output):
    """_summary_"""

    def __init__(
        self,
        context,
        path,
        variables=all,
        strftime="%Y%m%d%H%M%S",
        template="plot_{variable}_{date}.{format}",
        dpi=300,
        format="png",
        missing_value=None,
        output_frequency=None,
        write_initial_state=None,
    ):
        super().__init__(context, output_frequency=output_frequency, write_initial_state=write_initial_state)
        self.path = path
        self.format = format
        self.variables = variables
        self.strftime = strftime
        self.template = template
        self.dpi = dpi
        self.missing_value = missing_value

        if self.variables is not all:
            if not isinstance(self.variables, (list, tuple)):
                self.variables = [self.variables]

    def write_step(self, state):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        os.makedirs(self.path, exist_ok=True)

        longitudes = state["longitudes"]
        latitudes = state["latitudes"]
        triangulation = tri.Triangulation(fix(longitudes), latitudes)

        for name, values in state["fields"].items():

            if self.variables is not all and name not in self.variables:
                continue

            _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")

            missing_values = np.isnan(values)
            missing_value = self.missing_value
            if missing_value is None:
                min = np.nanmin(values)
                missing_value = min - np.abs(min) * 0.001

            values = np.where(missing_values, self.missing_value, values)

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
