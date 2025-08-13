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
from anemoi.utils.grib import units

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
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
@main_argument("path")
class PlotOutput(Output):
    """Use `earthkit-plots` to plot the outputs."""

    def __init__(
        self,
        context: Context,
        path: str,
        *,
        variables: list[str] | None = None,
        mode: str = "subplots",
        domain: str | list[str] | None = None,
        strftime: str = "%Y%m%d%H%M%S",
        template: str = "plot_{date}.{format}",
        format: str = "png",
        missing_value: float | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        **kwargs,
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
        mode : str, optional
            The plotting mode, can be "subplots" or "overlay", by default "subplots".
        domain : str | list[str] | None, optional
            The domain/s to plot, by default None.
        strftime : str, optional
            The date format string, by default "%Y%m%d%H%M%S".
        template : str, optional
            The template for plot filenames, by default "plot_{date}.{format}".
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
        self.missing_value = missing_value
        self.domain = domain
        self.mode = mode
        self.kwargs = kwargs

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        import earthkit.data as ekd
        import earthkit.plots as ekp

        os.makedirs(self.path, exist_ok=True)

        longitudes = fix(state["longitudes"])
        latitudes = state["latitudes"]
        date = state["date"]
        basetime = date - state["step"]

        plotting_fields = []

        for name, values in state["fields"].items():
            if self.skip_variable(name):
                continue

            variable = self.context.checkpoint.typed_variables[name]
            param = variable.param

            plotting_fields.append(
                ekd.ArrayField(
                    values,
                    {
                        "shortName": param,
                        "variable_name": param,
                        "step": state["step"],
                        "base_datetime": basetime,
                        "latitudes": latitudes,
                        "longitudes": longitudes,
                        "units": units(param),
                    },
                )
            )

        fig = ekp.quickplot(
            ekd.FieldList.from_fields((plotting_fields)), mode=self.mode, domain=self.domain, **self.kwargs
        )
        fname = self.template.format(date=date, format=self.format)
        fname = os.path.join(self.path, fname)

        fig.save(fname)
        del fig
