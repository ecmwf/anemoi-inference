# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path

import numpy as np
from anemoi.utils.grib import shortname_to_paramid
from anemoi.utils.grib import units

from anemoi.inference.context import Context
from anemoi.inference.decorators import ensure_dir
from anemoi.inference.decorators import main_argument
from anemoi.inference.types import FloatArray
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State
from anemoi.inference.utils.templating import render_template

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
@main_argument("dir")
@ensure_dir("dir")
class PlotOutput(Output):
    """Use `earthkit-plots` to plot the outputs."""

    def __init__(
        self,
        context: Context,
        dir: Path,
        *,
        variables: list[str] | None = None,
        mode: str = "subplots",
        domain: str | list[str] | None = None,
        schema: str | None = None,
        template: str = "plot_{date}.{format}",
        format: str = "png",
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        **kwargs,
    ) -> None:
        """Initialise the PlotOutput.

        Parameters
        ----------
        context : Context
            The context.
        dir : Path
            The directory to save the plots.
            If the directory does not exist, it will be created.
        variables : list, optional
            The list of variables to plot, by default all.
        mode : str, optional
            The plotting mode, can be "subplots" or "overlay", by default "subplots".
        domain : str | list[str] | None, optional
            The domain/s to plot, by default None.
        schema : str | None, optional
            The schema to use, by default None.
        template : str, optional
            The template for plot filenames, by default "plot_{date}.{format}".
            Has access to the following variables:
            - date: the date of the forecast step
            - basetime: the base time of the forecast
            - domain: the domain being plotted
            - format: the format of the plot
            - variables: the variables being plotted (joined by underscores)
        format : str, optional
            The format of the plot, by default "png".
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        **kwargs : Any
            Additional keyword arguments to pass to `earthkit.plots.quickplot`.
        """

        super().__init__(
            context,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )

        self.dir = dir
        self.format = format
        self.variables = variables
        self.template = template
        self.domain = domain
        self.mode = mode
        self.schema = schema
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

        if self.schema:
            ekp.schema.use(self.schema)

        longitudes = fix(state["longitudes"])
        latitudes = state["latitudes"]
        date = state["date"]
        basetime = date - state["step"]

        plotting_fields = []

        for name, values in state["fields"].items():
            if self.skip_variable(name):
                continue

            variable = self.typed_variables[name]
            param = variable.param

            plotting_fields.append(
                ekd.ArrayField(
                    values,
                    {
                        "param": param,
                        "paramId": shortname_to_paramid(param),
                        "shortName": param,
                        "variable_name": param,
                        "step": state["step"],
                        "base_datetime": basetime,
                        "valid_time": date,
                        "latitudes": latitudes,
                        "longitudes": longitudes,
                        "units": units(param),
                    },
                )
            )
        fig = ekp.quickplot(
            ekd.FieldList.from_fields(plotting_fields), mode=self.mode, domain=self.domain, **self.kwargs
        )
        fname = render_template(
            self.template,
            {
                "date": date,
                "basetime": basetime,
                "domain": self.domain,
                "format": self.format,
                "variables": "_".join(self.variables or []),
            },
        )
        fname = self.dir / fname

        fig.save(fname)
        del fig
