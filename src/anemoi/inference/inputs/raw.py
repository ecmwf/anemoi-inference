# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Raw input.

Reads the ``.npz`` files produced by :class:`anemoi.inference.outputs.raw.RawOutput`.

This is the counterpart of the ``raw`` output and makes it possible to
*stack* two models: the raw output of a first model can be fed as the raw
input (initial conditions) of a second model.

Each file written by ``RawOutput`` contains a single date and holds:

- ``field_<name>``: the values for variable ``<name>`` (shape ``(n_points,)``)
- ``date``: the valid date of the fields (ISO-ish string)
- ``latitudes`` / ``longitudes``: the grid coordinates

Given the list of dates requested by the runner (typically the ``lagged``
dates of the model, e.g. ``t-6h`` and ``t=0``), this input loads the matching
files and stacks the fields along the date dimension to build the input state.
"""

import datetime
import logging
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from anemoi.transform.variables import Variable
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State
from anemoi.inference.utils.templating import render_template

from ..decorators import ensure_dir
from ..decorators import format_dataset_name
from ..decorators import main_argument
from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)

FIELD_PREFIX = "field_"


@input_registry.register("raw")
@main_argument("dir")
@format_dataset_name("dir")
@ensure_dir("dir", create=False, must_exist=True, unique=False)
class RawInput(Input):
    """Reads the ``.npz`` files produced by :class:`RawOutput`.

    The naming convention (``template`` and ``strftime``) must match the one
    used by the ``raw`` output that produced the files.
    """

    trace_name = "raw"

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        dir: Path,
        template: str = "{date}.npz",
        strftime: str = "%Y%m%d%H%M%S",
        **kwargs: Any,
    ) -> None:
        """Initialise the RawInput.

        Parameters
        ----------
        context : Context
            The context in which the input is used.
        metadata : Metadata
            Metadata corresponding to the dataset this input is handling.
        dir : Path
            The directory containing the raw ``.npz`` files.
        template : str, optional
            The template for filenames, by default ``"{date}.npz"``.
            Must match the template used by the ``raw`` output that wrote
            the files. Variables available are ``date``, ``basetime`` and
            ``step``.
        strftime : str, optional
            The date format string, by default ``"%Y%m%d%H%M%S"``.
            Must match the one used by the ``raw`` output.
        **kwargs : Any
            Additional keyword arguments passed to :class:`Input`.
        """
        super().__init__(context, metadata, **kwargs)
        self.dir = Path(dir)
        self.template = template
        self.strftime = strftime

    def __repr__(self) -> str:
        """Return a string representation of the RawInput object."""
        return f"RawInput({self.dir})"

    def _filename(self, date: datetime.datetime, base_date: datetime.datetime | None = None) -> str:
        """Render the file name for a given date.

        Parameters
        ----------
        date : datetime.datetime
            The valid date of the fields.
        base_date : datetime.datetime, optional
            The base (reference) date used to compute the step. If None, the
            reference date of the context is used.

        Returns
        -------
        str
            The rendered file name (without directory).
        """
        base_date = base_date if base_date is not None else to_datetime(self.reference_date)
        step = date - base_date

        format_info = {
            "date": date.strftime(self.strftime),
            "basetime": base_date.strftime(self.strftime),
            "step": step,
        }
        return render_template(self.template, format_info)

    def _load_file(self, date: datetime.datetime, base_date: datetime.datetime | None = None) -> dict[str, Any]:
        """Load a single ``.npz`` file for the given date.

        Parameters
        ----------
        date : datetime.datetime
            The valid date of the fields to load.
        base_date : datetime.datetime, optional
            The base date used to compute the step in the file name template.

        Returns
        -------
        dict[str, Any]
            A dictionary with keys ``fields`` (mapping variable name to a 1D
            array), ``latitudes``, ``longitudes`` and ``date``.
        """
        path = self.dir / self._filename(date, base_date=base_date)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__}: no raw file found for date {date.isoformat()} at {path}. "
                "Check that `template` and `strftime` match the `raw` output that produced the files."
            )

        LOG.info("%s: loading %s", self.__class__.__name__, path)
        with np.load(path, allow_pickle=False) as data:
            fields = {
                key[len(FIELD_PREFIX) :]: np.asarray(data[key]) for key in data.files if key.startswith(FIELD_PREFIX)
            }
            latitudes = np.asarray(data["latitudes"]) if "latitudes" in data.files else None
            longitudes = np.asarray(data["longitudes"]) if "longitudes" in data.files else None
            file_date = str(data["date"]) if "date" in data.files else None

        return dict(fields=fields, latitudes=latitudes, longitudes=longitudes, date=file_date)

    def _build_state(self, dates: list[Date], *, base_date: datetime.datetime | None = None) -> State:
        """Build a state by stacking the fields of the requested dates.

        Parameters
        ----------
        dates : list of Date
            The dates for which to build the state.
        base_date : datetime.datetime, optional
            The base date used to compute the step in the file name template.

        Returns
        -------
        State
            The state with ``fields`` as ``dict[str, np.ndarray]`` of shape
            ``(len(dates), n_points)``.
        """
        if not dates:
            raise ValueError(f"{self.__class__.__name__}: no dates provided")

        dates = sorted(to_datetime(d) for d in dates)

        loaded = [self._load_file(date, base_date=base_date) for date in dates]

        latitudes = loaded[0]["latitudes"]
        longitudes = loaded[0]["longitudes"]

        requested = set(self.variables)

        # Determine the set of variables to expose (intersection of requested
        # variables and what is available in the files).
        available = set(loaded[0]["fields"].keys())
        missing = requested - available
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: variables {sorted(missing)} not found in raw files. "
                f"Available variables: {sorted(available)}"
            )

        typed_variables = self.metadata.typed_variables

        fields: dict[str, FloatArray] = {}
        state_variables: dict[str, Variable] = {}

        for name in self.variables:
            stacked = np.stack([entry["fields"][name] for entry in loaded], axis=0)
            fields[name] = stacked
            if name in typed_variables:
                state_variables[name] = typed_variables[name]

            if trace := self.context.tensor_handlers[self.dataset_name].trace:
                trace.from_input(name, self)

        state: State = dict(
            date=dates[-1],
            latitudes=latitudes,
            longitudes=longitudes,
            fields=fields,
            _input=self,
            _variables=state_variables,
        )

        return state

    def create_input_state(self, *, dates: list[Date], **kwargs: Any) -> State:
        """Create the input state for the given dates.

        Parameters
        ----------
        dates : list of Date
            The dates for which to create the input state (as computed by the
            runner from the model's ``lagged`` steps).
        **kwargs : Any
            Additional keyword arguments (ignored).

        Returns
        -------
        State
            The created input state.
        """
        return self._build_state(dates)

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        """Load the forcings state for the given dates.

        Parameters
        ----------
        dates : list of Date
            The dates for which to load the forcings.
        current_state : State
            The current state of the model.

        Returns
        -------
        State
            The loaded forcings state.
        """
        base_date = None
        if current_state.get("date") is not None:
            base_date = to_datetime(current_state["date"]) - current_state.get("step", datetime.timedelta())

        state = self._build_state(dates, base_date=base_date)
        state["dates"] = sorted(to_datetime(d) for d in dates)
        return state

    @cached_property
    def latitudes(self) -> FloatArray | None:
        """Return the latitudes of the raw files, if available."""
        return self._reference_coords[0]

    @cached_property
    def longitudes(self) -> FloatArray | None:
        """Return the longitudes of the raw files, if available."""
        return self._reference_coords[1]

    @cached_property
    def _reference_coords(self) -> tuple[FloatArray | None, FloatArray | None]:
        """Return the grid coordinates from the first available raw file."""
        files = sorted(self.dir.glob("*.npz"))
        if not files:
            return None, None
        with np.load(files[0], allow_pickle=False) as data:
            latitudes = np.asarray(data["latitudes"]) if "latitudes" in data.files else None
            longitudes = np.asarray(data["longitudes"]) if "longitudes" in data.files else None
        return latitudes, longitudes
