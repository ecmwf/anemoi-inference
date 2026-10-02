# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Raw input.

Reads the ``.npz`` files produced by :class:`anemoi.inference.outputs.raw.RawOutput`,
so that the output of a first model can be fed as the initial conditions of a second.

A ``manifest.json`` is required, and is the single source of truth for the filename
convention, the variables and the reference date. Coordinates come from ``grid.npz``
when the output wrote one, and from the step files otherwise.
"""

import datetime
import json
import logging
from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from anemoi.transform.variables import Variable
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.provenance import OutputManifest
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


@input_registry.register("raw")
@main_argument("dir")
@format_dataset_name("dir")
@ensure_dir("dir", create=False, must_exist=True, unique=False)
class RawInput(Input):
    """Reads the ``.npz`` files produced by :class:`RawOutput`.

    Everything describing the files is taken from their manifest, either read from the
    directory or supplied directly, so nothing about them is inferred or configurable.
    """

    trace_name = "raw"

    FIELD_PREFIX = "field_"
    MANIFEST_NAME = "manifest.json"
    GRID_NAME = "grid.npz"
    SUPPORTED_FORMAT = "anemoi-raw"
    SUPPORTED_VERSION = 1

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        dir: Path,
        manifest: OutputManifest | None = None,
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
        manifest : OutputManifest, optional
            The manifest describing the files. Read from the directory when not given.
        **kwargs : Any
            Additional keyword arguments passed to :class:`Input`.

        Raises
        ------
        ValueError
            If `variables` conflicts with the variables named in the manifest.
        """
        requested = kwargs.get("variables")
        super().__init__(context, metadata, **kwargs)
        self.dir = Path(dir)
        self.manifest = manifest if manifest is not None else self._read_manifest()

        self._check_manifest()
        if requested is not None:
            conflicting = sorted(set(requested) - set(self.manifest.variables))
            if conflicting:
                raise ValueError(
                    f"{self.__class__.__name__}: configured variables {conflicting} are not in the manifest. "
                    f"The files hold {sorted(self.manifest.variables)}."
                )

    def __repr__(self) -> str:
        """Return a string representation of the RawInput object."""
        return f"RawInput({self.dir})"

    def _read_manifest(self) -> OutputManifest:
        """Read the manifest from the raw directory.

        Returns
        -------
        OutputManifest
            The manifest describing the files.

        Raises
        ------
        FileNotFoundError
            If the directory holds no manifest.
        ValueError
            If the manifest cannot be read.
        """
        path = self.dir / self.MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__}: no {self.MANIFEST_NAME} in {self.dir}. "
                f"Write the files with `output_manifest: true`, or pass a manifest explicitly."
            )

        known = {field.name for field in fields(OutputManifest)}
        contents = json.loads(path.read_text())
        try:
            return OutputManifest(**{key: value for key, value in contents.items() if key in known})
        except TypeError as e:
            raise ValueError(f"{self.__class__.__name__}: cannot read {path}: {e}") from e

    def _check_manifest(self) -> None:
        """Warn about any mismatch between the manifest and the current run."""
        if self.manifest.format != self.SUPPORTED_FORMAT:
            LOG.warning("%s: manifest describes '%s', not '%s'.", self, self.manifest.format, self.SUPPORTED_FORMAT)

        if self.manifest.version > self.SUPPORTED_VERSION:
            LOG.warning(
                "%s: manifest is version %d, newer than %d.", self, self.manifest.version, self.SUPPORTED_VERSION
            )

        if self.manifest.dataset_name != self.dataset_name:
            LOG.warning(
                "%s: files were written for dataset '%s', reading as '%s'.",
                self,
                self.manifest.dataset_name,
                self.dataset_name,
            )

        written_with = self.manifest.checkpoint.get("path")
        current = str(self.context.checkpoint.path)
        if written_with is not None and written_with != current:
            LOG.warning("%s: files were written with checkpoint '%s', running with '%s'.", self, written_with, current)

    def _filename(self, date: datetime.datetime, base_date: datetime.datetime | None = None) -> str:
        """Render the file name for a given date.

        Parameters
        ----------
        date : datetime.datetime
            The valid date of the fields.
        base_date : datetime.datetime, optional
            The base (reference) date used to compute the step. If None, the reference
            date recorded in the manifest is used.

        Returns
        -------
        str
            The rendered file name (without directory).
        """
        if base_date is None:
            base_date = to_datetime(self.manifest.reference_date)

        format_info = {
            "date": date.strftime(self.manifest.strftime),
            "basetime": base_date.strftime(self.manifest.strftime),
            "step": date - base_date,
        }
        return render_template(self.manifest.template, format_info)

    def _load_file(self, date: datetime.datetime, base_date: datetime.datetime | None = None) -> dict[str, FloatArray]:
        """Load the fields of a single ``.npz`` file.

        Parameters
        ----------
        date : datetime.datetime
            The valid date of the fields to load.
        base_date : datetime.datetime, optional
            The base date used to compute the step in the file name template.

        Returns
        -------
        dict[str, FloatArray]
            Mapping of variable name to a 1D array of values.

        Raises
        ------
        FileNotFoundError
            If no file exists for the given date.
        """
        path = self.dir / self._filename(date, base_date=base_date)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__}: no raw file found for date {date.isoformat()} at {path}."
            )

        LOG.info("%s: loading %s", self.__class__.__name__, path)
        with np.load(path, allow_pickle=False) as data:
            return {
                key.replace(self.FIELD_PREFIX, ""): np.asarray(data[key])
                for key in data.files
                if key.startswith(self.FIELD_PREFIX)
            }

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

        Raises
        ------
        ValueError
            If no dates are given, or a required variable is not in the manifest.
        """
        if not dates:
            raise ValueError(f"{self.__class__.__name__}: no dates provided")

        missing = sorted(set(self.variables) - set(self.manifest.variables))
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: variables {missing} not found in raw files. "
                f"Available variables: {sorted(self.manifest.variables)}"
            )

        dates = list(to_datetime(d) for d in dates)
        loaded = [self._load_file(date, base_date=base_date) for date in dates]

        typed_variables = self.metadata.typed_variables

        fields: dict[str, FloatArray] = {}
        state_variables: dict[str, Variable] = {}

        for name in self.variables:
            fields[name] = np.stack([entry[name] for entry in loaded], axis=0)
            if name in typed_variables:
                state_variables[name] = typed_variables[name]

            if trace := self.context.tensor_handlers[self.dataset_name].trace:
                trace.from_input(name, self)

        return dict(
            date=dates[-1],
            latitudes=self.latitudes,
            longitudes=self.longitudes,
            fields=fields,
            _input=self,
            _variables=state_variables,
        )

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
        base_date = self.reference_date

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
        """Return the coordinates, from ``grid.npz`` if present, else from a step file."""
        path = self.dir / self.GRID_NAME
        if not path.exists():
            candidates = sorted(p for p in self.dir.glob("*.npz") if p.name != self.GRID_NAME)
            if not candidates:
                return None, None
            path = candidates[0]

        with np.load(path, allow_pickle=False) as data:
            latitudes = np.asarray(data["latitudes"]) if "latitudes" in data.files else None
            longitudes = np.asarray(data["longitudes"]) if "longitudes" in data.files else None
        return latitudes, longitudes
