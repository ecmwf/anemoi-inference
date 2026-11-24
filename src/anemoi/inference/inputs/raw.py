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
convention, the variables and the reference date.

With ``wait``, the files may be read while the first model is still writing them,
starting from its ``manifest.json.part``: each file is waited for until it appears,
until the writer has finished (the manifest has its final name), or until a timeout.
"""

import datetime
import json
import logging
import time
from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from anemoi.transform.variables import Variable
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.provenance import PARTIAL_SUFFIX
from anemoi.inference.provenance import OutputManifest
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State
from anemoi.inference.utils.templating import render_template

from ..decorators import format_dataset_name
from ..decorators import main_argument
from ..input import Input
from . import input_registry

LOG = logging.getLogger(__name__)


@input_registry.register("raw")
@main_argument("dir")
@format_dataset_name("dir")
class RawInput(Input):
    """Reads the ``.npz`` files produced by :class:`RawOutput`.

    Everything describing the files is taken from their manifest, either read from the
    directory or supplied directly, so nothing about them is inferred or configurable.
    """

    trace_name = "raw"

    MANIFEST_NAME = "manifest.json"
    SUPPORTED_FORMAT = "anemoi-raw"
    SUPPORTED_VERSION = 1

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        dir: Path,
        manifest: OutputManifest | None = None,
        variables: list[str] | None = None,
        wait: bool = False,
        timeout: float = 600,
        poll_interval: float = 1,
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
        variables : list of str or None
            List of variable names to be handled by the input, or None for all available variables.
        wait : bool, optional
            Wait for the directory, the manifest and each file to appear, for reading
            the files while they are being written. By default False.
        timeout : float, optional
            With `wait`, how long to wait for a single file, in seconds. By default 600.
            If the file has not appeared by then, a `TimeoutError` is raised.
        poll_interval : float, optional
            With `wait`, how often to check for a file, in seconds. By default 1.
        **kwargs : Any
            Additional keyword arguments passed to :class:`Input`.

        Raises
        ------
        FileNotFoundError
            If `wait` is not set and the directory does not exist.
        ValueError
            If `variables` conflicts with the variables named in the manifest.
        """
        super().__init__(context, metadata, variables=variables, **kwargs)
        self.dir = Path(dir)
        self._requested = variables
        self._manifest = manifest
        self.wait = wait
        self.timeout = timeout
        self.poll_interval = poll_interval

        if not wait:
            if not self.dir.is_dir():
                raise FileNotFoundError(f"Path '{self.dir}' does not exist.")
            _ = self.manifest  # everything is written already, so read and check the manifest now

    @cached_property
    def manifest(self) -> OutputManifest:
        """Return the manifest describing the files, reading it on first use."""
        manifest = self._manifest if self._manifest is not None else self._read_manifest()

        self._check_manifest(manifest)
        if self._requested is not None:
            conflicting = sorted(set(self._requested) - set(manifest.variables))
            if conflicting:
                raise ValueError(
                    f"{self.__class__.__name__}: configured variables {conflicting} are not in the manifest. "
                    f"The files hold {sorted(manifest.variables)}."
                )
        return manifest

    def __repr__(self) -> str:
        """Return a string representation of the RawInput object."""
        return f"RawInput({self.dir})"

    def _wait_for(self, path: Path) -> bool:
        """Return whether `path` exists, waiting for it if `wait` is set.

        Stops early, returning False, once the writer has finished (the manifest has its
        final name) without writing `path`.
        """
        if not self.wait:
            return path.exists()

        deadline = time.monotonic() + self.timeout
        logged = False
        while not path.exists():
            if (self.dir / self.MANIFEST_NAME).exists():
                return path.exists()
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.__class__.__name__}: {path} did not appear within {self.timeout} s.")
            if not logged:
                LOG.info("%s: waiting for %s", self.__class__.__name__, path)
                logged = True
            time.sleep(self.poll_interval)
        return True

    def _read_manifest(self) -> OutputManifest:
        """Read the manifest, or with `wait` its partial version while the writer is still running."""
        path = self.dir / self.MANIFEST_NAME
        partial = path.with_name(path.name + PARTIAL_SUFFIX)
        try:
            text = path.read_text()
        except FileNotFoundError:
            if not self.wait:
                hint = " The run writing it has not finished: set `wait` to read the files as they are written."
                raise FileNotFoundError(
                    f"{self.__class__.__name__}: no {self.MANIFEST_NAME} in {self.dir}."
                    + (hint if partial.exists() else "")
                ) from None
            if self._wait_for(partial):
                try:
                    path, text = partial, partial.read_text()
                except FileNotFoundError:  # renamed to its final name in the meantime
                    text = path.read_text()
            else:
                text = path.read_text()

        known = {field.name for field in fields(OutputManifest)}
        contents = json.loads(text)
        try:
            return OutputManifest(**{key: value for key, value in contents.items() if key in known})
        except TypeError as e:
            raise ValueError(f"{self.__class__.__name__}: cannot read {path}: {e}") from e

    def _check_manifest(self, manifest: OutputManifest) -> None:
        """Warn about any mismatch between the manifest and the current run."""
        if manifest.format != self.SUPPORTED_FORMAT:
            LOG.warning("%s: manifest describes '%s', not '%s'.", self, manifest.format, self.SUPPORTED_FORMAT)

        if manifest.version > self.SUPPORTED_VERSION:
            LOG.warning("%s: manifest is version %d, newer than %d.", self, manifest.version, self.SUPPORTED_VERSION)

        if manifest.dataset_name != self.dataset_name:
            LOG.warning(
                "%s: files were written for dataset '%s', reading as '%s'.",
                self,
                manifest.dataset_name,
                self.dataset_name,
            )

    def _load_file(
        self, date: datetime.datetime, base_date: datetime.datetime | None = None
    ) -> tuple[dict[str, FloatArray], FloatArray | None, FloatArray | None]:
        """Load the fields, latitudes and longitudes of a single ``.npz`` file."""
        if base_date is None:
            base_date = to_datetime(self.manifest.reference_date)

        format_info = {
            "date": date.strftime(self.manifest.strftime),
            "basetime": base_date.strftime(self.manifest.strftime),
            "step": date - base_date,
        }
        path = self.dir / render_template(self.manifest.template, format_info)

        if not self._wait_for(path):
            raise FileNotFoundError(
                f"{self.__class__.__name__}: no raw file found for date {date.isoformat()} at {path}."
            )

        LOG.info("%s: loading %s", self.__class__.__name__, path)
        with np.load(path, allow_pickle=False) as data:
            fields = {key.replace("field_", ""): np.asarray(data[key]) for key in data.files}
            latitudes = np.asarray(data["latitudes"]) if "latitudes" in data.files else None
            longitudes = np.asarray(data["longitudes"]) if "longitudes" in data.files else None

        missing = [name for name in self.variables if name not in fields]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: variables {missing} are not in {path}. "
                "After the initial state, a model writes its outputs only, without forcings: "
                "read the forcings from another input with `constant_forcings` or `dynamic_forcings`."
            )
        return fields, latitudes, longitudes

    def _build_state(
        self, dates: list[Date], *, base_date: datetime.datetime | None = None, ref_date_index: int = -1
    ) -> State:
        """Build a state by stacking the fields of the requested dates.

        Parameters
        ----------
        dates : list of Date
            The dates for which to build the state.
        base_date : datetime.datetime, optional
            The base date used to compute the step in the file name template.
        ref_date_index: int = -1
            The index in `dates` to use as reference date for the state.
            Defaults to -1, which means the last date in the list.

        Returns
        -------
        State
            The state with ``fields`` as ``dict[str, np.ndarray]`` of shape
            ``(len(dates), n_points)``.

        Raises
        ------
        ValueError
            If no dates are given, or a requested variable is not in a file.
        """
        if not dates:
            raise ValueError(f"{self.__class__.__name__}: no dates provided")

        dates = sorted(to_datetime(d) for d in dates)
        loaded = [self._load_file(date, base_date=base_date) for date in dates]
        _, latitudes, longitudes = loaded[0]
        loaded = [entry for entry, _, _ in loaded]

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
            date=dates[ref_date_index],
            latitudes=latitudes,
            longitudes=longitudes,
            fields=fields,
            _input=self,
            _variables=state_variables,
        )

    def create_input_state(
        self, *, dates: list[Date], ref_date_index: int = -1, select_reference_date: bool = False, **kwargs: Any
    ) -> State:
        """Create the input state for the given dates.

        Parameters
        ----------
        dates : list of Date
            The dates for which to create the input state (as computed by the
            runner from the model's ``lagged`` steps).
        ref_date_index : int = -1
            The index in `dates` to use as reference date for the state.
        select_reference_date : bool, optional
            Whether the files must have been written for the run's reference date,
            as the temporal downscaler requires. By default False.
        **kwargs : Any
            Additional keyword arguments (ignored).

        Returns
        -------
        State
            The created input state.

        Raises
        ------
        ValueError
            If `select_reference_date` is set and the files were written for another reference date.
        """
        if select_reference_date:
            assert self.reference_date is not None, "Reference date must be set when selecting reference date."
            written = self.manifest.reference_date
            if written is None or to_datetime(written) != to_datetime(self.reference_date):
                raise ValueError(
                    f"{self.__class__.__name__}: files in {self.dir} were written for reference date {written}, "
                    f"not {self.reference_date}."
                )
        return self._build_state(dates, ref_date_index=ref_date_index)

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

        state = self._build_state(dates, base_date=base_date, ref_date_index=-1)
        state["dates"] = sorted(to_datetime(d) for d in dates)
        return state
