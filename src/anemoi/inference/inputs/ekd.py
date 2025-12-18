# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import glob
import logging
import os
import re
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
from typing import Any

import earthkit.data as ekd
import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.data.utils.dates import to_datetime
from numpy.typing import DTypeLike

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from ..checks import check_data
from ..input import Input

LOG = logging.getLogger(__name__)


def find_variable(data: Any, name: str, namer: callable, **kwargs: Any) -> Any:
    """Find a variable in an earthkit FieldList/FieldArray.

    Parameters
    ----------
    data : Any
        The data to search (FieldList or FieldArray).
    name : str
        The name of the variable to find.
    namer: callable
        The namer function to use for naming fields.
    **kwargs : Any
        Additional arguments for selecting the variable.

    Returns
    -------
    Any
        The selected variable (FieldArray subset).
    """

    def _name(field: Any, _: Any, original_metadata: dict[str, Any]) -> str:
        return namer(field, original_metadata)

    data = FieldArray([f.clone(name=_name) for f in data])
    return data.sel(name=name, **kwargs)


class RulesNamer:
    """A namer that uses rules to generate names."""

    def __init__(self, rules: Any, default_namer: Callable[[Any, dict[str, Any]], str]) -> None:
        """Initialize the RulesNamer.

        Parameters
        ----------
        rules : List[List[Dict[str, Any], Dict[str, Any]]]
            The rules for naming.
        default_namer : Callable[[Any, Dict[str, Any]], str]
            The default namer to use if no rules match.
        """
        self.rules = rules
        self.default_namer = default_namer

    def __call__(self, field: Any, original_metadata: dict[str, Any]) -> str:
        """Generate a name for the field.

        Parameters
        ----------
        field : Any
            The field for which to generate a name.
        original_metadata : Dict[str, Any]
            The original metadata of the field.

        Returns
        -------
        str
            The generated name.
        """
        for rule in self.rules:
            assert len(rule) == 2, rule
            ok = True
            for k, v in rule[0].items():
                if original_metadata.get(k) != v:
                    ok = False
            if ok:
                return self.substitute(rule[1], field, original_metadata)

        return self.default_namer(field, original_metadata)

    def substitute(self, template: str, field: Any, original_metadata: dict[str, Any]) -> str:
        """Substitute placeholders in the template with metadata values.

        Parameters
        ----------
        template : str
            The template string with placeholders.
        field : Any
            The field for which to generate a name.
        original_metadata : Dict[str, Any]
            The original metadata of the field.

        Returns
        -------
        str
            The generated name with placeholders substituted.
        """
        matches = re.findall(r"\{(.+?)\}", template)
        matches = {m: original_metadata.get(m) for m in matches}
        return template.format(**matches)


class EkdInput(Input):
    """Handles earthkit-data FieldList as input."""

    def __init__(
        self,
        context: Context,
        *,
        namer: Any | None = None,
        **kwargs,
    ) -> None:
        """Initialize the EkdInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        namer : Optional[Union[Callable[[Any, Dict[str, Any]], str], Dict[str, Any]]]
            Optional namer for the input.
        """
        super().__init__(context, **kwargs)

        if isinstance(namer, dict):
            # TODO: a factory for namers
            assert "rules" in namer, namer
            assert len(namer) == 1, namer
            namer = RulesNamer(namer["rules"], self.checkpoint.default_namer())

        self._namer = namer if namer is not None else self.checkpoint.default_namer()
        assert callable(self._namer), type(self._namer)

    def _filter_and_sort(
        self, data: Any, *, dates: list[Any], title: str, select_reference_date: bool = False, **kwargs
    ) -> Any:
        """Filter and sort the data (earthkit FieldList/FieldArray).

        Parameters
        ----------
        data : Any
            The data to filter and sort (FieldList or FieldArray).
        dates : List[Any]
            The list of dates to select.
        title : str
            The title for logging.
        select_reference_date: bool, optional
            Also include the reference date when selecting data from the FieldList.
            If False (default), only the valid date is considered.
        **kwargs : Any
            Additional arguments for selecting the variable.

        Returns
        -------
        Any
            The filtered and sorted data (FieldArray).
        """

        def _name(field: Any, _: Any, original_metadata: dict[str, Any]) -> str:
            return self._namer(field, original_metadata)

        data = FieldArray([f.clone(name=_name) for f in data])

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        if select_reference_date:
            data = data.sel(
                name=self.variables,
                valid_datetime=valid_datetime,
                dataDate=int(self.reference_date.strftime("%Y%m%d")),
                dataTime=int(self.reference_date.strftime("%H%M")),
            ).order_by(
                name=self.variables,
                valid_datetime="ascending",
            )
        else:
            data = data.sel(name=self.variables, valid_datetime=valid_datetime).order_by(
                name=self.variables,
                valid_datetime="ascending",
            )

        check_data(title, data, self.variables, dates, self.context.checkpoint)

        return data

    def _find_variable(self, data: Any, name: str, **kwargs: Any) -> Any:
        """Find a variable in the data (earthkit FieldList/FieldArray selection).

        Parameters
        ----------
        data : Any
            The data to search (FieldList or FieldArray).
        name : str
            The name of the variable to find.
        **kwargs : Any
            Additional arguments for selecting the variable.

        Returns
        -------
        Any
            The selected variable (FieldArray subset).
        """

        return find_variable(data, name, self._namer, **kwargs)

    def _create_state(
        self,
        fields: ekd.FieldList,
        *,
        dates: list[Date],
        latitudes: FloatArray | None = None,
        longitudes: FloatArray | None = None,
        dtype: DTypeLike = np.float32,
        flatten: bool = True,
        ref_date_index: int = -1,
        **kwargs,
    ) -> State:
        """Create a state from an ekd.FieldList.

        Notes
        -----
        - The `fields` argument must be an earthkit FieldList (or FieldArray-compatible).
        - This method intentionally converts state["fields"] from a FieldList to
          a Dict[str, np.ndarray] with shape (len(dates), n_points).
        - Pre-processors are run while state["fields"] is still a FieldList.

        Parameters
        ----------
        fields : ekd.FieldList
            The ekd fields.
        dates : List[Date]
            The dates for which to create the input state.
        latitudes : Optional[FloatArray]
            The latitudes.
        longitudes : Optional[FloatArray]
            The longitudes.
        dtype : DTypeLike
            The data type.
        flatten : bool
            Whether to flatten the data.
        ref_date_index: int = -1
            If 0 takes the first date, if -1 takes the last date in sequence.
        **kwargs : Any
            Additional arguments for selecting the variable.

        Returns
        -------
        State
            The created input state with state["fields"] as Dict[str, np.ndarray].
        """
        if latitudes is None and longitudes is None:
            try:
                latitudes, longitudes = fields[0].grid_points()
                LOG.info(
                    "%s: using `latitudes` and `longitudes` from the first input field",
                    self.__class__.__name__,
                )
            except Exception as e:
                LOG.info(
                    "%s: could not get `latitudes` and `longitudes` from the input fields.",
                    self.__class__.__name__,
                )
                latitudes = self.checkpoint.latitudes
                longitudes = self.checkpoint.longitudes
                if latitudes is not None and longitudes is not None:
                    LOG.info(
                        "%s: using `latitudes` and `longitudes` found in the checkpoint.",
                        self.__class__.__name__,
                    )
                else:
                    LOG.error(
                        "%s: could not find `latitudes` and `longitudes` in the input fields or the checkpoint.",
                        self.__class__.__name__,
                    )
                    raise e

        state = dict(date=dates[ref_date_index], latitudes=latitudes, longitudes=longitudes, fields=fields)

        # allow hooks to operate on the FieldList before conversion to numpy
        state = self.pre_process(state)

        fields = state["fields"]
        state_fields = {}

        if len(fields) == 0:
            raise ValueError("No input fields provided")

        dates = sorted([to_datetime(d) for d in dates])
        date_to_index = {d.isoformat(): i for i, d in enumerate(dates)}

        fields = self._filter_and_sort(fields, dates=dates, title="Create input state", **kwargs)

        check = defaultdict(set)

        n_points = fields[0].to_numpy(dtype=dtype, flatten=flatten).size
        for field in fields:
            name, valid_datetime = field.metadata("name"), field.metadata("valid_datetime")
            if name not in state_fields:
                state_fields[name] = np.full(
                    shape=(len(dates), n_points),
                    fill_value=np.nan,
                    dtype=dtype,
                )

            date_idx = date_to_index[valid_datetime]

            try:
                state_fields[name][date_idx] = field.to_numpy(dtype=dtype, flatten=flatten)
            except ValueError:
                LOG.error(
                    "Error with field %s: expected shape=%s, got shape=%s", name, state_fields[name].shape, field.shape
                )
                LOG.error("dates %s", dates)
                LOG.error("number_of_grid_points %s", self.checkpoint.number_of_grid_points)
                raise

            if date_idx in check[name]:
                LOG.error("Duplicate dates for %s: %s", name, date_idx)
                LOG.error("Expected %s", list(date_to_index.keys()))
                LOG.error("Got %s", list(check[name]))
                raise ValueError(f"Duplicate dates for {name}")

            check[name].add(date_idx)
        state["fields"] = state_fields
        for name, idx in check.items():
            if len(idx) != len(dates):
                LOG.error("Missing dates for %s: %s", name, idx)
                LOG.error("Expected %s", list(date_to_index.keys()))
                LOG.error("Got %s", list(idx))
                raise ValueError(f"Missing dates for {name}")

        if self.context.trace:
            for name in check.keys():
                self.context.trace.from_input(name, self)

        # This is our chance to communicate output object
        # This is useful for GRIB that requires a template field
        # to be used as output
        self.set_private_attributes(state, fields)

        state["_input"] = self

        return state

    def _create_input_state(
        self,
        input_fields: ekd.FieldList,
        *,
        date: Date | None = None,
        variables: list[str] | None = None,
        latitudes: FloatArray | None = None,
        longitudes: FloatArray | None = None,
        dtype: DTypeLike = np.float32,
        flatten: bool = True,
        constant: bool = False,
        ref_date_index: int = -1,
        **kwargs,
    ) -> State:
        """Create the input state.

        Parameters
        ----------
        input_fields : ekd.FieldList
            The input fields.
        date : Date
            The date for which to create the input state.
        variables : Optional[List[str]]
            List of variables.
        latitudes : Optional[FloatArray]
            The latitudes.
        longitudes : Optional[FloatArray]
            The longitudes.
        dtype : DTypeLike
            The data type.
        flatten : bool
            Whether to flatten the data.
        constant: bool
            Whether the field is constant or dynamic
        ref_date_index: int = -1
            If 0 takes the first date, if -1 takes the last date in sequence.
        **kwargs : Any
            Additional arguments for selecting the variable.
        Returns
        -------
        State
            The created input state.
        """
        if date is None:
            date = input_fields.order_by(valid_datetime="ascending")[-1].datetime()["valid_time"]
            LOG.info(
                "%s: `date` not provided, using the most recent date: %s", self.__class__.__name__, date.isoformat()
            )

        if constant:
            dates = [date]
        else:
            dates = [date + h for h in self.checkpoint.lagged]

        return self._create_state(
            input_fields,
            dates=dates,
            latitudes=latitudes,
            longitudes=longitudes,
            dtype=dtype,
            flatten=flatten,
            ref_date_index=ref_date_index,
            **kwargs,
        )

    def _load_forcings_state(self, fields: ekd.FieldList, *, dates: list[Date], current_state: State) -> State:
        """Load the forcings state.

        Parameters
        ----------
        fields : ekd.FieldList
            The fields to load.
        dates : List[Any]
            The list of dates to load.
        current_state : Dict[str, Any]
            The current state.

        Returns
        -------
        State
            The loaded forcings state.
        """
        return self._create_state(
            fields,
            dates=dates,
            latitudes=current_state.get("latitudes", None),
            longitudes=current_state.get("longitudes", None),
            dtype=np.float32,
            flatten=True,
        )

    def set_private_attributes(self, state: State, fields: ekd.FieldList) -> None:  # type: ignore
        """Set private attributes to the state.

        Provides geography information if available retrieved from the fields (FieldList/FieldArray).
        """
        geography_information = {}

        def get_geography_info(key: str) -> str | None:
            try:
                combo = list(getattr(f.metadata().geography, key, lambda: None)() for f in fields)
            except NotImplementedError:  # Issue with earthkit.data throwing error here
                return None
            if len(set(map(str, combo))) == 1 and combo[0] != "None":
                return combo[0]
            return None

        if area := get_geography_info("mars_area"):
            geography_information["area"] = area
        if grid := get_geography_info("mars_grid"):
            geography_information["grid"] = grid

        if geography_information:
            state["_geography"] = geography_information


@main_argument("path")
class FieldlistInput(EkdInput):
    """Handles earthkit-data FieldList as input."""

    patterns: tuple[str, ...]

    def __init__(
        self,
        context: Context,
        *,
        path: str,
        **kwargs: Any,
    ) -> None:
        """Initialise the FieldlistInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        path : str
            Path, directory or glob pattern to file(s). Examples:
              - "/path/to/file.grib"
              - "/path/to/*.grib"
              - "/path/to/**/*.grib2"
              - "/path/to/directory/"
        namer : Optional[Any]
            Optional namer for the input.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, **kwargs)
        self.path = path

    def create_input_state(self, *, date: Date | None, ref_date_index: int = -1, **kwargs) -> State:
        """Create the input state for the given date.

        Parameters
        ----------
        date : Optional[Date]
            The date for which to create the input state.
        ref_date_index : int = -1
            If 0 takes the first date, if -1 takes the last date in sequence.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        State
            The created input state.
        """
        return self._create_input_state(self._fieldlist, date=date, ref_date_index=ref_date_index, **kwargs)

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        """Load the forcings state for the given variables and dates.

        Parameters
        ----------
        dates : List[Date]
            List of dates for which to load the forcings.
        current_state : State
            The current state of the input.

        Returns
        -------
        State
            The loaded forcings state.
        """

        return self._load_forcings_state(
            self._fieldlist,
            dates=dates,
            current_state=current_state,
        )

    @cached_property
    def _fieldlist(self) -> ekd.FieldList:
        """Get the input fieldlist from the file or collection."""
        path = self.path

        # Case 1: explicit glob pattern
        if glob.has_magic(path):
            matches = glob.glob(path, recursive=True)
            files = [p for p in matches if os.path.isfile(p)]
            if not files:
                LOG.warning("No files matched pattern %r", path)
                return ekd.from_source("empty")  # type: ignore[reportReturnType]
            return ekd.from_source("file", sorted(files))  # type: ignore[reportReturnType]

        # Case 2: directory path -> search for files recursively
        if os.path.isdir(path):
            files = []
            for pat in self.patterns:
                files.extend(glob.glob(os.path.join(path, "**", pat), recursive=True))
            files = [f for f in sorted(set(files)) if os.path.isfile(f)]
            if not files:
                LOG.warning("Directory %r contains no files which match patterns %r", path, self.patterns)
                return ekd.from_source("empty")  # type: ignore[reportReturnType]
            return ekd.from_source("file", files)  # type: ignore[reportReturnType]

        # Case 3: single file path
        try:
            if os.path.getsize(path) == 0:
                LOG.warning("File %r is empty", path)
                return ekd.from_source("empty")  # type: ignore[reportReturnType]
        except FileNotFoundError:
            LOG.warning("Path %r not found", path)
            return ekd.from_source("empty")  # type: ignore[reportReturnType]

        return ekd.from_source("file", path)  # type: ignore[reportReturnType]
