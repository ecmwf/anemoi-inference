# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import earthkit.data as ekd
import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.data.utils.dates import to_datetime
from numpy.typing import DTypeLike

from anemoi.inference.context import Context
from anemoi.inference.types import Date
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from ..checks import check_data
from ..input import Input

LOG = logging.getLogger(__name__)


class NoMask:
    """No mask to apply."""

    def apply(self, field: Any) -> Any:
        """Apply the mask.

        Parameters
        ----------
        field : Any
            The field to which the mask is applied.

        Returns
        -------
        Any
            The field with the mask applied.
        """
        return field


class ApplyMask:
    """Apply a mask to a field."""

    def __init__(self, mask: Any) -> None:
        """Initialize the ApplyMask.

        Parameters
        ----------
        mask : Any
            The mask to apply.
        """
        self.mask = mask

    def apply(self, field: Any) -> Any:
        """Apply the mask.

        Parameters
        ----------
        field : Any
            The field to which the mask is applied.

        Returns
        -------
        Any
            The field with the mask applied.
        """
        return field[self.mask]


class RulesNamer:
    """A namer that uses rules to generate names."""

    def __init__(self, rules: List[Dict[str, Any]], default_namer: Callable[[Any, Dict[str, Any]], str]) -> None:
        """Initialize the RulesNamer.

        Parameters
        ----------
        rules : List[Dict[str, Any]]
            The rules for naming.
        default_namer : Callable[[Any, Dict[str, Any]], str]
            The default namer to use if no rules match.
        """
        self.rules = rules
        self.default_namer = default_namer

    def __call__(self, field: Any, original_metadata: Dict[str, Any]) -> str:
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

    def substitute(self, template: str, field: Any, original_metadata: Dict[str, Any]) -> str:
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
        namer: Optional[Union[Callable[[Any, Dict[str, Any]], str], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the EkdInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        namer : Optional[Union[Callable[[Any, Dict[str, Any]], str], Dict[str, Any]]]
            Optional namer for the input.
        """
        super().__init__(context)

        if isinstance(namer, dict):
            # TODO: a factory for namers
            assert "rules" in namer, namer
            assert len(namer) == 1, namer
            namer = RulesNamer(namer["rules"], self.checkpoint.default_namer())

        self._namer = namer if namer is not None else self.checkpoint.default_namer()
        assert callable(self._namer), type(self._namer)

    def _create_state(
        self,
        input_fields: ekd.FieldList,
        *,
        variables: Optional[List[str]] = None,
        date: Optional[Date] = None,
        latitudes: Optional[FloatArray] = None,
        longitudes: Optional[FloatArray] = None,
        dtype: DTypeLike = np.float32,
        flatten: bool = True,
    ) -> State:
        """Create the input state.

        Parameters
        ----------
        input_fields : ekd.FieldList
            The input fields.
        variables : Optional[List[str]]
            List of variables.
        date : Optional[Date]
            The date for which to create the input state.
        latitudes : Optional[FloatArray]
            The latitudes.
        longitudes : Optional[FloatArray]
            The longitudes.
        dtype : DTypeLike
            The data type.
        flatten : bool
            Whether to flatten the data.

        Returns
        -------
        State
            The created input state.
        """
        for processor in self.context.pre_processors:
            LOG.info("Processing with %s", processor)
            input_fields = processor.process(input_fields)

        if variables is None:
            variables = self.checkpoint.variables_from_input(include_forcings=True)

        if len(input_fields) == 0:
            raise ValueError("No input fields provided")

        # Newer checkpoints may have latitudes and longitudes
        if latitudes is None:
            latitudes = self.checkpoint.latitudes
            if latitudes is not None:
                LOG.info(
                    "%s: using `latitudes` found in the checkpoint.",
                    self.__class__.__name__,
                )

        if longitudes is None:
            longitudes = self.checkpoint.longitudes
            if longitudes is not None:
                LOG.info(
                    "%s: using `longitudes` found in the checkpoint.git.",
                    self.__class__.__name__,
                )

        if date is None:
            date = input_fields.order_by(valid_datetime="ascending")[-1].datetime()["valid_time"]
            LOG.info(
                "%s: `date` not provided, using the most recent date: %s", self.__class__.__name__, date.isoformat()
            )

        if isinstance(date, list):
            # FIXME: find out why we have a list of dates
            dates = [date[-1] + h for h in self.checkpoint.lagged]
            assert date == dates, (date, dates)
        else:
            date = to_datetime(date)
            dates = [date + h for h in self.checkpoint.lagged]
        date_to_index = {d.isoformat(): i for i, d in enumerate(dates)}

        input_state = dict(date=date, latitudes=latitudes, longitudes=longitudes, fields=dict())

        fields = input_state["fields"]

        input_fields = self._filter_and_sort(
            input_fields,
            variables=variables,
            dates=dates,
            title="Create input state",
        )
        mask = self.checkpoint.grid_points_mask
        mask = ApplyMask(mask) if mask is not None else NoMask()

        check = defaultdict(set)

        for field in input_fields:

            if input_state["latitudes"] is None:
                input_state["latitudes"], input_state["longitudes"] = field.grid_points()
                LOG.info(
                    "%s: using `latitudes` and `longitudes` from the first input field",
                    self.__class__.__name__,
                )

            name, valid_datetime = field.metadata("name"), field.metadata("valid_datetime")
            if name not in fields:
                fields[name] = np.full(
                    shape=(len(dates), self.checkpoint.number_of_grid_points),
                    fill_value=np.nan,
                    dtype=dtype,
                )

            date_idx = date_to_index[valid_datetime]

            try:
                fields[name][date_idx] = mask.apply(field.to_numpy(dtype=dtype, flatten=flatten))
            except ValueError:
                LOG.error("Error with field %s: expected shape=%s, got shape=%s", name, fields[name].shape, field.shape)
                LOG.error("dates %s", dates)
                LOG.error("number_of_grid_points %s", self.checkpoint.number_of_grid_points)
                raise

            if date_idx in check[name]:
                LOG.error("Duplicate dates for %s: %s", name, date_idx)
                LOG.error("Expected %s", list(date_to_index.keys()))
                LOG.error("Got %s", list(check[name]))
                raise ValueError(f"Duplicate dates for {name}")

            check[name].add(date_idx)

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
        self.set_private_attributes(input_state, input_fields)

        return input_state

    def _filter_and_sort(self, data: Any, *, variables: List[str], dates: List[Date], title: str) -> Any:
        """Filter and sort the data.

        Parameters
        ----------
        data : Any
            The data to filter and sort.
        variables : List[str]
            The list of variables to select.
        dates : List[Any]
            The list of dates to select.
        title : str
            The title for logging.

        Returns
        -------
        Any
            The filtered and sorted data.
        """

        def _name(field: Any, _: Any, original_metadata: Dict[str, Any]) -> str:
            return self._namer(field, original_metadata)

        data = FieldArray([f.clone(name=_name) for f in data])

        valid_datetime = [_.isoformat() for _ in dates]
        LOG.info("Selecting fields %s %s", len(data), valid_datetime)

        # for f in data:
        #     LOG.info("Field %s %s", f.metadata("name"), f.metadata("valid_datetime"))

        data = data.sel(name=variables, valid_datetime=valid_datetime).order_by(
            name=variables, valid_datetime="ascending"
        )

        check_data(title, data, variables, dates)

        return data

    def _find_variable(self, data: Any, name: str, **kwargs: Any) -> Any:
        """Find a variable in the data.

        Parameters
        ----------
        data : Any
            The data to search.
        name : str
            The name of the variable to find.
        **kwargs : Any
            Additional arguments for selecting the variable.

        Returns
        -------
        Any
            The selected variable.
        """

        def _name(field: Any, _: Any, original_metadata: Dict[str, Any]) -> str:
            return self._namer(field, original_metadata)

        data = FieldArray([f.clone(name=_name) for f in data])
        return data.sel(name=name, **kwargs)

    def _load_forcings_state(
        self,
        fields: ekd.FieldList,
        variables: List[str],
        dates: List[Date],
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load the forcings state.

        Parameters
        ----------
        fields : ekd.FieldList
            The fields to load.
        variables : List[str]
            The list of variables to load.
        dates : List[Any]
            The list of dates to load.
        current_state : Dict[str, Any]
            The current state.

        Returns
        -------
        Dict[str, Any]
            The loaded forcings state.
        """
        for processor in self.context.pre_processors:
            LOG.info("Processing with %s", processor)
            fields = processor.process(fields)

        return self._create_state(
            fields,
            variables=variables,
            date=dates,
            latitudes=current_state["latitudes"],
            longitudes=current_state["longitudes"],
            dtype=np.float32,
            flatten=True,
        )
