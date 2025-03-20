# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import abc
import datetime
import logging
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.input import Input

from ..types import State

LOG = logging.getLogger(__name__)


class RulesNamer:
    """A namer that uses rules to generate names."""

    def __init__(self, rules: Any, default_namer: Callable[[Any, Dict[str, Any]], str]) -> None:
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


class EarthKitInput(Input):
    """Handles earthkit-data FieldList as input."""

    def __init__(self, context: Context, namer: Optional[Callable[[Any, Any], str]] = None) -> None:
        super().__init__(context)

        if isinstance(namer, dict):
            # TODO: a factory for namers
            assert "rules" in namer, namer
            assert len(namer) == 1, namer
            namer = RulesNamer(namer["rules"], self.checkpoint.default_namer())

        self._namer = namer if namer is not None else self.checkpoint.default_namer()
        assert callable(self._namer), type(self._namer)

    def create_state(self, *, date=None, variables=None, initial: bool = True) -> State:
        # NOTE: all logic involing the checkpoint should take place here

        # define request (using checkpoint)
        variables = self.checkpoint_variables if variables is None else variables
        date = np.datetime64(date).astype(datetime.datetime)
        dates = [date + h for h in self.checkpoint.lagged] if initial else [date]

        # get raw state fieldlist
        state = self._raw_state_fieldlist(dates=dates, variables=variables)

        # pre-process state fieldlist
        for processor in self.context.pre_processors:
            LOG.info("Processing with %s", processor)
            state = processor.process(state)

        return self.fieldlist_to_state(state)

    @abc.abstractmethod
    def _raw_state_fieldlist(self, dates: list[datetime.datetime], variables: list[str]) -> ekd.FieldList:
        """Load the raw state fieldlist for the given dates and variables.

        Parameters
        ----------
        dates : list[datetime.datetime]
            List of dates for which to load the raw state.
        variables : list[str]
            List of variables to load.

        Returns
        -------
        ekd.FieldList
            The raw state fieldlist.
        """
        pass

    def fieldlist_to_state(self, fieldlist: ekd.FieldList) -> State:
        """Convert a fieldlist to a state dictionary.

        Parameters
        ----------
        fieldlist : earthkit.data.FieldList
            The fieldlist to convert.

        Returns
        -------
        State
            The converted state.
        """

        state = {"fields": {}}
        for fields in fieldlist.group_by("name"):
            name = fields[-1].metadata("name")
            state["fields"][name] = fields.values
        date = fieldlist.metadata("valid_datetime")[-1]
        state["date"] = np.datetime64(date).astype(datetime.datetime)
        state["latitudes"], state["longitudes"] = fields[-1].grid_points()
        return state
