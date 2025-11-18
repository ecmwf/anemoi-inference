# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from typing import Any
from typing import Literal

import earthkit.data as ekd

from anemoi.inference.decorators import main_argument
from anemoi.inference.inputs.ekd import find_variable
from anemoi.inference.types import State

from . import TemplateProvider
from . import template_provider_registry

LOG = logging.getLogger(__name__)


@template_provider_registry.register("file")
@main_argument("path")
class FileTemplates(TemplateProvider):
    """Template provider using a single GRIB file."""

    def __init__(
        self,
        manager: Any,
        *,
        path: str,
        mode: Literal["auto", "first", "last"] = "first",
        variable: str | list | None = None,
    ) -> None:
        """Initialize the FileTemplates instance.

        Parameters
        ----------
        manager : Any
            The manager instance.
        path : str
            The path to the GRIB file.
        mode : Literal["auto", "first", "last"], optional
            THe method with which to select a message from the grib file to use as template, by default "first":
            - "auto": select variable from the grib file matching the output variable name
            - "first": use the first message in the grib file
            - "last": use the last message in the grib file
        variable : str | list, optional
            The output variable name(s) for which to use this template file. If empty, applies to all variables.
        """
        self.manager = manager
        self.path = path
        self.mode = mode
        self.variables = variable if isinstance(variable, list) else [variable] if variable is not None else None

    @cached_property
    def _data(self):
        return ekd.from_source("file", self.path)

    def template(self, variable: str, lookup: dict[str, Any], state: State, **kwargs) -> ekd.Field:
        if self.variables is not None and variable not in self.variables:
            return None

        match self.mode:
            case "first":
                return self._data[0]
            case "last":
                return self._data[-1]
            case "auto":
                namer = getattr(state.get("_input"), "_namer", self.manager.owner.context.checkpoint.default_namer())
                field = find_variable(self._data, variable, namer)
                if len(field) > 0:
                    return field[0]

        return None
