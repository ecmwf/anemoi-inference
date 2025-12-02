# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

import earthkit.data as ekd

from anemoi.inference.types import State

from . import TemplateProvider
from . import template_provider_registry
from .manager import TemplateManager

LOG = logging.getLogger(__name__)


@template_provider_registry.register("input")
class InputTemplates(TemplateProvider):
    """Use input fields as the output GRIB template."""

    def __init__(self, manager: TemplateManager, **fallback: dict[str, str]) -> None:
        """Initialize the template provider.

        Parameters
        ----------
        manager : TemplateManager
            The manager for the template provider.
        **fallback : dict[str, str]
            A mapping of output to input variable names to use as templates from the input,
            used as fallback when the output variable is not present in the input state (e.g., for diagnostic variables).
        """
        super().__init__(manager)

        self.fallback = fallback

    def __repr__(self):
        info = f"{self.__class__.__name__}{{fallback}}"
        if fallback := ", ".join(f"{k}:{v}" for k, v in self.fallback.items()):
            fallback = f"(fallback {fallback})"
        return info.format(fallback=fallback)

    def template(
        self,
        variable: str,
        lookup: dict[str, Any],
        *,
        state: State,
        **kwargs,
    ) -> ekd.Field | None:
        if template := state.get("_grib_templates_for_output", {}).get(variable):
            return template

        if fallback_variable := self.fallback.get(variable):
            if template := state.get("_grib_templates_for_output", {}).get(fallback_variable):
                return template
            LOG.warning(f"Fallback variable '{fallback_variable}' for output '{variable}' not found in input state.")

        return None
