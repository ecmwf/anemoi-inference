# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from typing import Any

import earthkit.data as ekd

from anemoi.inference.types import State

from . import create_template_provider

LOG = logging.getLogger(__name__)


class TemplateManager:
    """A class to manage GRIB templates."""

    def __init__(self, owner: Any, templates: list[str] | str | None = None) -> None:
        """Initialize the TemplateManager.

        Parameters
        ----------
        owner : Any
            The owner of the TemplateManager.
        templates : Optional[Union[List[str], str]], optional
            A list of template names or a single template name, by default None.
        """
        self.owner = owner
        self.checkpoint = owner.context.checkpoint
        self.typed_variables = self.checkpoint.typed_variables

        self._template_cache = {}

        if templates is None:
            templates = []

        if not isinstance(templates, (list, tuple)):
            templates = [templates]

        if len(templates) == 0:
            templates = ["builtin"]

        self.templates_providers = [create_template_provider(self, template) for template in templates]

    def template(self, name: str, state: State, typed_variables: list[Any]) -> ekd.Field | None:
        """Get the template for a given name and state.

        Parameters
        ----------
        name : str
            The name of the template.
        state : State
            The state object containing template information.
        typed_variables : list of Any
            The list of typed variables.

        Returns
        -------
        Optional[ekd.Field]
            The template field if found, otherwise None.
        """
        assert name is not None, name

        # Use input fields as templates
        self._template_cache.update(state.get("_grib_templates_for_output", {}))

        if name not in self._template_cache:
            self.load_template(name, state, typed_variables)

        return self._template_cache.get(name)

    def load_template(self, name: str, state: State, typed_variables: list[Any]) -> ekd.Field | None:
        """Load the template for a given name and state.

        Parameters
        ----------
        name : str
            The name of the template.
        state : State
            The state object containing template information.
        typed_variables : list of Any
            The list of typed variables.

        Returns
        -------
        Optional[ekd.Field]
            The template field if found, otherwise None.
        """

        checkpoint = self.owner.context.checkpoint

        typed = typed_variables[name]

        lookup = dict(
            name=name,
            grid=self._grid(checkpoint.grid),
            area=checkpoint.area,
            time_processing=typed.time_processing,
            number_of_grid_points=checkpoint.number_of_grid_points,
        )

        for key, value in typed.grib_keys.items():
            if key in ("step", "date", "time", "hdate"):
                continue
            lookup[key] = value

        lookup.update(self.owner.template_lookup(name))

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Loading template for `{name}` with lookup:")
            LOG.debug("%s", json.dumps(lookup, indent=2, default=str))

        tried = []
        for provider in self.templates_providers:
            template = provider.template(name, lookup)
            if template is not None:
                self._template_cache[name] = template
                return

            tried.append(provider)

        LOG.warning(f"Could not find template for `{name}` in {tried}")
        LOG.warning(f"Loading template for `{name}` with lookup:")
        LOG.warning("%s", json.dumps(lookup, indent=2, default=str))
        return None

    def _grid(self, grid: str | list[float] | tuple[int, int]) -> str:
        """Convert the grid information to a standardised format.

        Parameters
        ----------
        grid : Union[str, list of float, tuple of int]
            The grid information.

        Returns
        -------
        str
            The standardised grid format.
        """

        if isinstance(grid, str):
            return grid.upper()

        if isinstance(grid, (tuple, list)) and len(grid) == 2:
            if grid[0] == grid[1]:
                return grid[0]
            return f"{grid[0]}x{grid[1]}"

        return grid
