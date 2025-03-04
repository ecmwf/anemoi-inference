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
from typing import Dict
from typing import List
from typing import Optional

import earthkit.data as ekd
import yaml
from anemoi.utils.registry import Registry

from anemoi.inference.config import Configuration

LOG = logging.getLogger(__name__)


template_provider_registry = Registry(__name__)


def create_template_provider(owner: Any, config: Configuration) -> "TemplateProvider":
    """Create a template provider from the given configuration.

    Parameters
    ----------
    owner : Any
        The owner of the template provider.
    config : Configuration
        The configuration for the template provider.

    Returns
    -------
    TemplateProvider
        The created template provider.
    """
    return template_provider_registry.from_config(config, owner)


class TemplateProvider:
    """Base class for template providers."""

    def __init__(self, manager: Any) -> None:
        """Initialize the template provider.

        Parameters
        ----------
        manager : Any
            The manager for the template provider.
        """
        self.manager = manager

    def template(self, variable: str, lookup: Dict[str, Any]) -> ekd.Field:
        """Get the template for the given variable and lookup.

        Parameters
        ----------
        variable : str
            The variable to get the template for.
        lookup : Dict[str, Any]
            The lookup dictionary.

        Returns
        -------
        ekd.Field
            The template field.
        """
        raise NotImplementedError()


class IndexTemplateProvider(TemplateProvider):
    """Template provider based on an index file."""

    def __init__(self, manager: Any, index_path: str) -> None:
        """Initialize the index template provider.

        Parameters
        ----------
        manager : Any
            The manager for the template provider.
        index_path : str
            The path to the index file.
        """
        super().__init__(manager)
        self.index_path = index_path

        with open(index_path) as f:
            self.templates = yaml.safe_load(f)

        if not isinstance(self.templates, list):
            raise ValueError("Invalid templates.yaml, must be a list")

        # TODO: use pydantic
        for template in self.templates:
            if not isinstance(template, list):
                raise ValueError(f"Invalid template in templates.yaml, must be a list: {template}")
            if len(template) != 2:
                raise ValueError(f"Invalid template in templates.yaml, must have exactly 2 elements: {template}")

            match, grib = template
            if not isinstance(match, dict):
                raise ValueError(f"Invalid match in templates.yaml, must be a dict: {match}")

            if not isinstance(grib, str):
                raise ValueError(f"Invalid grib in templates.yaml, must be a string: {grib}")

    def template(self, variable: str, lookup: Dict[str, Any]) -> Optional[ekd.Field]:
        """Get the template for the given variable and lookup.

        Parameters
        ----------
        variable : str
            The variable to get the template for.
        lookup : Dict[str, Any]
            The lookup dictionary.

        Returns
        -------
        Optional[ekd.Field]
            The template field if found, otherwise None.
        """

        def _as_list(value: Any) -> List[Any]:
            if not isinstance(value, list):
                return [value]
            return value

        for template in self.templates:
            match, grib = template
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug("%s", [(lookup.get(k), _as_list(v)) for k, v in match.items()])

            if all(lookup.get(k) in _as_list(v) for k, v in match.items()):
                return self.load_template(grib, lookup)

        return None

    def load_template(self, grib: str, lookup: Dict[str, Any]) -> Optional[ekd.Field]:
        """Load the template for the given GRIB and lookup.

        Parameters
        ----------
        grib : str
            The GRIB string.
        lookup : Dict[str, Any]
            The lookup dictionary.

        Returns
        -------
        Optional[ekd.Field]
            The loaded template field if found, otherwise None.
        """
        raise NotImplementedError()
