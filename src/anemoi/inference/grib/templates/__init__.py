# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import TYPE_CHECKING
from typing import Any

import earthkit.data as ekd
import yaml
from anemoi.utils.registry import Registry

from anemoi.inference.config import Configuration
from anemoi.inference.output import Output

if TYPE_CHECKING:
    from .manager import TemplateManager

LOG = logging.getLogger(__name__)


template_provider_registry = Registry(__name__)


def create_template_provider(owner: Output, config: Configuration) -> "TemplateProvider":
    """Create a template provider from the given configuration.

    Parameters
    ----------
    owner : Output
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

    def __init__(self, manager: "TemplateManager") -> None:
        """Initialize the template provider.

        Parameters
        ----------
        manager : TemplateManager
            The manager for the template provider.
        """
        self.manager = manager

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def template(self, variable: str, lookup: dict[str, Any], **kwargs) -> ekd.Field | None:
        """Get the template for the given variable and lookup.

        Parameters
        ----------
        variable : str
            The variable to get the template for.
        lookup : Dict[str, Any]
            The lookup dictionary.
        kwargs
            Extra arguments for specific template providers.

        Returns
        -------
        ekd.Field | None
            The template field.
        """
        raise NotImplementedError()


class IndexTemplateProvider(TemplateProvider):
    """Template provider based on an index file."""

    def __init__(self, manager: "TemplateManager", index: str | list) -> None:
        """Initialize the index template provider.

        Parameters
        ----------
        manager : TemplateManager
            The manager for the template provider.
        index_path : str | list
            The path to the index.yaml file, or its contents directly as a list.
        """
        super().__init__(manager)
        self.index_path = index

        if isinstance(index, str):
            with open(index) as f:
                self.templates = yaml.safe_load(f)
        else:
            self.templates = index

        if not isinstance(self.templates, list):
            raise ValueError(f"Invalid index, must be a list. Got {self.templates}")

        # TODO: use pydantic
        for template in self.templates:
            if not isinstance(template, list):
                raise ValueError(f"Invalid template index element, must be a list. Got {template}")
            if len(template) != 2:
                raise ValueError(
                    f"Expected template index to be a 2-elements list as `[matching filter, grib file]`. Got {template}."
                )

            match, grib = template
            if not isinstance(match, dict):
                raise ValueError(f"Invalid match in index element, must be a dict: {match}")

            if not isinstance(grib, str):
                raise ValueError(f"Invalid grib in index element, must be a string: {grib}")

    def template(self, variable: str, lookup: dict[str, Any], **kwargs) -> ekd.Field | None:
        def _as_list(value: Any) -> list[Any]:
            if not isinstance(value, list):
                return [value]
            return value

        for template in self.templates:
            match, grib = template
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(f"Matching {match} -> {[(lookup.get(k), _as_list(v)) for k, v in match.items()]}")

            if all(lookup.get(k) in _as_list(v) for k, v in match.items()):
                return self.load_template(grib, lookup)

        return None

    def load_template(self, grib: str, lookup: dict[str, Any]) -> ekd.Field | None:
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
