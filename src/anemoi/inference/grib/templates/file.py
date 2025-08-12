# (C) Copyright 2024 Anemoi contributors.
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

from . import TemplateProvider
from . import template_provider_registry

LOG = logging.getLogger(__name__)


@template_provider_registry.register("file")
class FileTemplates(TemplateProvider):
    """Template provider using a single GRIB file."""

    def __init__(self, manager: Any, path: str) -> None:
        """Initialize the FileTemplates instance.

        Parameters
        ----------
        manager : Any
            The manager instance.
        path : str
            The path to the GRIB file.
        """
        self.manager = manager
        self.path = path

    def template(self, grib: str, lookup: dict[str, Any]) -> ekd.Field:
        """Retrieve the template from the GRIB file.

        Parameters
        ----------
        grib : str
            The GRIB string.
        lookup : Dict[str, Any]
            The lookup dictionary.

        Returns
        -------
        ekd.Field
            The field from the GRIB file.
        """
        return ekd.from_source("file", self.path)[0]
