# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from typing import Any

import earthkit.data as ekd

from . import IndexTemplateProvider
from . import template_provider_registry

LOG = logging.getLogger(__name__)


@template_provider_registry.register("samples")
class SamplesTemplates(IndexTemplateProvider):
    """Class to provide GRIB templates from sample files."""

    def load_template(self, grib: str, lookup: dict[str, Any]) -> ekd.Field | None:
        """Load a GRIB template based on the provided lookup dictionary.

        Parameters
        ----------
        grib : str
            The GRIB template string.
        lookup : Dict[str, Any]
            The lookup dictionary to format the GRIB template string.

        Returns
        -------
        Optional[ekd.Field]
            The loaded GRIB template field, or None if the template is not found.
        """
        template = grib.format(**lookup)
        if not os.path.exists(template):
            LOG.warning(f"Template not found: {template}")
            return None

        return ekd.from_source("file", template)[0]
