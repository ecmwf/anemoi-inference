# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import warnings
from abc import abstractmethod

from . import Output

LOG = logging.getLogger(__name__)


class GribOutput(Output):
    """
    Handles grib
    """

    def __init__(self, runner, *, allow_nans=False):
        super().__init__(runner)
        self._first = True
        self.typed_variables = self.checkpoint.typed_variables
        self.allow_nans = allow_nans
        self.quiet = set()

    def write_initial_state(self, state):
        # We trust the GribInput class to provide the templates
        # matching the input state

        if "_grib_templates_for_output" not in state:
            # We can currently only write grib output if we have a grib input
            raise ValueError("GRIB output only works if the input is GRIB (for now).")

        templates = state["_grib_templates_for_output"]

        for name in state["fields"]:
            self.write_message(None, template=templates[name])

    def write_state(self, state):

        reference_date = state["reference_date"]
        date = state["date"]

        if "_grib_templates_for_output" not in state:
            if "_grib_templates_for_output" not in self.quiet:
                self.quiet.add("_grib_templates_for_output")
                LOG.warning("Input is not GRIB.")

        templates = state.get("_grib_templates_for_output", {})

        for name, value in state["fields"].items():
            variable = self.typed_variables[name]
            if variable.is_accumulation:
                warnings.warn("🚧 TEMPORARY CODE 🚧: accumulations are not supported yet")
                continue

            keys = {}

            template = templates.get(name)
            if template is None:
                if name not in self.quiet:

                    LOG.warning("No GRIB template found for `%s`. This may lead to unexpected results.", name)
                grib_keys = variable.grib_keys.copy()
                for key in ("class", "type", "stream", "expver", "date", "time", "step"):
                    grib_keys.pop(key, None)
                if name not in self.quiet:
                    LOG.warning("Using %s", grib_keys)
                    self.quiet.add(name)
                keys.update(grib_keys)

            keys.update(
                edition=2,
                date=reference_date.strftime("%Y-%m-%d"),
                time=reference_date.hour,
                step=(date - reference_date).total_seconds() // 3600,
                typeOfProcessedData=1,  # Forecast
            )

            try:
                self.write_message(value, template=template, **keys)
            except Exception as e:
                LOG.error("Error writing field %s", name)
                LOG.error("Keys: %s", keys)
                LOG.error("Template: %s", template)
                raise e

    @abstractmethod
    def write_message(self, message, *args, **kwargs):
        pass
