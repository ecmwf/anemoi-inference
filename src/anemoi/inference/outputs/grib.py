# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from abc import abstractmethod

from ..output import Output

LOG = logging.getLogger(__name__)


class GribOutput(Output):
    """
    Handles grib
    """

    def __init__(self, context, *, allow_nans=False, encoding=None):
        super().__init__(context)
        self._first = True
        self.typed_variables = self.checkpoint.typed_variables
        self.allow_nans = allow_nans
        self.quiet = set()
        self.encoding = encoding if encoding is not None else {}
        self.edition = self.encoding.get("edition")

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

        reference_date = self.context.reference_date
        date = state["date"]

        step = date - reference_date
        step = step.total_seconds() / 3600
        assert int(step) == step, step
        step = int(step)

        if "_grib_templates_for_output" not in state:
            if "_grib_templates_for_output" not in self.quiet:
                self.quiet.add("_grib_templates_for_output")
                LOG.warning("Input is not GRIB.")

        templates = state.get("_grib_templates_for_output", {})

        for name, value in state["fields"].items():
            keys = {}

            variable = self.typed_variables[name]
            if variable.is_accumulation:
                self.set_accumulation(keys, 0, step)

            def _clostest_template(name):
                best = None, None
                best_similarity = 0
                md1 = self.typed_variables[name]
                for name2, template in templates.items():
                    md2 = self.typed_variables[name2]
                    similarity = md1.similarity(md2)
                    if similarity > best_similarity:
                        best = template, name2
                        best_similarity = similarity
                return best

            template = templates.get(name)
            if template is None:
                if name not in self.quiet:
                    LOG.warning("No GRIB template found for `%s`. This may lead to unexpected results.", name)

                grib_keys = variable.grib_keys.copy()
                for key in ("class", "type", "stream", "expver", "date", "time", "step"):
                    grib_keys.pop(key, None)

                template, name2 = _clostest_template(name)

                if name not in self.quiet:
                    if name2 is not None:
                        LOG.warning("Using template for `%s` with keys %s", name2, grib_keys)
                    else:
                        LOG.warning("Using %s", grib_keys)
                    self.quiet.add(name)

                keys.update(grib_keys)

            keys.update(
                date=reference_date.strftime("%Y-%m-%d"),
                time=reference_date.hour,
                step=step,
            )

            self.set_forecast(keys, reference_date, step)
            self.set_other_keys(keys, variable)

            if self.edition is not None:
                keys["edition"] = self.edition

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug("Encoding GRIB %s\n%s", template, json.dumps(keys, indent=4))

            try:
                self.write_message(value, template=template, **keys)
            except Exception as e:
                LOG.error("Error writing field %s", name)
                LOG.error("Template: %s", template)
                LOG.error("Keys:\n%s", json.dumps(keys, indent=4))
                raise e

    @abstractmethod
    def write_message(self, message, *args, **kwargs):
        pass

    def set_forecast(self, keys, reference_date, step):
        keys["date"] = reference_date.strftime("%Y-%m-%d")
        keys["time"] = reference_date.hour
        keys["step"] = step
        keys["type"] = "fc"

        if self.edition == 2:
            keys["typeOfProcessedData"] = 1

        grib_keys = self.encoding.get("forecast", {})
        keys.update(grib_keys)

    def set_accumulation(self, keys, start, end):
        keys["startStep"] = start
        keys["endStep"] = end
        keys["stepType"] = "accum"

        if self.edition == 2:
            keys["typeOfStatisticalProcessing"] = 1

        grib_keys = self.encoding.get("accumulation", {})
        keys.update(grib_keys)

    def set_other_keys(self, keys, variable):
        grib_keys = {k: v for k, v in self.encoding.items() if not isinstance(v, (dict, list))}
        keys.update(grib_keys)

        per_variable = self.encoding.get("per_variable", {})
        per_variable = per_variable.get(variable.name, {})
        keys.update(per_variable)
