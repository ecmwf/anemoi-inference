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
from collections import defaultdict

import earthkit.data as ekd
import numpy as np

from ..decorators import main_argument
from ..grib.encoding import check_encoding
from . import output_registry
from .grib import GribOutput

LOG = logging.getLogger(__name__)

# There is a bug with hindcasts, where these keys are not added to the 'mars' namespace
MARS_MAYBE_MISSING_KEYS = (
    "number",
    "step",
    "time",
    "date",
    "hdate",
    "type",
    "stream",
    "expver",
    "class",
    "levtype",
    "levelist",
    "param",
)


def _is_valid(mars, keys):
    if "number" in keys and "number" not in mars:
        LOG.warning("`number` is missing from mars namespace")
        return False

    if "referenceDate" in keys and "hdate" not in mars:
        LOG.warning("`hdate` is missing from mars namespace")
        return False

    return True


class ArchiveCollector:
    """Collects archive requests"""

    UNIQUE = {"date", "hdate", "time", "referenceDate", "type", "stream", "expver"}

    def __init__(self) -> None:
        self.expect = 0
        self._request = defaultdict(set)

    def add(self, field):
        self.expect += 1
        for k, v in field.items():
            self._request[k].add(str(v))
            if k in self.UNIQUE:
                if len(self._request[k]) > 1:
                    raise ValueError(f"Field {field} has different values for {k}: {self._request[k]}")

    @property
    def request(self):
        return {k: sorted(v) for k, v in self._request.items()}


@output_registry.register("grib")
@main_argument("path")
class GribFileOutput(GribOutput):
    """Handles grib files"""

    def __init__(
        self,
        context,
        *,
        path,
        encoding=None,
        archive_requests=None,
        check_encoding=True,
        templates=None,
        grib1_keys=None,
        grib2_keys=None,
        modifiers=None,
        **kwargs,
    ):
        super().__init__(
            context,
            encoding=encoding,
            templates=templates,
            grib1_keys=grib1_keys,
            grib2_keys=grib2_keys,
            modifiers=modifiers,
        )
        self.path = path
        self.output = ekd.new_grib_output(self.path, split_output=True, **kwargs)
        self.archiving = defaultdict(ArchiveCollector)
        self.archive_requests = archive_requests
        self.check_encoding = check_encoding
        self._namespace_bug_fix = False

    def __repr__(self):
        return f"GribFileOutput({self.path})"

    def write_message(self, message, template, **keys):
        # Make sure `name` is not in the keys, otherwise grib_encoding will fail
        if template is not None and template.metadata("name", default=None) is not None:
            # We cannot clear the metadata...
            class Dummy:
                def __init__(self, template):
                    self.template = template
                    self.handle = template.handle

                def __repr__(self):
                    return f"Dummy({self.template})"

            template = Dummy(template)

        # LOG.info("Writing message to %s %s", template, keys)
        try:
            self.collect_archive_requests(
                self.output.write(
                    message,
                    template=template,
                    check_nans=self.context.allow_nans,
                    **keys,
                ),
                template,
                **keys,
            )
        except Exception as e:
            import eccodes

            LOG.error("Error writing message to %s", self.path)
            LOG.error("eccodes: %s", eccodes.__version__)
            LOG.error("Template: %s, Keys: %s", template, keys)
            LOG.error("Exception: %s", e)
            if message is not None and np.isnan(message.data).any():
                LOG.error("Message contains NaNs (%s, %s) (allow_nans=%s)", keys, template, self.context.allow_nans)
            raise

    def collect_archive_requests(self, written, template, **keys):

        if not self.archive_requests and not self.check_encoding:
            return

        handle, path = written

        while True:

            if self._namespace_bug_fix:
                import eccodes
                from earthkit.data.readers.grib.codes import GribCodesHandle

                handle = GribCodesHandle(eccodes.codes_clone(handle._handle), None, None)

            mars = {k: v for k, v in handle.items("mars")}

            if _is_valid(mars, keys):
                break

            if self._namespace_bug_fix:
                raise ValueError("Namespace bug: %s" % mars)

            # Try again with the namespace bug
            LOG.warning("Namespace bug detected, trying again")
            self._namespace_bug_fix = True

        if self.check_encoding:
            check_encoding(handle, keys)

        if self.archive_requests:
            self.archiving[path].add(mars)

    def close(self):
        self.output.close()

        if not self.archive_requests:
            return

        path = self.archive_requests["path"]
        extra = self.archive_requests.get("extra", {})
        patch = self.archive_requests.get("patch", {})
        indent = self.archive_requests.get("indent", None)

        def _patch(r):
            if self.context.config.use_grib_paramid:
                from anemoi.utils.grib import shortname_to_paramid

                param = r.get("param", [])
                if not isinstance(param, list):
                    param = [param]
                param = [shortname_to_paramid(p) for p in param]
                r["param"] = param

            for k, v in patch.items():
                if v is None:
                    r.pop(k, None)
                else:
                    r[k] = v

            return r

        with open(self.archive_requests["path"], "w") as f:
            requests = []

            for path, archive in self.archiving.items():
                assert path is not None, "Path is None"
                request = dict(expect=archive.expect)
                request["source"] = path
                request.update(_patch(archive.request))
                request.update(extra)
                requests.append(request)

            json.dump(requests, f, indent=indent)
            f.write("\n")
