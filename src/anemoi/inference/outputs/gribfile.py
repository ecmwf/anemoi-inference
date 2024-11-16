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

from . import output_registry
from .grib import GribOutput

LOG = logging.getLogger(__name__)


class ArchiveCollector:
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
class GribFileOutput(GribOutput):
    """Handles grib files"""

    def __init__(
        self, context, *, path, allow_nans=False, encoding=None, archive_requests=None, check_encoding=False, **kwargs
    ):
        super().__init__(context, allow_nans=allow_nans, encoding=encoding)
        self.path = path
        self.output = ekd.new_grib_output(self.path, split_output=True, **kwargs)
        self.archiving = defaultdict(ArchiveCollector)
        self.archive_requests = archive_requests
        self.check_encoding = check_encoding

    def __repr__(self):
        return f"GribFileOutput({self.path})"

    def write_message(self, message, template, **keys):
        try:
            self.collect_archive_requests(
                self.output.write(
                    message,
                    template=template,
                    check_nans=self.allow_nans,
                    **keys,
                ),
                template,
                **keys,
            )
        except Exception as e:
            LOG.error("Error writing message to %s: %s", self.path, e)
            if message is not None and np.isnan(message.data).any():
                LOG.error("Message contains NaNs (%s, %s)", keys, template)
            raise e

    def collect_archive_requests(self, written, template, **keys):

        if not self.archive_requests and not self.check_encoding:
            return

        handle, path = written

        mars = handle.as_namespace("mars")

        if self.check_encoding:
            ok = True
            for k, v in keys.items():
                if k == "param":
                    try:
                        int(v)
                        k = "paramId"
                    except ValueError:
                        try:
                            float(v)
                            k = "param"
                        except ValueError:
                            k = "shortName"

                if k == "date":
                    v = int(str(v).replace("-", ""))

                if k == "time":
                    v = int(v)
                    if v < 100:
                        v *= 100

                w = handle.get(k)

                if w != v:
                    LOG.error("Field %s has different value for %s: %s != %s", path, k, w, v)
                    ok = False

            if not ok:
                import eccodes

                raise ValueError(f"GRIB field could not be encoded correctly (eccodes {eccodes.__version__})")

        if self.archive_requests:
            self.archiving[path].add(mars)

    def close(self):
        self.output.close()

        if not self.archive_requests:
            return

        path = self.archive_requests["path"]
        extra = self.archive_requests.get("extra", {})

        with open(self.archive_requests["path"], "w") as f:
            requests = []

            for path, archive in self.archiving.items():
                assert path is not None, "Path is None"
                request = dict(expect=archive.expect)
                request["source"] = path
                request.update(archive.request)
                request.update(extra)
                requests.append(request)

            json.dump(requests, f, indent=4)
