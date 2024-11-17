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
        self,
        context,
        *,
        path,
        allow_nans=False,
        encoding=None,
        archive_requests=None,
        check_encoding=False,
        templates=None,
        **kwargs,
    ):
        super().__init__(context, allow_nans=allow_nans, encoding=encoding, templates=templates)
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
            import eccodes

            LOG.error("Error writing message to %s", self.path)
            LOG.error("eccodes: %s", eccodes.__version__)
            LOG.error("Exception: %s", e)
            if message is not None and np.isnan(message.data).any():
                LOG.error("Message contains NaNs (%s, %s) (allow_nans=%s)", keys, template, self.allow_nans)
            raise

    def collect_archive_requests(self, written, template, **keys):

        if not self.archive_requests and not self.check_encoding:
            return

        handle, path = written

        mars = handle.as_namespace("mars")

        if self.check_encoding:
            self._check_encoding(handle, keys)

        if self.archive_requests:
            self.archiving[path].add(mars)

    def close(self):
        self.output.close()

        if not self.archive_requests:
            return

        path = self.archive_requests["path"]
        extra = self.archive_requests.get("extra", {})
        patch = self.archive_requests.get("patch", {})

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

            json.dump(requests, f, indent=4)

    def _check_encoding(self, handle, keys):

        def same(w, v, k):
            if type(v) is type(w):
                return v == w
            return str(w) == str(v)

        mismatches = {}
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

            if not same(w, v, k):
                mismatches[k] = (w, v)

        if mismatches:
            raise ValueError(f"GRIB field could not be encoded. Mismatches={mismatches}")
