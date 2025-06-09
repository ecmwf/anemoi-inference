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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import DataRequest
from anemoi.inference.types import FloatArray

from ..decorators import main_argument
from ..grib.encoding import GribWriter
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


def _is_valid(mars: Dict[str, Any], keys: Dict[str, Any]) -> bool:
    """Check if the mars dictionary contains valid keys.

    Parameters
    ----------
    mars : Dict[str, Any]
        The mars dictionary.
    keys : Dict[str, Any]
        The keys dictionary.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if "number" in keys and "number" not in mars:
        LOG.warning("`number` is missing from mars namespace")
        return False

    if "referenceDate" in keys and "hdate" not in mars:
        LOG.warning("`hdate` is missing from mars namespace")
        return False

    return True


class ArchiveCollector:
    """Collects archive requests."""

    UNIQUE = {"date", "hdate", "time", "referenceDate", "type", "stream", "expver"}

    def __init__(self) -> None:
        self.expect = 0
        self._request = defaultdict(set)

    def add(self, field: Dict[str, Any]) -> None:
        """Add a field to the archive request.

        Parameters
        ----------
        field : Dict[str,Any]
            The field dictionary.
        """
        self.expect += 1
        for k, v in field.items():
            self._request[k].add(str(v))
            if k in self.UNIQUE:
                if len(self._request[k]) > 1:
                    raise ValueError(f"Field {field} has different values for {k}: {self._request[k]}")

    @property
    def request(self) -> DataRequest:
        """Get the archive request."""
        return {k: sorted(v) for k, v in self._request.items()}


@output_registry.register("grib")
@main_argument("path")
class GribFileOutput(GribOutput):
    """Handles grib files."""

    def __init__(
        self,
        context: Context,
        *,
        path: str,
        encoding: Optional[Dict[str, Any]] = None,
        archive_requests: Optional[Dict[str, Any]] = None,
        check_encoding: bool = True,
        templates: Optional[Union[List[str], str]] = None,
        grib1_keys: Optional[Dict[str, Any]] = None,
        grib2_keys: Optional[Dict[str, Any]] = None,
        modifiers: Optional[List[str]] = None,
        output_frequency: Optional[int] = None,
        write_initial_state: Optional[bool] = None,
        variables: Optional[List[str]] = None,
        split_output: bool = True,
    ) -> None:
        """Initialize the GribFileOutput.

        Parameters
        ----------
        context : Context
            The context.
        path : str
            The path to save the grib files.
        encoding : dict, optional
            The encoding dictionary, by default None.
        archive_requests : dict, optional
            The archive requests dictionary, by default None.
        check_encoding : bool, optional
            Whether to check encoding, by default True.
        templates : list or str, optional
            The templates list or string, by default None.
        grib1_keys : dict, optional
            The grib1 keys dictionary, by default None.
        grib2_keys : dict, optional
            The grib2 keys dictionary, by default None.
        modifiers : list, optional
            The list of modifiers, by default None.
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        variables : list, optional
            The list of variables, by default None.
        split_output : bool, optional
            Whether to split the output, by default True.
        """
        super().__init__(
            context,
            encoding=encoding,
            templates=templates,
            grib1_keys=grib1_keys,
            grib2_keys=grib2_keys,
            modifiers=modifiers,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
            variables=variables,
        )
        self.path = path
        self.output = GribWriter(self.path, split_output=split_output)
        self.archiving = defaultdict(ArchiveCollector)
        self.archive_requests = archive_requests
        self.check_encoding = check_encoding
        self._namespace_bug_fix = False

    def __repr__(self) -> str:
        """Return a string representation of the GribFileOutput object."""
        return f"GribFileOutput({self.path})"

    def write_message(self, message: FloatArray, template: ekd.Field, **keys: Dict[str, Any]) -> None:
        """Write a message to the grib file.

        Parameters
        ----------
        message : FloatArray
            The message array.
        template : ekd.Field
            A ekd.Field use as a template for GRIB encoding.
        **keys : Dict[str, Any]
            Additional keys for the message.
        """
        # Make sure `name` is not in the keys, otherwise grib_encoding will fail
        if template is not None and template.metadata("name", default=None) is not None:
            # We cannot clear the metadata...
            class Dummy:
                def __init__(self, template: ekd.Field) -> None:
                    self.template = template
                    self.handle = template.handle

                def __repr__(self) -> str:
                    return f"Dummy({self.template})"

            template = Dummy(template)

        # LOG.info("Writing message to %s %s", template, keys)
        try:
            self.collect_archive_requests(
                self.output.write(
                    values=message,
                    template=template,
                    metadata=keys,
                    check_nans=self.context.allow_nans,
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

    def collect_archive_requests(self, written: tuple, template: object, **keys: Any) -> None:
        """Collect archive requests.

        Parameters
        ----------
        written : tuple
            The written tuple.
        template : object
            The template object.
        **keys : Any
            Additional keys for the archive requests.
        """
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

    def close(self) -> None:
        """Close the grib file."""
        self.output.close()

        if not self.archive_requests:
            return

        path = self.archive_requests["path"]
        extra = self.archive_requests.get("extra", {})
        patch = self.archive_requests.get("patch", {})
        indent = self.archive_requests.get("indent", None)

        def _patch(r: DataRequest) -> DataRequest:
            if self.context.config.use_grib_paramid:
                param = r.get("param", [])
                if not isinstance(param, list):
                    param = [param]

                # Check if we're using param ids already
                try:
                    float(next(iter(param)))
                except ValueError:
                    from anemoi.utils.grib import shortname_to_paramid

                    r["param"] = [shortname_to_paramid(p) for p in param]

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
