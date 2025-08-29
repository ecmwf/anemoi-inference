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
from io import IOBase
from typing import Any

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.types import DataRequest
from anemoi.inference.types import FloatArray
from anemoi.inference.types import ProcessorConfig

from ..decorators import main_argument
from ..grib.encoding import GribWriter
from ..grib.encoding import check_encoding
from . import output_registry
from .grib import BaseGribOutput

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


def _fix(mars: dict[str, Any], keys: dict[str, Any]) -> None:
    """Check if the mars dictionary contains valid keys and fix it.

    Parameters
    ----------
    mars : Dict[str, Any]
        The mars dictionary.
    keys : Dict[str, Any]
        The keys dictionary.
    """
    if "number" in keys and "number" not in mars:
        LOG.debug(f"`number` is missing from mars namespace, setting it to {keys['number']}")
        mars["number"] = keys["number"]

    if "referenceDate" in keys and "hdate" not in mars:
        LOG.debug(f"`hdate` is missing from mars namespace, setting it to {keys['referenceDate']}")
        mars["hdate"] = keys["referenceDate"]

    if "startStep" in keys and "endStep" in keys and keys.get("stepType") != "accum":
        if mars.get("step") != f"{keys['startStep']}-{keys['endStep']}":
            LOG.debug(
                f"{keys.get('stepType')} `step={mars.get('step')}` is not a range,  setting it to {keys['startStep']}-{keys['endStep']}."
            )
            mars["step"] = f"{keys['startStep']}-{keys['endStep']}"


class ArchiveCollector:
    """Collects archive requests."""

    UNIQUE = {"date", "hdate", "time", "referenceDate", "type", "stream", "expver"}

    def __init__(self) -> None:
        self.expect = 0
        self._request = defaultdict(set)

    def add(self, field: dict[str, Any]) -> None:
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


class GribIoOutput(BaseGribOutput):
    """Output class for grib io.

    This class handles writing grib and collecting archive requests.
    It extends the BaseGribOutput class and implements the write_message method.
    """

    def __init__(
        self,
        context: Context,
        *,
        out: str | IOBase,
        post_processors: list[ProcessorConfig] | None = None,
        encoding: dict[str, Any] | None = None,
        archive_requests: dict[str, Any] | None = None,
        check_encoding: bool = True,
        templates: list[str] | str | None = None,
        grib1_keys: dict[str, Any] | None = None,
        grib2_keys: dict[str, Any] | None = None,
        modifiers: list[str] | None = None,
        variables: list[str] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        split_output: bool = True,
    ) -> None:
        """Initialize the GribIOOutput.

        Parameters
        ----------
        context : Context
            The context.
        out : Union[str, IOBase]
            Path or file-like object to write the grib data to.
            If a string, it should be a file path.
            If a file-like object, it should be opened in binary write mode.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
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
            Cannot be `True` if `out` is a file-like object.
        """
        super().__init__(
            context,
            post_processors,
            encoding=encoding,
            templates=templates,
            grib1_keys=grib1_keys,
            grib2_keys=grib2_keys,
            modifiers=modifiers,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
            variables=variables,
        )
        self.out = out
        self.output = GribWriter(self.out, split_output)
        self.archiving = defaultdict(ArchiveCollector)
        self.archive_requests = archive_requests
        self.check_encoding = check_encoding
        self._namespace_bug_fix = False

    def __repr__(self) -> str:
        """Return a string representation of the GribIOOutput object."""
        return f"{type(self).__name__ }({self.out})"

    def write_message(self, message: FloatArray, template: ekd.Field, **keys: dict[str, Any]) -> None:
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

            LOG.error("Error writing message to %s", self.out)
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

        mars = {k: v for k, v in handle.items("mars")}
        _fix(mars, keys)

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


@output_registry.register("grib")
@main_argument("path")
class GribFileOutput(GribIoOutput):
    """Handles grib files."""

    def __init__(
        self,
        context: Context,
        *,
        path: str,
        post_processors: list[ProcessorConfig] | None = None,
        encoding: dict[str, Any] | None = None,
        archive_requests: dict[str, Any] | None = None,
        check_encoding: bool = True,
        templates: list[str] | str | None = None,
        grib1_keys: dict[str, Any] | None = None,
        grib2_keys: dict[str, Any] | None = None,
        modifiers: list[str] | None = None,
        variables: list[str] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        split_output: bool = True,
    ) -> None:
        """Initialize the GribFileOutput.

        Parameters
        ----------
        context : Context
            The context.
        path : str
            Path to the grib file to write the data to.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
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
            out=path,
            post_processors=post_processors,
            encoding=encoding,
            archive_requests=archive_requests,
            check_encoding=check_encoding,
            templates=templates,
            grib1_keys=grib1_keys,
            grib2_keys=grib2_keys,
            modifiers=modifiers,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
            variables=variables,
            split_output=split_output,
        )
