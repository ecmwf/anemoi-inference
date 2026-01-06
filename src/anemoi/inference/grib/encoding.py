# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from datetime import datetime
from datetime import timedelta
from io import IOBase
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Hashable

import earthkit.data as ekd
from earthkit.data.utils.dates import to_timedelta

from anemoi.inference.types import FloatArray
from anemoi.inference.utils.templating import render_template

if TYPE_CHECKING:
    from anemoi.transform.variables import Variable

LOG = logging.getLogger(__name__)


GRIB1_ONLY: list[str] = []

GRIB2_ONLY: list[str] = ["typeOfGeneratingProcess"]


ORDERING = (
    "edition",
    "typeOfLevel",
    "stepType",
    "productDefinitionTemplateNumber",
    "eps",
    "number",
)

ORDERING = {k: i for i, k in enumerate(ORDERING)}


def _ordering(item: tuple) -> int:
    """Get the ordering index for a given item.

    Parameters
    ----------
    item : tuple
        The item to get the ordering index for.

    Returns
    -------
    int
        The ordering index.
    """
    return ORDERING.get(item[0], 999)


def _param(param: Any) -> str:
    """Determine the parameter type based on its value.

    Parameters
    ----------
    param : str
        The parameter value.

    Returns
    -------
    str
        The parameter type.
    """
    try:
        int(param)
        return "paramId"
    except ValueError:
        try:
            float(param)
            return "param"
        except ValueError:
            return "shortName"


def _step_in_hours(step: timedelta) -> int:
    """Convert a step to hours.

    Parameters
    ----------
    step : timedelta
        The step to convert.

    Returns
    -------
    int
        The step in hours.
    """
    step = step.total_seconds() / 3600
    assert int(step) == step
    return int(step)


STEP_TYPE = {
    "accumulation": "accum",
    "average": "avg",
    "maximum": "max",
    "minimum": "min",
    "instantaneous": None,
}


def encode_time_processing(
    *,
    result: dict[str, Any],
    template: ekd.Field,
    variable: "Variable",
    date: datetime,
    step: timedelta,
    previous_step: timedelta | None,
    start_steps: dict[str, timedelta],
    edition: int,
    ensemble: bool,
) -> None:
    """Encode time processing information into the result dictionary.

    Parameters
    ----------
    result : dict[str, Any]
        The result dictionary to update.
    template : ekd.Field
        The template field.
    variable : Variable
        The variable containing time processing information.
    date : datetime
        The date and time.
    step : timedelta
        The current step.
    previous_step : timedelta | None
        The previous step.
    start_steps : dict[str, timedelta]
        The start steps dictionary.
    edition : int
        The GRIB edition.
    ensemble : bool
        Whether the data is part of an ensemble.
    """
    assert edition in (1, 2)

    if variable.time_processing is None:
        result["step"] = _step_in_hours(step)
        result["stepType"] = "instant"
        return

    if previous_step is None:
        previous_step = step

    if period := getattr(variable, "period", None):
        start = step - period
<<<<<<< HEAD
        if start < as_timedelta(0):
            assert result["time"] in (
                0,
                600,
                1200,
                1800,
            ), f"Unexpected time {result['time']} for variable {variable.name} with period {period}"
            date = datetime(
                year=result["date"] // 10000,
                month=(result["date"] // 100) % 100,
                day=result["date"] % 100,
                hour=result["time"] // 100,
                minute=result["time"] % 100,
=======
        if start < to_timedelta(0):
            LOG.warning(
                f"Negative start step {_step_in_hours(start)} for variable {variable.name} with period {_step_in_hours(period)} at output step {_step_in_hours(step)}"
>>>>>>> 291e21c56f7610ee30689a53b201e1bc339f9c14
            )

            date += start
            step -= start
<<<<<<< HEAD

            LOG.warning(
                f"Start step {start} is negative for variable {variable.name} with period {period}, setting reference date {date.isoformat()}."
            )

            start = as_timedelta(0)
            result["date"] = date.year * 10000 + date.month * 100 + date.day
=======
            start = to_timedelta(0)

            result["date"] = int(date.strftime("%Y%m%d"))
>>>>>>> 291e21c56f7610ee30689a53b201e1bc339f9c14
            result["time"] = date.hour * 100 + date.minute
    else:
        # backwards compatibility with old transform or if period is missing from the metadata
        start = previous_step
        warnings.warn(
            f"{variable.name} {variable.time_processing} does not have a period set, using previous_step as start={_step_in_hours(start)}."
        )

    # give post-processors a chance to modify the start step
    start = _step_in_hours(start_steps.get(variable.name, start))
    end = _step_in_hours(step)

    result["startStep"] = start
    result["endStep"] = end
    result["stepType"] = STEP_TYPE[variable.time_processing]

    if edition == 1:
        return

    if ensemble:
        result["productDefinitionTemplateNumber"] = 11
    else:
        result["productDefinitionTemplateNumber"] = 8


LEVTYPES = {
    "pl": "isobaricInhPa",
    "ml": "hybrid",
    "pt": "theta",
    "pv": "potentialVorticity",
}


def grib_keys(
    *,
    values: FloatArray,
    template: ekd.Field,
    variable: "Variable",
    ensemble: bool,
    param: int | float | str | None,
    date: datetime,
    step: timedelta,
    previous_step: timedelta | None,
    start_steps: dict[str, timedelta],
    keys: dict[str, Any],
    grib1_keys: dict[int | float | str, dict[str, Any]] = {},
    grib2_keys: dict[int | float | str, dict[str, Any]] = {},
) -> dict[str, Any]:
    """Generate GRIB keys for encoding.

    Parameters
    ----------
    values : FloatArray
        The values to encode.
    template : ekd.Field
        The template to use.
    variable : Variable
        The variable containing GRIB keys.
    ensemble : bool
        Whether the data is part of an ensemble.
    param : int | float | str | None
        The parameter value.
    date : datetime
        The date and time.
    step : Any
        The current step.
    previous_step : timedelta | None
        The previous step.
    start_steps : dict[str, timedelta]
        The start steps dictionary.
    keys : dict[str, Any]
        The initial keys dictionary.
    grib1_keys : dict[int | float | str, dict[str, Any]], optional
        Additional GRIB1 keys.
    grib2_keys : dict[int | float | str, dict[str, Any]], optional
        Additional GRIB2 keys.

    Returns
    -------
    dict[str, Any]
        The generated GRIB keys.
    """
    result = keys.copy()

    edition = keys.get("edition")
    if edition is None and template is not None:
        edition = template.metadata("edition")

    if edition is None:
        edition = 1

    if edition == 1:
        for k in GRIB2_ONLY:
            result.pop(k, None)

    if edition == 2:
        for k in GRIB1_ONLY:
            result.pop(k, None)

    result["edition"] = edition

    if param is not None:
        result.setdefault(_param(param), param)

        if edition == 1:
            result.update(grib1_keys.get(param, {}))

        if edition == 2:
            result.update(grib2_keys.get(param, {}))

    result.setdefault("type", "fc")

    if result.get("type") in ("an", "fc"):
        # For organisations that do not use type
        result.setdefault("dataType", result.pop("type"))

    result["date"] = int(date.strftime("%Y%m%d"))
    result["time"] = date.hour * 100

    encode_time_processing(
        result=result,
        template=template,
        variable=variable,
        date=date,
        step=step,
        previous_step=previous_step,
        start_steps=start_steps,
        edition=edition,
        ensemble=ensemble,
    )

    # 1 if local definition is present, like for ECMWF GRIBs
    if template is not None:
        local_use_present = template.metadata("localUsePresent", default=0)
    else:
        local_use_present = 0

    for k, v in variable.grib_keys.items():
        if k not in (
            "domain",
            "type",
            "expver",
            "class",
            "param",
            "number",
            "step",
            "date",
            "hdate",
            "time",
            "timespan",
            "valid_datetime",
            "variable",
        ):
            if k == "stream":
                if v in ("oper", "wave") and local_use_present:
                    result.setdefault(k, v)
                continue

            if k == "levtype":
                v = LEVTYPES.get(v)
                if v is None:
                    continue
                k = "typeOfLevel"
            result.setdefault(k, v)

    result = {k: v for k, v in sorted(result.items(), key=_ordering) if v is not None}

    return result


def check_encoding(handle: Any, keys: dict[str, Any], first: bool = True) -> None:
    """Check if the GRIB encoding matches the expected keys.

    Parameters
    ----------
    handle : Any
        The GRIB handle.
    keys : Dict[str, Any]
        The expected keys.
    first : bool, optional
        Whether this is the first check.

    Raises
    ------
    ValueError
        If the GRIB field could not be encoded.
    """

    def same(w: Any, v: Any, k: str) -> bool:
        if type(v) is type(w):
            # Keep `mypy` happy
            equal: bool = v == w
            return equal
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

        if isinstance(v, int):
            w = handle.get_long(k)
        else:
            w = handle.get(k)

        if not same(w, v, k):
            mismatches[k] = f'Expected "{v}" but got "{w}"'

    if mismatches:

        if first:
            import eccodes
            from earthkit.data.readers.grib.codes import GribCodesHandle

            handle = GribCodesHandle(eccodes.codes_clone(handle._handle), None, None)
            return check_encoding(handle, keys, first=False)

        raise ValueError(f"GRIB field could not be encoded. Mismatches={mismatches}")


def encode_message(
    *,
    values: Any | None,
    template: Any,
    metadata: dict[str, Any],
    check_nans: bool = False,
    missing_value: int | float = 9999,
) -> Any:
    """Encode a GRIB message.

    Parameters
    ----------
    values : Optional[Any]
        The values to encode.
    template : Any
        The template to use.
    metadata : Dict[str, Any]
        The metadata for the GRIB message.
    check_nans : bool, optional
        Whether to check for NaNs in the values.
    missing_value : Union[int, float], optional
        The value to use for missing data.

    Returns
    -------
    Any
        The encoded GRIB handle.
    """
    metadata = metadata.copy()  # avoid modifying the original metadata
    handle = template.handle.clone()

    if check_nans and values is not None:
        import numpy as np

        if np.isnan(values).any():
            # missing_value = np.finfo(values.dtype).max
            missing_value = missing_value
            values = np.nan_to_num(values, nan=missing_value)
            metadata["missingValue"] = missing_value
            metadata["bitmapPresent"] = 1

    if int(metadata.get("deleteLocalDefinition", 0)):
        for k in ("class", "type", "stream", "expver", "setLocalDefinition"):
            metadata.pop(k, None)

    metadata.setdefault("generatingProcessIdentifier", 255)

    LOG.debug("GribOutput.metadata %s", metadata)

    single = {}
    multiple = {}
    for k, v in metadata.items():
        if isinstance(v, (int, float, str, bool)):
            single[k] = v
        else:
            multiple[k] = v

    try:
        # Try to set all metadata at once
        # This is needed when we set multiple keys that are interdependent
        handle.set_multiple(single)
    except Exception as e:
        LOG.error("Failed to set metadata at once: %s", e)
        # Try again, but one by one
        for k, v in single.items():
            handle.set(k, v)

    for k, v in multiple.items():
        handle.set(k, v)

    if values is not None:
        handle.set_values(values)

    return handle


class GribWriter:
    """Write GRIB messages to one or more files."""

    def __init__(self, out: Path | IOBase, split_output: bool = True) -> None:
        """Initialize the GribWriter.

        Parameters
        ----------
        out : Union[Path, IOBase]
            Path or file-like object to write the grib data to.
            If a string, it should be a file path.
            If a file-like object, it should be opened in binary write mode.
        split_output : bool, optional
            Whether to split the output into multiple files.
        """

        if isinstance(out, IOBase) and split_output:
            raise ValueError("Cannot split output when `out` is a file-like object.")

        self.out = out
        self.split_output = split_output
        self._files: dict[Hashable, IOBase] = {}

    def close(self) -> None:
        """Close all open files."""
        for f in self._files.values():
            f.close()

    def __enter__(self) -> "GribWriter":
        """Enter the runtime context related to this object.

        Returns
        -------
        GribWriter
            The GribWriter instance.
        """
        return self

    def __exit__(self, exc_type: type | None, exc_value: BaseException | None, trace: Any | None) -> None:
        """Exit the runtime context related to this object.

        Parameters
        ----------
        exc_type : Optional[type]
            The exception type.
        exc_value : Optional[BaseException]
            The exception value.
        trace : Optional[Any]
            The traceback object.
        """
        self.close()

    def write(
        self,
        *,
        values: Any | None,
        template: Any,
        metadata: dict[str, Any],
        check_nans: bool = False,
        missing_value: int | float = 9999,
    ) -> tuple:
        """Write a GRIB message to the target file.

        Parameters
        ----------
        values : Optional[Any]
            The values to encode.
        template : Any
            The template to use.
        metadata : Dict[str, Any]
            The metadata for the GRIB message.
        check_nans : bool, optional
            Whether to check for NaNs in the values.
        missing_value : Union[int, float], optional
            The value to use for missing data.

        Returns
        -------
        tuple
            The encoded GRIB handle and the file path.
        """

        while True:
            try:
                handle = encode_message(
                    values=values,
                    check_nans=check_nans,
                    metadata=metadata,
                    template=template,
                    missing_value=missing_value,
                )
                break
            except Exception as e:
                if metadata.get("edition") == 2:
                    raise
                # Try again with edition 2
                LOG.warning("Failed to encode GRIB message with edition 1, retrying with edition 2: %s", e)
                metadata["edition"] = 2

        file, path = self.target(handle)
        handle.write(file)

        return handle, path

    def target(self, handle: Any) -> tuple[IOBase, Path | str]:
        """Determine the target file for the GRIB message.

        Parameters
        ----------
        handle : Any
            The GRIB handle.

        Returns
        -------
        tuple
            The file object and the file path.
        """
        if self.split_output:
            assert not isinstance(self.out, IOBase), "Cannot split output when `out` is a file-like object."
            out = render_template(str(self.out), handle)
        else:
            out = self.out

        if isinstance(out, IOBase):
            # self._files['out'] = out
            return out, "out"

        if out not in self._files:
            self._files[out] = open(out, "wb")

        return self._files[out], out
