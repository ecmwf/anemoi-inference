# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import earthkit.data as ekd

LOG = logging.getLogger(__name__)


GRIB1_ONLY: List[str] = []

GRIB2_ONLY: List[str] = ["typeOfGeneratingProcess"]


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


def _step_in_hours(step: Any) -> int:
    """Convert a step to hours.

    Parameters
    ----------
    step : Any
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
    result: Dict[str, Any],
    template: ekd.Field,
    variable: Any,
    step: Any,
    previous_step: Optional[Any],
    start_steps: Dict[Any, Any],
    edition: int,
    ensemble: bool,
) -> None:
    """Encode time processing information into the result dictionary.

    Parameters
    ----------
    result : Dict[str, Any]
        The result dictionary to update.
    template : ekd.Field
        The template field.
    variable : Any
        The variable containing time processing information.
    step : Any
        The current step.
    previous_step : Optional[Any]
        The previous step.
    start_steps : Dict[Any, Any]
        The start steps dictionary.
    edition : int
        The GRIB edition.
    ensemble : bool
        Whether the data is part of an ensemble.
    """
    assert edition in (1, 2)

    if variable.time_processing is None:
        result["step"] = _step_in_hours(step)
        # result["startStep"] = _step_in_hours(step)
        # result["endStep"] = _step_in_hours(step)
        result["stepType"] = "instant"
        return

    if previous_step is None:
        if not variable.is_accumulation:
            LOG.warning(f"No previous step available for time processing `{variable.time_processing}` for `{variable}`")
        previous_step = step

    start = _step_in_hours(start_steps.get(variable, previous_step))
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
    values: Any,
    template: Any,
    variable: Any,
    ensemble: bool,
    param: Optional[Union[int, float, str]],
    date: int,
    time: int,
    step: Any,
    previous_step: Optional[Any],
    start_steps: Dict[Any, Any],
    keys: Dict[str, Any],
    grib1_keys: Dict[Union[int, float, str], Dict[str, Any]] = {},
    grib2_keys: Dict[Union[int, float, str], Dict[str, Any]] = {},
) -> Dict[str, Any]:
    """Generate GRIB keys for encoding.

    Parameters
    ----------
    values : Any
        The values to encode.
    template : Any
        The template to use.
    variable : Any
        The variable containing GRIB keys.
    ensemble : bool
        Whether the data is part of an ensemble.
    param : Optional[Union[int, float, str]]
        The parameter value.
    date : int
        The date value.
    time : int
        The time value.
    step : Any
        The current step.
    previous_step : Optional[Any]
        The previous step.
    start_steps : Dict[Any, Any]
        The start steps dictionary.
    keys : Dict[str, Any]
        The initial keys dictionary.
    grib1_keys : Dict[Union[int, float, str], Dict[str, Any]], optional
        Additional GRIB1 keys.
    grib2_keys : Dict[Union[int, float, str], Dict[str, Any]], optional
        Additional GRIB2 keys.

    Returns
    -------
    Dict[str, Any]
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

    result["date"] = date
    result["time"] = time

    encode_time_processing(
        result=result,
        template=template,
        variable=variable,
        step=step,
        previous_step=previous_step,
        start_steps=start_steps,
        edition=edition,
        ensemble=ensemble,
    )

    for k, v in variable.grib_keys.items():
        if k not in ("domain", "type", "stream", "expver", "class", "param", "number", "step", "date", "hdate", "time"):
            if k == "levtype":
                v = LEVTYPES.get(v)
                if v is None:
                    continue
                k = "typeOfLevel"
            result.setdefault(k, v)

    result = {k: v for k, v in sorted(result.items(), key=_ordering) if v is not None}

    return result


def check_encoding(handle: Any, keys: Dict[str, Any], first: bool = True) -> None:
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
            mismatches[k] = 'Expected "{}" but got "{}"'.format(v, w)

    if mismatches:

        if first:
            import eccodes
            from earthkit.data.readers.grib.codes import GribCodesHandle

            handle = GribCodesHandle(eccodes.codes_clone(handle._handle), None, None)
            return check_encoding(handle, keys, first=False)

        raise ValueError(f"GRIB field could not be encoded. Mismatches={mismatches}")


def encode_message(
    *,
    values: Optional[Any],
    template: Any,
    metadata: Dict[str, Any],
    check_nans: bool = False,
    missing_value: Union[int, float] = 9999,
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

    def __init__(self, path: str, split_output: bool = True) -> None:
        """Initialize the GribWriter.

        Parameters
        ----------
        path : str
            The path to the output file.
        split_output : bool, optional
            Whether to split the output into multiple files.
        """
        self._files: Dict[str, Any] = {}
        self.filename = path
        self.split_output = split_output

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

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], trace: Optional[Any]) -> None:
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
        values: Optional[Any],
        template: Any,
        metadata: Dict[str, Any],
        check_nans: bool = False,
        missing_value: Union[int, float] = 9999,
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
        handle = encode_message(
            values=values,
            check_nans=check_nans,
            metadata=metadata,
            template=template,
            missing_value=missing_value,
        )

        file, path = self.target(handle)
        handle.write(file)

        return handle, path

    def target(self, handle: Any) -> tuple:
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
            path = render_template(self.filename, handle)
        else:
            path = self.filename

        if path not in self._files:
            self._files[path] = open(path, "wb")

        return self._files[path], path


_TEMPLATE_EXPRESSION_PATTERN = re.compile(r"\{(.*?)\}")


def render_template(template: str, handle: Dict) -> str:
    """Render a template string with the given keyword arguments.

    Given a template string such as '{dateTime}_{step:03}.grib' and
    the GRIB handle, this function will replace the expressions in the
    template with the corresponding values from the handle, formatted
    according to the optional format specifier.

    For example, the template '{dateTime}_{step:03}.grib' with a handle
    containing 'dateTime' as '202501011200' and 'step' as 6 will
    produce '202501011200_006.grib'.

    Parameters
    ----------
    template : str
        The template string to render.
    handle : Dict
        The earthkit.data handle manager.

    Returns
    -------
    str
        The rendered template string.
    """
    expressions = _TEMPLATE_EXPRESSION_PATTERN.findall(template)
    expr_format = [el.split(":") if ":" in el else [el, ""] for el in expressions]
    keys = {k[0]: format(handle.get(k[0]), k[1]) for k in expr_format}
    path = template.format(**keys)
    return path
