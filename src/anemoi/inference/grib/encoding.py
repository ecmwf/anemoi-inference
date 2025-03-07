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

LOG = logging.getLogger(__name__)


GRIB1_ONLY = []

GRIB2_ONLY = ["typeOfGeneratingProcess"]


ORDERING = (
    "edition",
    "typeOfLevel",
    "stepType",
    "productDefinitionTemplateNumber",
    "eps",
    "number",
)

ORDERING = {k: i for i, k in enumerate(ORDERING)}


def _ordering(item):
    return ORDERING.get(item[0], 999)


def _param(param):
    try:
        int(param)
        return "paramId"
    except ValueError:
        try:
            float(param)
            return "param"
        except ValueError:
            return "shortName"


def _step_in_hours(step):
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
    result,
    template,
    variable,
    step,
    previous_step,
    start_steps,
    edition,
    ensemble,
):
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
    values,
    template,
    variable,
    ensemble,
    param,
    date,
    time,
    step,
    previous_step,
    start_steps,
    keys,
    grib1_keys={},
    grib2_keys={},
):
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

    result["eps"] = 1 if ensemble else 0

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


def check_encoding(handle, keys, first=True):
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


def encode_message(*, values, template, metadata, check_nans=False, missing_value=9999):
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

    def __init__(self, path, split_output=False):
        self._files = {}
        self.filename = path

        if split_output:
            self.split_output = re.findall(r"\{(.*?)\}", self.filename)
        else:
            self.split_output = None

    def close(self):
        for f in self._files.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.close()

    def write(
        self,
        *,
        values,
        template,
        metadata,
        check_nans=False,
        missing_value=9999,
    ):
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

    def target(self, handle):

        if self.split_output:
            path = self.filename.format(**{k: handle.get(k) for k in self.split_output})
        else:
            path = self.filename

        if path not in self._files:
            self._files[path] = open(path, "wb")

        return self._files[path], path
