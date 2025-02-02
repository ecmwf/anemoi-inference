# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

LOG = logging.getLogger(__name__)


GRIB1_ONLY = []

GRIB2_ONLY = ["typeOfGeneratingProcess"]


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


def encode_time_processing(*, result, template, variable, step, previous_step, edition, ensemble):
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

    start = _step_in_hours(previous_step)
    end = _step_in_hours(step)

    # if variable.is_accumulation:
    if start > 0:
        result["stepRange"] = "%d-%d" % (start, end)
    else:
        result["stepRange"] = end
    # else:
    # result["startStep"] = start
    # result["endStep"] = end
    result["stepType"] = STEP_TYPE[variable.time_processing]

    if edition == 1:
        return

    if ensemble:
        result["productDefinitionTemplateNumber"] = 11
    else:
        result["productDefinitionTemplateNumber"] = 8


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
    keys,
    quiet,
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

    if date is not None:
        result["date"] = date

    if time is not None:
        result.setdefault("time", time)

    encode_time_processing(
        result=result,
        template=template,
        variable=variable,
        step=step,
        previous_step=previous_step,
        edition=edition,
        ensemble=ensemble,
    )

    for k, v in variable.grib_keys.items():
        if k not in ("domain", "type", "stream", "expver", "class", "param", "number", "step", "date", "time"):
            if k == "levtype" and v in ("sfc", "o2d"):
                continue
            result.setdefault(k, v)

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
