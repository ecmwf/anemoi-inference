# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

LOG = logging.getLogger(__name__)


def grib_keys(*, values, template, accumulation, param, date, time, step, type, stream, keys):
    result = keys.copy()

    edition = keys.get("edition")
    if edition is None and template is not None:
        edition = template.metadata("edition")
        # centre = template.metadata("centre")

    result["edition"] = edition if edition is not None else 2

    if param is not None:
        result.setdefault("param", param)

    if type is not None:
        result.setdefault("type", type)

    if stream is not None:
        result.setdefault("stream", stream)

    if date is not None:
        result["date"] = date

    if time is not None:
        result.setdefault("time", time)

    # 0: instantaneous, deterministic
    # 1: instantaneous, ensemble
    # 8: time processed, deterministic
    # 11: time processed, ensemble

    if accumulation:
        result["startStep"] = 0
        result["endStep"] = step
        result["stepType"] = "accum"

        if edition == 2:
            result["productDefinitionTemplateNumber"] = 8
            if result.get("type") in ("pf", "cf"):
                result["productDefinitionTemplateNumber"] = 11

    else:
        result["step"] = step
        if edition == 2:
            result["productDefinitionTemplateNumber"] = 0
            if result.get("type") in ("pf", "cf"):
                result["productDefinitionTemplateNumber"] = 1

    return result


def check_encoding(handle, keys):
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
