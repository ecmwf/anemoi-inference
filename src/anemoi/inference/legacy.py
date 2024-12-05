# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from collections import defaultdict

LOG = logging.getLogger(__name__)


def warn(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"Using legacy {func.__name__}, please try to patch your weights.")
        return func(*args, **kwargs)

    return wrapper


class LegacyMixin:

    # `self` is a `Metadata` object

    @warn
    def _legacy_variables_metadata(self):

        # Assumes ECMWF data from MARS
        result = {}
        unkowns = []
        for variable in self.variables:
            if variable in (
                "insolation",
                "cos_solar_zenith_angle",
                "cos_julian_day",
                "sin_julian_day",
                "cos_local_time",
                "sin_local_time",
            ):
                result[variable] = dict(computed_forcing=True, constant_in_time=False)
                continue

            if variable in ("cos_latitude", "cos_longitude", "sin_latitude", "sin_longitude"):
                result[variable] = dict(computed_forcing=True, constant_in_time=True)
                continue

            param_level = variable.split("_")
            if len(param_level) == 2:
                # Assumes levtype=pl for now
                levtype = "pl"

                try:
                    levelist = int(param_level[1])
                    mars = dict(param=param_level[0], levelist=levelist, levtype=levtype)
                    result[variable] = dict(mars=mars)
                    continue
                except ValueError:
                    pass

            if variable in (
                "z",
                "lsm",
                "sdor",
                "slor",
                "cl",
                "cvh",
                "cvl",
                "slt",
                "tvh",
                "tvl",
            ):
                mars = dict(param=variable, levtype="sfc")
                result[variable] = dict(mars=mars, constant_in_time=True)
                continue

            if variable in ("cp", "tp", "sf", "ro", "ssrd", "strd"):
                mars = dict(param=variable, levtype="sfc", type="fc", stream="oper")
                result[variable] = dict(mars=mars, process="accumulation", period=(0, 6))

                continue

            if variable in ("sp", "msl", "10u", "10v", "2t", "2d", "skt", "tcw"):
                mars = dict(param=variable, levtype="sfc")
                result[variable] = dict(mars=mars)
                continue

            unkowns.append(variable)
            mars = dict(param=variable, levtype="sfc")
            result[variable] = dict(mars=mars)

        if unkowns:
            warnings.warn(f"Unknown variables: {unkowns}, assuming an/sfc")

        return result

    def _legacy_check_variables_metadata(self, variables):

        if variables == {}:
            variables.update(self._legacy_variables_metadata())
            return

        first = True
        for variable, metadata in variables.items():
            if "mars" not in metadata:
                if "param" in metadata:
                    if first:
                        LOG.warning("Old metadata format detected. Please update your weights.")
                        first = False
                    metadata["mars"] = metadata.copy()

            if variable in (
                "insolation",
                "cos_solar_zenith_angle",
                "cos_julian_day",
                "sin_julian_day",
                "cos_local_time",
                "sin_local_time",
            ):
                ok = metadata.get("computed_forcing") is True and metadata.get("constant_in_time") is False
                if not ok:
                    if first:
                        LOG.warning("Old metadata format detected. Please update your weights.")
                        first = False
                    metadata["computed_forcing"] = True
                    metadata["constant_in_time"] = False

            if variable in ("cos_latitude", "cos_longitude", "sin_latitude", "sin_longitude"):
                ok = metadata.get("computed_forcing") is True and metadata.get("constant_in_time") is True
                if not ok:
                    if first:
                        LOG.warning("Old metadata format detected. Please update your weights.")
                        first = False
                    metadata["computed_forcing"] = True
                    metadata["constant_in_time"] = True

    @warn
    def _legacy_number_of_grid_points(self):

        POINTS = {"o96": 40_320, "n320": 542_080}

        return POINTS[self.grid.lower()]

    @warn
    def _legacy_data_request(self):
        from anemoi.utils.config import find

        result = find(self._metadata["dataset"], "data_request")
        if len(result) == 0:
            raise ValueError("No data_request found in metadata")

        if len(result) > 1:
            check = ("grid", "area")
            checks = defaultdict(set)
            for r in result:
                for c in check:
                    checks[c].add(str(r.get(c)))

            for c in check:
                if len(checks[c]) > 1:
                    warnings.warn(f"{c} is ambigous: {checks[c]}")

            result = [r for r in result if r["grid"] is not None]

        return result[0]
