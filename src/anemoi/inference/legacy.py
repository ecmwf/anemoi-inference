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
from collections.abc import Callable
from functools import wraps
from typing import Any

from anemoi.inference.types import DataRequest

from .protocol import MetadataProtocol

LOG = logging.getLogger(__name__)


def warn(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to issue a warning when using legacy functions.

    Parameters
    ----------
    func : function
        The legacy function to be wrapped.

    Returns
    -------
    function
        The wrapped function with a warning.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(f"Using legacy {func.__name__}, please try to patch your weights.")
        return func(*args, **kwargs)

    return wrapper


class LegacyMixin(MetadataProtocol):

    # `self` is a `Metadata` object

    @warn
    def _legacy_variables_metadata(self) -> dict[str, dict[str, Any]]:
        """Generate metadata for legacy variables.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Metadata for each variable.
        """
        result: dict[str, Any] = {}
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

    def _legacy_check_variables_metadata(self, variables: dict[str, dict[str, Any]]) -> None:
        """Check and update metadata for legacy variables.

        Parameters
        ----------
        variables : Dict[str, Dict[str, Any]]
            The metadata dictionary to be checked and updated.
        """
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
    def _legacy_number_of_grid_points(self) -> int:
        """Get the number of grid points for the legacy grid.

        Returns
        -------
        int
            Number of grid points.
        """
        POINTS = {"o96": 40_320, "n320": 542_080}

        return POINTS[self.grid.lower()]

    @warn
    def _legacy_data_request(self) -> DataRequest:
        """Retrieve the data request from metadata.

        Returns
        -------
        DataRequest
            The data request information.

        Raises
        ------
        ValueError
            If no data request is found in metadata.
        """
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

            result = [r for r in result if r["grid"]]

        return result[0]
