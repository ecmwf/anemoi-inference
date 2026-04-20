# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from anemoi.utils.config import DotDict

LOG = logging.getLogger(__name__)


def input_types_config(config: DotDict, *names: str) -> DotDict:
    """Get the input type configuration from the config using a list of possible names, returning the first one found.
    For example, names could be `["constant_forcings", "input"]` to get forcings from the `constant_forcings` input, or the main input if not specified.
    """
    for name in names:
        if name.startswith("-"):
            deprecated = True
            name = name[1:]  # Remove the leading dash
        else:
            deprecated = False
        if config.get(name):
            if deprecated:
                LOG.warning(
                    f"🚫 The `{name}` input forcings configuration is deprecated. "
                    f"Please use the `{names[0]}` configuration instead."
                )
            if name != names[0]:
                LOG.info(f"Loading `config.{names[0]}` from `config.{name}`")

            return config[name]

    return config.input


def multi_datasets_config(config: Any, dataset_name: str, datasets: list[str], strict: bool = True):
    """Extract the configuration for a specific dataset name from a multi-dataset config entry.
    If the config only has a single entry, the entry corresponding to the dataset name is returned if it exists, otherwise the config is returned as is.
    For example, used with a config like:
        ```
        output:
            era5:
                grib: out-era5.grib
            cerra:
                netcdf: out-cerra.nc
        ```

    If `strict` is True and `config` is a dictionary with multiple keys but `dataset_name` is not one of them, an error is raised.
    If `strict` is False, the config is returned as is if `dataset_name` is not one of the keys. Used for config-entries that can take multi-key dicts (like `patch_metadata`).
    """
    if isinstance(config, dict):
        if any(dataset in config for dataset in datasets):
            assert all(dataset in config for dataset in datasets), (
                f"Some but not all datasets in the checkpoint are present in {config}\n"
                f"Datasets in checkpoint: {datasets}, datasets in config: {list(config.keys())}"
            )

        if strict and len(config.keys()) > 1:
            assert (
                dataset_name in config
            ), f"Dataset name '{dataset_name}' not found in config. Available keys: {list(config.keys())}"

        if dataset_name in config:
            return config[dataset_name]

    return config
