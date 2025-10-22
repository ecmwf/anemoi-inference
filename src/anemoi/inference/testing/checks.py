# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from . import testing_registry

if TYPE_CHECKING:
    from anemoi.transform.variables import Variable

    from anemoi.inference.checkpoint import Checkpoint

LOG = logging.getLogger(__name__)

# checks here are used in the integration tests and can be reused in other test suites
# checks must use keyword arguments only and must have a **kwargs


@testing_registry.register("check_grib")
def check_grib(
    *,
    file: Path,
    expected_variables: list["Variable"],
    grib_keys: dict = {},
    check_accum: str | None = None,
    check_nans=False,
    **kwargs,
) -> None:
    LOG.info(f"Checking GRIB file: {file}")
    import earthkit.data as ekd
    import numpy as np

    ds = ekd.from_source("file", file)

    assert len(ds) > 0, "No fields found in the GRIB file."

    params = set(ds.metadata("param"))
    expected_params = [var.param for var in expected_variables]
    assert (
        set(expected_params) == params
    ), f"Expected parameters {set(expected_params)} do not match found parameters {params}."

    for field in ds:
        for key, value in grib_keys.items():
            assert field[key] == value, f"Key {key} does not match expected value {value}."
        assert not np.all(field.values == 0), f"Field {field} is zero."

        if check_nans:
            assert not any(np.isnan(field.values)), f"Field {field} contains NaN values."

    # check time continuity
    fields = ds.sel(param=expected_params[0])
    previous_field = fields[0]
    for field in fields[1:]:
        assert field["step"] > previous_field["step"] and field["valid_time"] > previous_field["valid_time"], (
            f"Field step {field['step']} (valid_time {field['valid_time']}) is not greater than previous field "
            f"step {previous_field['step']} (valid_time {previous_field['valid_time']})."
        )
        previous_field = field

    if not check_accum:
        return

    # check that the accumulation field is accumulating
    fields = list(ds.sel(param=check_accum))
    if len(fields) < 2:
        raise ValueError(f"No fields found for accumulation check: {check_accum}")

    averages = [np.average(field.values) for field in fields]
    assert all(curr > prev for prev, curr in zip(averages, averages[1:])), f"{check_accum} is not accumulating"


@testing_registry.register("check_grib_cutout")
def check_grib_cutout(
    *,
    file: Path,
    checkpoint: "Checkpoint",
    reference_grib: str,
    reference_datetime: str | None = None,
    mask="lam_0",
    **kwargs,
):
    """check shape and values of inner region against a reference GRIB"""

    LOG.info(f"Checking cutout: {file}")
    import earthkit.data as ekd
    import numpy as np

    ds = ekd.from_source("file", file)
    ref_ds = ekd.from_source("file", reference_grib)

    if not reference_datetime:
        reference_datetime = ref_ds.order_by(valid_datetime="ascending")[-1].metadata("valid_time")

    assert len(ds) > 0, "No fields found in the output GRIB file."
    assert len(ref_ds) > 0, "No fields found in the reference GRIB file."

    mask = checkpoint.load_supporting_array(f"{mask}/cutout_mask")
    prognostic_params = [
        var.param for var in checkpoint.typed_variables.values() if var.name in checkpoint.prognostic_variables
    ]

    for param in prognostic_params:
        fields = ds.sel(param=param)
        ref_fields = ref_ds.sel(param=param, valid_time=reference_datetime)

        assert len(fields) > 0, f"No fields found for variable {param} in output file."
        assert len(ref_fields) > 0, f"No fields found for variable {param} in reference file at {reference_datetime}."

        for field in fields:
            assert field.values.shape[-1] == np.sum(
                mask
            ), f"Variable {param} shape {field.shape[-1]} does not match mask size {np.sum(mask)}."
            assert np.allclose(
                field.values, ref_fields.sel(level=field.metadata("level"))[0].values
            ), f"Variable {param} in LAM does not match reference data at {reference_datetime}."


@testing_registry.register("check_with_xarray")
def check_with_xarray(
    *, file: Path, expected_variables: list["Variable"], check_accum: str | None = None, check_nans=False, **kwargs
) -> None:
    LOG.info(f"Checking file: {file}")
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(file)

    assert len(ds.data_vars) > 0, "No data found in the xarray compatible file."

    # lat and lon are variables in the file, but they are not variables of the model so skip them
    skip = {"latitude", "longitude"}
    params = set(ds.data_vars) - skip
    expected_params = [var.name for var in expected_variables]
    assert (
        set(expected_params) == params
    ), f"Expected parameters {set(expected_params)} do not match found parameters {params}."

    for var in ds.data_vars:
        if var in skip:
            continue
        assert not np.all(ds[var].values == 0), f"Variable {var} is zero."

        if check_nans:
            assert not np.isnan(ds[var].values).any(), f"Variable {var} contains NaN values."

    if not check_accum:
        return

    data_vars = list(ds.data_vars[check_accum])
    if len(data_vars) < 2:
        raise ValueError(f"No variables found for accumulation check: {check_accum}")

    averages = [np.average(data.values) for data in data_vars]
    assert all(curr > prev for prev, curr in zip(averages, averages[1:])), f"{check_accum} is not accumulating"


@testing_registry.register("check_cutout_with_xarray")
def check_cutout_with_xarray(
    *,
    file: Path,
    checkpoint: "Checkpoint",
    mask="lam_0",
    reference_date: str = None,
    reference_dataset={},
    reference_file=None,
    **kwargs,
) -> None:
    LOG.info(f"Checking cutout: {file}")
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(file)

    # check shape of inner region against the mask in the checkpoint
    mask = checkpoint.load_supporting_array(f"{mask}/cutout_mask")
    for var in ds.data_vars:
        assert ds[var].shape[-1] == np.sum(
            mask
        ), f"Variable {var} shape {ds[var].shape[-1]} does not match mask size {np.sum(mask)}."

    # check that the extracted inner region matches the reference input dataset
    # the mock model passes input to output, values in the output file should match the input dataset at the reference date
    if reference_dataset:
        from anemoi.datasets import open_dataset

        ref_ds = open_dataset(**reference_dataset, start=reference_date, end=reference_date)

        for var in checkpoint.prognostic_variables:
            assert var in ds.data_vars, f"Variable {var} not found in output file."
            ref_idx = ref_ds.name_to_index[var]
            for data in ds[var]:
                assert np.allclose(
                    data.values, ref_ds[0, ref_idx, 0, :]
                ), f"Variable {var} in output does not match reference data at {reference_date}."
    elif reference_file:
        # check against a reference file, implement when needed
        raise NotImplementedError("Reference file check is not implemented yet.")


@testing_registry.register("check_file_exist")
def check_file_exist(*, file: Path, **kwargs) -> None:
    LOG.info(f"Checking file exists: {file}")
    assert file.exists(), f"File {file} does not exist."
    assert file.stat().st_size > 0, f"File {file} is empty."


@testing_registry.register("check_files_in_directory")
def check_files_in_directory(*, file: Path, expected_files: int | None = None, **kwargs) -> None:
    LOG.info(f"Checking directory: {file}")
    assert file.exists() and file.is_dir(), f"Directory {file} does not exist or is not a directory."
    if expected_files is not None:
        actual_files = [f for f in file.iterdir() if f.is_file()]
        if expected_files < 0:
            assert len(actual_files) > 0, "Expected at least one file, but found none."
        else:
            assert (
                len(actual_files) == expected_files
            ), f"Expected {expected_files} files, but found {len(actual_files)}."
