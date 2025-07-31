# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging
from typing import Any
from typing import Literal

from pydantic import Field

from anemoi.inference.types import ProcessorConfig

from . import Configuration

LOG = logging.getLogger(__name__)


class RunConfiguration(Configuration):
    """Configuration class for a default runner."""

    description: str | None = None

    checkpoint: str | dict[Literal["huggingface"], dict[str, Any] | str]
    """A path to an Anemoi checkpoint file."""

    runner: str | dict[str, Any] = "default"
    """The runner to use."""

    lead_time: str | int | datetime.timedelta = "10d"
    """The lead time for the forecast. This can be a string, an integer or a timedelta object.
    If an integer, it represents a number of hours. Otherwise, it is parsed by :func:`anemoi.utils.dates.as_timedelta`.
    """

    name: str | None = None
    """Used by prepml."""

    verbosity: int = 0
    """The verbosity level of the runner. This can be 0 (default), 1, 2 or 3."""

    use_profiler: bool = False
    """If True, the inference will be profiled, producing time and memory report."""

    world_size: int | None = 1
    """Number of parallel processes, used for parallel inference without SLURM."""

    report_error: bool = False
    """If True, the runner list the training versions of the packages in case of error."""

    input: str | dict[str, Any] = "test"
    output: str | dict[str, Any] = "printer"

    pre_processors: list[ProcessorConfig] = []
    post_processors: list[ProcessorConfig] = []

    forcings: dict[str, dict[str, Any]] | None = None
    """Where to find the forcings."""

    device: str = "cuda"
    """The device on which the model should run. This can be "cpu", "cuda" or any other value supported by PyTorch."""

    precision: str | None = None
    """The precision in which the model should run. If not provided, the model will use the precision used during training."""

    allow_nans: bool | None = None
    """
    - If None (default), the model will check for NaNs in the input. If NaNs are found, the model issue a warning and `allow_nans` to True.
    - If False, the model will raise an exception if NaNs are found in the input and output.
    - If True, the model will allow NaNs in the input and output.
    """

    use_grib_paramid: bool = False
    """If True, the runner will use the grib parameter ID when generating MARS requests."""

    write_initial_state: bool = True
    """Wether to write the initial state to the output file. If the model is multi-step, only fields at the forecast reference date are
    written.
    """

    predict_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra keyword arguments to pass to the model's predict_step method. Will ignore kwargs that are already passed by the runner."""

    typed_variables: dict[str, dict] = Field(default_factory=dict)
    """A list of typed variables to support the encoding of outputs."""

    output_frequency: str | None = None
    """The frequency at which to write the output. This can be a string or an integer. If a string, it is parsed by :func:`anemoi.utils.dates.as_timedelta`."""

    env: dict[str, str | int] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured.
    """

    patch_metadata: dict[str, Any] = {}
    """A dictionary of metadata to patch the checkpoint metadata with. This is used to test new features or to work around
    issues with the checkpoint metadata.
    """

    development_hacks: dict[str, Any] = {}
    """A dictionary of development hacks to apply to the runner. This is used to test new features or to work around."""

    trace_path: str | None = None
    """A path to a directory where to store the trace of the runner. This is useful to debug the runner."""

    debugging_info: dict[str, Any] = {}
    """A dictionary to store debug information. This is ignored."""
