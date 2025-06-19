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
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from . import Configuration

LOG = logging.getLogger(__name__)


class RunConfiguration(Configuration):
    """Configuration class for a default runner."""

    description: Optional[str] = None

    checkpoint: Union[str, Dict[Literal["huggingface"], Union[Dict[str, Any], str]]]
    """A path to an Anemoi checkpoint file."""

    runner: Union[str, Dict[str, Any]] = "default"
    """The runner to use."""

    lead_time: Union[str, int, datetime.timedelta] = "10d"
    """The lead time for the forecast. This can be a string, an integer or a timedelta object.
    If an integer, it represents a number of hours. Otherwise, it is parsed by :func:`anemoi.utils.dates.as_timedelta`.
    """

    name: Optional[str] = None
    """Used by prepml."""

    verbosity: int = 0
    """The verbosity level of the runner. This can be 0 (default), 1, 2 or 3."""

    use_profiler: bool = False
    """If True, the inference will be profiled, producing time and memory report."""

    world_size: Optional[int] = 1
    """Number of parallel processes, used for parallel inference without SLURM."""

    report_error: bool = False
    """If True, the runner list the training versions of the packages in case of error."""

    input: Union[str, Dict[str, Any]] = "test"
    output: Union[str, Dict[str, Any]] = "printer"

    pre_processors: List[Union[str, Dict[str, Any]]] = []
    post_processors: Optional[List[Union[str, Dict[str, Any]]]] = None  # temporary, default accum from start #131

    forcings: Optional[Dict[str, Dict[str, Any]]] = None
    """Where to find the forcings."""

    device: str = "cuda"
    """The device on which the model should run. This can be "cpu", "cuda" or any other value supported by PyTorch."""

    precision: Optional[str] = None
    """The precision in which the model should run. If not provided, the model will use the precision used during training."""

    allow_nans: Optional[bool] = None
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

    output_frequency: Optional[str] = None
    """The frequency at which to write the output. This can be a string or an integer. If a string, it is parsed by :func:`anemoi.utils.dates.as_timedelta`."""

    env: Dict[str, Union[str, int]] = {}
    """Environment variables to set before running the model. This may be useful to control some packages
    such as `eccodes`. In certain cases, the variables mey be set too late, if the package for which they are intended
    is already loaded when the runner is configured.
    """

    patch_metadata: Dict[str, Any] = {}
    """A dictionary of metadata to patch the checkpoint metadata with. This is used to test new features or to work around
    issues with the checkpoint metadata.
    """

    development_hacks: Dict[str, Any] = {}
    """A dictionary of development hacks to apply to the runner. This is used to test new features or to work around."""

    trace_path: Optional[str] = None
    """A path to a directory where to store the trace of the runner. This is useful to debug the runner."""

    debugging_info: Dict[str, Any] = {}
    """A dictionary to store debug information. This is ignored."""
