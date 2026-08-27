# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from datetime import timedelta
from functools import cached_property
from types import MappingProxyType as frozendict
from typing import Optional

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import ComputedForcings
from anemoi.inference.output import Output
from anemoi.inference.runner import Kind
from anemoi.inference.types import FloatArray, State
from anemoi.inference.variables import Variables
from anemoi.utils.spectral import InverseDCT2D, DCT2D

from ..checkpoint import Checkpoint
from ..metadata import Metadata
from . import runner_registry
from .downscaling import DownscalingRunner

LOG = logging.getLogger(__name__)


@runner_registry.register("equal_SNR_downscaling")
class EqualSNRDownscalingRunner(DownscalingRunner):
    """Custom runner class for inference.

    This class provides an implementation for running inference with an equal SNR (as suggested by Falck et. al.).
    """
    def __init__(
        self,
        config: RunConfiguration,
        time_step: int | str | timedelta,
        field_shape: tuple[int, ...] | None = None,
        hres_zarr: str | None = None,
        noise_scheduler_params: dict | None = None,
        sampler_params: dict | None = None,
    ):
        super().__init__(
            config=config,
            time_step=time_step,
            field_shape=field_shape,
            hres_zarr=hres_zarr,
            noise_scheduler_params=noise_scheduler_params,
            sampler_params=sampler_params
        )

        # Reading dimensions and variance
        _, arrays = load_metadata(self._checkpoint.path, supporting_arrays=True)
        self.NX = len(torch.unique(torch.from_numpy(arrays['longitudes'])))
        self.NY = len(torch.unique(torch.from_numpy(arrays['latitudes'])))
        self.variance = torch.from_numpy(arrays["variance"]).to(self.device)

        # Initializing transforms
        self.itransform = InverseDCT2D(self.NX, self.NY, norm='ortho')
        self.transform = DCT2D(self.NX, self.NY, norm='ortho')


    def predict_step(self, model, input_tensor_torch, **kwargs) -> torch.Tensor:
        date = kwargs["date"]
        step = kwargs["step"]

        input_date = date - step
        low_res_tensor = input_tensor_torch
        high_res_tensor = self._prepare_high_res_input_tensor(input_date)

        LOG.info("Low res tensor shape: %s", low_res_tensor.shape)
        LOG.info("High res tensor shape: %s", high_res_tensor.shape)

        # TODO: remove?
        print("self.noise_scheduler_params", self.noise_scheduler_params)
        print("self.sampler_params", self.sampler_params)

        # Injecte ces éléments vers model.sample() (cf. diagramme uml)
        kwargs["itransform"] = self.itransform
        kwargs["variance"] = self.variance

        output_tensor = model.predict_step(
            low_res_tensor,
            high_res_tensor,
            noise_scheduler_params=self.noise_scheduler_params,
            sampler_params=self.sampler_params,
            **kwargs,
        )

        return output_tensor