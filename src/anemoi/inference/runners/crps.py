# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from typing import Any

from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


@runner_registry.register("crps")
class CrpsRunner(DefaultRunner):
    """Runner for CRPS (Continuous Ranked Probability Score).

    Inherits from DefaultRunner.
    """

    def predict_step(self, model: Any, input_tensor_torch: Any, **kwargs: Any) -> Any:
        """Perform a prediction step using the model.

        Parameters
        ----------
        model : Any
            The model to use for prediction.
        input_tensor_torch : torch.Tensor
            The input tensor for the model.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The prediction result.
        """
        warnings.warn("CRPS runner is deprecated, use DefaultRunner instead")
        return model.predict_step(input_tensor_torch, kwargs["fcstep"])
