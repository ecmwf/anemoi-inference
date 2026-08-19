# (C) Copyright 2025-2026 Anemoi contributors.
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

    def predict_step(self, model: Any, input_tensor_torch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Perform a prediction step using the model.

        Parameters
        ----------
        model : Any
            The model to use for prediction.
        input_tensor_torch : dict[str, Any]
            The input tensors for the model.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            The prediction result, keyed by the same keys as the input tensor dictionary.
        """
        warnings.warn("CRPS runner is deprecated, use DefaultRunner instead")
        assert len(input_tensor_torch) == 1, "CRPS runner only supports legacy single dataset models"

        dataset, tensor = next(iter(input_tensor_torch.items()))

        return {dataset: model.predict_step(tensor, kwargs["fcstep"])}
