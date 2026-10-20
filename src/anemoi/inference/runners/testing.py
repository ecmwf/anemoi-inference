# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.lazy import torch
from anemoi.inference.testing.mock_model import SimpleMockModel
from anemoi.inference.testing.mock_model import VariableOutputSetting

from . import runner_registry
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


class NoModelMixing:
    @cached_property
    def model(self) -> "torch.nn.Module":

        checkpoint: Checkpoint = self.checkpoint  # type: ignore
        multi_metadata = checkpoint.multi_dataset_metadata

        return SimpleMockModel(multi_metadata, prognostic_output=VariableOutputSetting.ONES)


@runner_registry.register("no-model")
class NoModelRunner(NoModelMixing, DefaultRunner):
    """Runner for running tests.

    Inherits from DefaultRunner.
    """
