# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import pprint
import sys
import types

from anemoi.inference.decorators import main_argument
from anemoi.inference.lazy import torch
from anemoi.inference.modifiers import modifier_registry
from anemoi.inference.modifiers.modifier import Modifier

LOG = logging.getLogger(__name__)


@modifier_registry.register("attention")
@main_argument("attention_implementation")
class AttentionModifier(Modifier):
    def __init__(
        self,
        context,
        attention_implementation: str,
        *,
        config_path: str = "model.processor",
        processor_model_path: str = "model.processor",
        layer_kernels: dict | None = None,
        instantiation_kwargs: dict | None = None,
    ):
        super().__init__(context)
        self.attention_implementation = attention_implementation
        self.config_path = config_path
        self.processor_model_path = processor_model_path
        self._layer_kernels = layer_kernels
        self._instantiation_kwargs = instantiation_kwargs or {"recursive": False}

    def pre_modify(self) -> None:
        """Mock flash_attn module before modifying the model."""

        LOG.warning("Mocking `flash_attn` module for AttentionModifier pre-modify step.")

        class MockFlashAttn(types.ModuleType):
            def __getattr__(self, name):
                if not name.startswith("__"):
                    LOG.warning(f"Accessing mocked `flash_attn` attribute: {name}")
                return None

        sys.modules["flash_attn"] = MockFlashAttn("flash_attn")
        sys.modules["flash_attn.flash_attn_interface"] = MockFlashAttn("flash_attn.flash_attn_interface")

        return super().pre_modify()

    def modify(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """Modify the given model by changing the attention implementation.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be modified.

        Returns
        -------
        torch.nn.Module
            The modified model with updated attention implementation.
        """
        # Navigate to the processor config based on the provided path
        config = self.context.checkpoint._metadata._config

        processor_config = config
        for attr in self.config_path.split("."):
            if not hasattr(processor_config, attr):
                raise AttributeError(f"Attribute '{attr}' not found in the configuration path '{self.config_path}'.")
            processor_config = getattr(processor_config, attr)

        if "attention_implementation" not in processor_config:
            raise AttributeError(
                f"'attention_implementation' not found in the processor configuration at path '{self.config_path}'."
            )

        processor_config["attention_implementation"] = self.attention_implementation
        if self._layer_kernels is not None:
            processor_config["layer_kernels"] = self._layer_kernels

        if "layer_kernels" not in processor_config:
            LOG.warning(
                "Layer kernels not specified in processor config; you may need to set the `layer_kernels` key to the same as the `layer_kernels` in the model configuration."
            )

        model_with_processor = model
        for attr in self.processor_model_path.split(".")[:-1]:
            if not hasattr(model_with_processor, attr):
                raise AttributeError(f"Attribute '{attr}' not found in the model path '{self.processor_model_path}'.")
            model_with_processor = getattr(model_with_processor, attr)

        from hydra.utils import instantiate

        processor_config["num_channels"] = model_with_processor.num_channels

        LOG.info("Set attention implementation to: %s", self.attention_implementation)
        LOG.info("Processor config after modification:\n%s", pprint.pformat(dict(processor_config)))

        model_processor = instantiate(processor_config, **self._instantiation_kwargs).to(self.context.device)

        old_processor_state = getattr(model_with_processor, self.processor_model_path.split(".")[-1]).state_dict()
        setattr(model_with_processor, self.processor_model_path.split(".")[-1], model_processor)
        getattr(model_with_processor, self.processor_model_path.split(".")[-1]).load_state_dict(
            old_processor_state, strict=True
        )

        return model
