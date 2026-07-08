# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import types
import sys
import torch

def checkpoint_contains_module(checkpoint_path, module_name):
    """ Checks if a checkpoint metadata references a given module name."""
    classes = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)

    return any(class_name.startswith(module_name) for class_name in classes)

def spoof_flash_attn():
    """ allows loading a checkpoint with flash attention v2"""
    flash_attn = types.ModuleType("flash_attn")
    sys.modules["flash_attn"] = flash_attn
    flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface
    flash_attn_interface.flash_attn_func = None

