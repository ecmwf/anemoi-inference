# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_available_device() -> "torch.device":
    """Get the available device for PyTorch.

    Returns
    -------
    torch.device
        The available device, either 'cuda', 'mps', or 'cpu'.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
