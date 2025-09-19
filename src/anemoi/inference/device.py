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
    import os

    import torch

    if torch.cuda.is_available():
        local_rank_env = os.environ.get("LOCAL_RANK")
        slurm_local = os.environ.get("SLURM_LOCALID")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
        elif slurm_local is not None:
            local_rank = int(slurm_local)
        else:
            local_rank = 0
        torch.cuda.set_device(local_rank)  # important for NCCL
        return torch.device(f"cuda:{local_rank}")
        return torch.device("mps")
    else:
        return torch.device("cpu")
