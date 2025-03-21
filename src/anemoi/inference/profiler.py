# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import socket
import time
from contextlib import contextmanager
from typing import Generator

import torch

LOG = logging.getLogger(__name__)


@contextmanager
def ProfilingLabel(label: str, use_profiler: bool) -> Generator[None, None, None]:
    """Add label to function so that the profiler can recognize it, only if the use_profiler option is True.

    Parameters
    ----------
    label : str
        Name or description to identify the function.
    use_profiler : bool
        Wrap the function with the label if True, otherwise just execute the function as it is.

    Returns
    -------
    Generator[None, None, None]
        Yields to the caller.
    """
    if use_profiler:
        with torch.autograd.profiler.record_function(label):
            torch.cuda.nvtx.range_push(label)
            yield
            torch.cuda.nvtx.range_pop()
    else:
        yield


@contextmanager
def ProfilingRunner(use_profiler: bool) -> Generator[None, None, None]:
    """Perform time and memory usage profiles of the wrapped code.

    Parameters
    ----------
    use_profiler : bool
        Whether to profile the wrapped code (True) or not (False).

    Returns
    -------
    Generator[None, None, None]
        Yields to the caller.
    """
    dirname = f"profiling-output/{socket.gethostname()}-{int(time.time())}"
    if use_profiler:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            profile_memory=True,
            record_shapes=True,
            activities=activities,
            with_flops=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dirname),
        ) as prof:
            yield
        try:
            torch.cuda.memory._dump_snapshot(f"{dirname}/memory_snapshot.pickle")
        except Exception as e:
            LOG.error(f"Failed to capture memory snapshot {e}")
        torch.cuda.memory._record_memory_history(enabled=None)
        row_limit = 10
        LOG.info(
            f"Top {row_limit} kernels by runtime on CPU:\n {prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=row_limit)}"
        )
        LOG.info(
            f"Top {row_limit} kernels by runtime on CUDA:\n {prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=row_limit)}"
        )
        LOG.info("Memory summary \n%s", torch.cuda.memory_summary())
        LOG.info(
            f"Memory snapshot and trace file stored to '{dirname}'. To view the memory snapshot, upload the pickle file to 'https://pytorch.org/memory_viz'. To view the trace file, see 'https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#use-tensorboard-to-view-results-and-analyze-model-performance'"
        )
    else:
        yield
