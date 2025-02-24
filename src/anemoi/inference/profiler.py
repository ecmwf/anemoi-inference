# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from contextlib import contextmanager

import torch

LOG = logging.getLogger(__name__)


@contextmanager
def ProfilingLabel(label: str, use_profiler: bool) -> None:
    """
    Add label to function so that the profiler can recognize it, only if the use_profiler option is True.

    Parameters
    ----------
    label : str
        Name or description to identify the function.
    use_profiler : bool
        Wrap the function with the label if True, otherwise just execute the function as it is.

    """
    if use_profiler:
        with torch.autograd.profiler.record_function(label):
            torch.cuda.nvtx.range_push(label)
            yield
            torch.cuda.nvtx.range_pop()
    else:
        yield


@contextmanager
def ProfilingRunner(use_profiler: bool) -> None:
    """
    Perform time and memory usage profiles of the wrapped code.

    Parameters
    ----------
    use_profiler : bool
        Weither to profile the wrapped code (True) or not (False).

    """
    dirname = "profiling-output"
    if use_profiler:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
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
        if torch.cuda.is_available():
            prof.export_memory_timeline(f"{dirname}/memory_timeline.html", device="cuda:0")
    else:
        yield
