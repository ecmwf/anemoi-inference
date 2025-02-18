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
def ProfilingLabel(label, use_profiler):

    if use_profiler:
        with torch.autograd.profiler.record_function(label):
            torch.cuda.nvtx.range_push(label)
            yield
            torch.cuda.nvtx.range_pop()
    else:
        yield


@contextmanager
def ProfilingRunner(use_profiler):
    if use_profiler:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        with torch.autograd.profiler.profile(with_stack=True, profile_memory=True) as prof:
            yield
        try:
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        except Exception as e:
            LOG.error(f"Failed to capture memory snapshot {e}")
        torch.cuda.memory._record_memory_history(enabled=None)
        LOG.info("Average usage \n%s", prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        LOG.info("Memory summary \n%s", torch.cuda.memory_summary())
    else:
        yield
