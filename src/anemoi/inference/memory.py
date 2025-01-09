# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import gc
import logging
import os

import torch
from anemoi.utils.humanize import bytes_to_human

LOG = logging.getLogger(__name__)


class Collector:
    """Collects memory usage information."""

    def __init__(self):
        self.known = set()
        self.last_total = 0
        self.last_title = "START"
        self.n = 0
        self.plot = int(os.environ.get("MEMORY_PLOT", "0"))

    def __call__(self, title):
        gc.collect()
        total = 0
        tensors = set()
        newobj = []

        for obj in gc.get_objects():
            try:

                if torch.is_tensor(obj) or ("data" in obj.__dict__ and torch.is_tensor(obj.data)):
                    if id(obj) not in self.known:
                        newobj.append(obj)
                    tensors.add(id(obj))
                    total += obj.element_size() * obj.nelement()

            except Exception:
                pass

        added = tensors - self.known
        removed = self.known - tensors

        if newobj and self.plot:
            import objgraph

            gc.collect()
            objgraph.show_backrefs(newobj, filename=f"backref-{self.n:04d}.pdf")
            self.n += 1

        delta = total - self.last_total
        if delta < 0:
            what = "decrease"
            delta = -delta
        else:
            what = "increase"

        LOG.info(
            "[%s] => [%s] memory %s %s (memory=%s, tensors=%s, added=%s, removed=%s)",
            self.last_title,
            title,
            what,
            bytes_to_human(delta),
            bytes_to_human(total),
            len(tensors),
            len(added),
            len(removed),
        )

        self.last_total = total
        self.last_title = title
        self.known = tensors
