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
import random

import torch
from anemoi.utils.humanize import bytes_to_human

LOG = logging.getLogger(__name__)


class Collector:
    """Collects memory usage information."""

    def __init__(self):
        self.known = set()
        self.last_total = 0
        self.last_title = "START"

    def __call__(self, title, quiet=True):
        gc.collect()
        total = 0
        tensors = set()

        for obj in gc.get_objects():
            try:

                if torch.is_tensor(obj) or ("data" in obj.__dict__ and torch.is_tensor(obj.data)):
                    tensors.add(id(obj))
                    total += obj.element_size() * obj.nelement()

            except Exception:
                pass

        added = tensors - self.known
        removed = self.known - tensors

        if not quiet:
            import objgraph

            one = random.choice(added)
            del added
            gc.collect()
            objgraph.show_backrefs([one], filename="backref.png")

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
