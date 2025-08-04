# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import torch


class LazyDevice(str):
    def __init__(self, prefer: list[str] | None = None):
        """prefer: Ordered list of preferred device types.
        Options: 'cuda', 'mps', 'cpu'
        Default: ['cuda', 'mps', 'cpu']
        """
        if prefer is None:
            prefer = ["cuda", "mps", "cpu"]
        self.prefer = prefer
        self._device = None

    def resolve(self) -> str:
        if self._device is not None:
            return self._device

        for dev in self.prefer:
            if dev == "cuda" and torch.cuda.is_available():
                self._device = "cuda"
                break
            elif dev == "mps" and torch.backends.mps.is_available():
                self._device = "mps"
                break
            elif dev == "cpu":
                self._device = "cpu"
                break

        if self._device is None:
            raise RuntimeError(f"No supported devices found among: {self.prefer}")

        return self._device

    def __str__(self):
        return self.resolve()

    def __repr__(self):
        return self.resolve()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.resolve()

    def __getattribute__(self, key):
        if key in ["_device", "resolve", "prefer"]:
            return super().__getattribute__(key)
        return getattr(self.resolve(), key)


device = LazyDevice()

__all__ = ["device"]
