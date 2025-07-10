# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""List of precisions supported by the inference runner."""

########################################################################################################
# Don't import torch here, it takes a long time to load and is not needed for the runner registration. #
########################################################################################################


from functools import cached_property


class LazyDict:
    """A dictionary that lazily loads its values.
    So we don't import torch at the top level, which can be slow.
    """

    def get(self, key, default=None):
        return self._mapping.get(key, default)

    def keys(self):
        return self._mapping.keys()

    def values(self):
        return self._mapping.values()

    def items(self):
        return self._mapping.items()

    def __getitem__(self, key):
        return self._mapping[key]

    @cached_property
    def _mapping(self):
        import torch

        return {
            "16-mixed": torch.float16,
            "16": torch.float16,
            "32": torch.float32,
            "b16-mixed": torch.bfloat16,
            "b16": torch.bfloat16,
            "bf16-mixed": torch.bfloat16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "f16": torch.float16,
            "f32": torch.float32,
            "float16": torch.float16,
            "float32": torch.float32,
        }


PRECISIONS = LazyDict()

if __name__ == "__main__":
    # This is just to make sure that the module can be imported without errors.
    # It will not be executed when the module is imported, only when run as a script.

    print("Available precisions:", list(PRECISIONS.keys()))
    print("Available precisions:", list(PRECISIONS.values()))
    print("Available precisions:", list(PRECISIONS.items()))

    print("Torch float16:", PRECISIONS["16"])
    print("Torch bfloat16:", PRECISIONS["b16"])
    print("Torch float32:", PRECISIONS["32"])
