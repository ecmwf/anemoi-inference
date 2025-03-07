# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
import logging
from os import PathLike
from typing import Any
from typing import Literal

import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list

from ..processor import Processor
from . import pre_processor_registry

LOG = logging.getLogger(__name__)

_PILLOW_AVAILABLE = False
try:
    from PIL import Image

    _PILLOW_AVAILABLE = True
except ImportError:
    pass

_EARTHKIT_REGRID_AVAILABLE = False
try:
    import earthkit.regrid

    _EARTHKIT_REGRID_AVAILABLE = True
except ImportError:
    pass

VALID_METHODS = Literal["add", "multiply", "replace"]


@pre_processor_registry.register("overlay")
class ArrayOverlay(Processor):
    """Overlay an array/image to a field.

    Pads the overlay to the 1440x721 aspect ratio, and regrids from 0.25,0.25
    to the field grid.

    Usage
    ------
        Provide an array to overlay to the field.

        RGB channels are averaged and regridded to the field grid.

        The overlay is then rescaled and added to the field.

        If white is to be transparent, set invert=True.

    Example
    --------
    ```yaml
    pre_processors:
        - overlay:
            overlay: "https://get.ecmwf.int/repository/anemoi/assets/duck.jpg"
            fields:
                - {"level": 850, "shortName": "t"}
            rescale: 10
            method: "add"
            invert: true
    ```
    """

    def __init__(
        self,
        context,
        overlay: PathLike,
        fields: list[dict[str, Any]],
        rescale: float = 1,
        method: VALID_METHODS = "add",
        invert: bool = False,
    ):
        if not _PILLOW_AVAILABLE:
            raise ImportError("Pillow is required for this pre-processor.")
        if not _EARTHKIT_REGRID_AVAILABLE:
            raise ImportError("earthkit.regrid is required for this pre-processor.")

        super().__init__(context)

        self._overlay = overlay
        self._fields = fields
        self._rescale = rescale
        self._method = method
        self._invert = invert

    @functools.cached_property
    def overlay(self) -> "Image":
        """Load the overlay."""

        overlay = self._overlay
        if str(overlay).startswith("http"):
            from urllib.request import urlopen

            overlay = urlopen(overlay)

        return Image.open(overlay)

    def _pad_overlay_to_aspect_ratio(self, overlay: "Image", target_aspect_ratio: float):
        """Pad an overlay to a target aspect ratio."""
        width, height = overlay.size
        current_aspect_ratio = width / height

        if current_aspect_ratio > target_aspect_ratio:
            new_height = int(width / target_aspect_ratio)
            padding = (0, (new_height - height) // 2)
        else:
            new_width = int(height * target_aspect_ratio)
            padding = ((new_width - width) // 2, 0)

        new_size = (width + 2 * padding[0], height + 2 * padding[1])
        new_overlay = Image.new("RGB", new_size, (255, 255, 255) if self._invert else (0, 0, 0))
        new_overlay.paste(overlay, padding)

        return new_overlay

    def _overlay_to_array(self, overlay: "Image", grid) -> np.ndarray:
        """Regrid an image to field gridspec, and average the RGB channels."""

        overlay = overlay.resize((1440, 721), Image.Resampling.BILINEAR)

        overlay_array = np.array(overlay).copy()

        overlay_array = np.mean(overlay_array, axis=-1)  # .swapaxes(-1, 0)
        overlay_array = np.nan_to_num(overlay_array, 0)

        return earthkit.regrid.interpolate(overlay_array, {"grid": (0.25, 0.25)}, {"grid": grid})

    @functools.lru_cache
    def prepare_overlay(self, grid):
        """Prepare the overlay for the field grid."""
        padded_overlay = self._pad_overlay_to_aspect_ratio(self.image, 1440 / 721)
        return self._overlay_to_array(padded_overlay, grid)

    def process(self, fields):
        result = []
        for field in fields:
            metadata = dict(field.metadata().items())
            for field_dict in self._fields:
                if all(key in metadata.keys() and metadata[key] == val for key, val in field_dict.items()):
                    break
            else:
                result.append(field)
                continue

            LOG.info("ImageOverlay: Applying image overlay to %s.", field)

            data = field.to_numpy()

            image_array = self.prepare_overlay(self.checkpoint.grid) / 255.0
            rescaled_array = ((1 - image_array) if self._invert else image_array) * self._rescale

            if self._method == "add":
                data += rescaled_array
            elif self._method == "multiply":
                data *= rescaled_array
            elif self._method == "replace":
                data = np.where(rescaled_array > 0, rescaled_array, data)
            else:
                raise ValueError(f"Unknown method {self._method}.")

            np.save("overlay.npy", data)
            result.append(new_field_from_numpy(data, template=field))

        return new_fieldlist_from_list(result)

    def __repr__(self):
        return (
            f"ArrayOverlay(overlay='{self._overlay}', fields={list(map(dict, self._fields))}, rescale={self._rescale})"
        )
