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

TARGET_RESOLUTION = (721, 1440)


@pre_processor_registry.register("image_overlay")
class ImageOverlay(Processor):
    """Overlay an image to a field.

    Usage:
        Provide an image to overlay to the field.
        RGB channels are averaged and regridded to the field grid.
        The image is then rescaled and added to the field.

        If white is to be transparent, set invert=True.
    """

    def __init__(
        self,
        context,
        image: PathLike,
        fields: list[dict[str, Any]],
        rescale: float = 1,
        method: str = "add",
        invert: bool = False,
        **kwargs,
    ):
        if not _PILLOW_AVAILABLE:
            raise ImportError("Pillow is required for this pre-processor.")
        if not _EARTHKIT_REGRID_AVAILABLE:
            raise ImportError("earthkit.regrid is required for this pre-processor.")

        super().__init__(context)

        self._image = image
        self._fields = fields
        self._rescale = rescale
        self._method = method
        self._invert = invert

    @functools.cached_property
    def image(self) -> "Image":
        """Load the image and pad it to the target aspect ratio."""

        image = self._image
        if str(image).startswith("http"):
            from urllib.request import urlopen

            image = urlopen(image)

        image = Image.open(image)
        return self._pad_image_to_aspect_ratio(image, TARGET_RESOLUTION[1] / TARGET_RESOLUTION[0])

    def _pad_image_to_aspect_ratio(self, image: "Image", target_aspect_ratio: float):
        """Pad an image to a target aspect ratio."""
        width, height = image.size
        current_aspect_ratio = width / height

        if current_aspect_ratio > target_aspect_ratio:
            new_height = int(width / target_aspect_ratio)
            padding = (0, (new_height - height) // 2)
        else:
            new_width = int(height * target_aspect_ratio)
            padding = ((new_width - width) // 2, 0)

        new_size = (width + 2 * padding[0], height + 2 * padding[1])
        new_image = Image.new("RGB", new_size, (255, 255, 255) if self._invert else (0, 0, 0))
        new_image.paste(image, padding)

        return new_image

    def _image_to_array(self, image: "Image", grid) -> np.ndarray:
        """Regrid an image to field gridspec, and average the RGB channels."""

        image = image.resize(reversed(TARGET_RESOLUTION), Image.Resampling.BILINEAR)

        image_array = np.array(image).copy()

        image_array = np.mean(image_array, axis=-1)  # .swapaxes(-1, 0)
        image_array = np.nan_to_num(image_array, 0)

        return earthkit.regrid.interpolate(image_array, {"grid": (0.25, 0.25)}, {"grid": grid})

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
            outgrid = self.checkpoint.grid
            image_array = self._image_to_array(self.image, outgrid) / 255.0

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
        return f"ImageOverlay(image='{self._image}', fields={list(map(dict, self._fields))}, rescale={self._rescale})"
