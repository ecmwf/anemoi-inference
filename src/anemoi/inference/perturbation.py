# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any
import torch
from .checkpoint import Checkpoint

LOG = logging.getLogger(__name__)
R = 6371.0  # Earth radius in km


def haversine(coords: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Compute haversine distance between multiple coordinates and a single point.

    Args:
        coords: (N, 2) tensor of [lat, lon] in degrees
        point: (2,) tensor of [lat, lon] in degrees

    Returns:
        (N,) tensor of distances in kilometers
    """    
    coords_rad = torch.deg2rad(coords)
    point_rad = torch.deg2rad(point)

    lat1, lon1 = coords_rad[:, 0], coords_rad[:, 1]
    lat2, lon2 = point_rad[0], point_rad[1]

    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c


class Perturbation:
    """Perturbation class."""

    def __init__(
        self, 
        checkpoint: str, 
        perturbed_variable: str, 
        perturbation_location: float, 
        perturbation_radius_km: float = 100.0,
        patch_metadata: dict[str, Any] = {}
    ) -> None:
        """Initialize the Perturbation.

        Parameters
        ----------
        perturbed_variable : str
            The variable to perturb.
        """
        assert len(perturbation_location) == 2, "perturbation_location must be a tuple of (lat, lon)"
        assert perturbation_location[0] >= -90 and perturbation_location[0] <= 90, "Latitude must be between -90 and 90"
        assert perturbation_location[1] >= 0 and perturbation_location[1] <= 360, "Longitude must be between 0 and 360"
        self.perturbed_variable = perturbed_variable
        self.perturbation_location = torch.tensor(perturbation_location)
        self.perturbation_radius_km = perturbation_radius_km
        self._checkpoint = Checkpoint(checkpoint, patch_metadata=patch_metadata)

    @property
    def variable_to_output_tensor_index(self) -> dict[str, int]:
        return self._checkpoint._metadata.variable_to_output_tensor_index

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            1,
            1, 
            self._checkpoint._metadata.number_of_grid_points, 
            len(self._checkpoint._metadata.variable_to_output_tensor_index)
        )

    @property
    def coords(self) -> torch.Tensor:
        lats = torch.from_numpy(self._checkpoint._metadata._supporting_arrays["latitudes"])
        lons = torch.from_numpy(self._checkpoint._metadata._supporting_arrays["longitudes"])
        return torch.stack([lats, lons], dim=-1)

    def create(self, *args, **kwargs) -> torch.Tensor:
        """Get the perturbation data."""
        var_idx = self.variable_to_output_tensor_index[self.perturbed_variable]
        perturbation = torch.zeros(self.output_shape)

        # Get index of the closest point
        dists = haversine(self.coords, self.perturbation_location)
        closest_idx = torch.where(dists < self.perturbation_radius_km)[0]

        assert len(closest_idx) > 0, "No grid points found within the specified perturbation radius."

        perturbation[..., closest_idx, var_idx] = 1.0
        return perturbation
