# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Ensemble DA Cycling Runner (one member per invocation).

Runs DA cycling + forecast for ensemble checkpoints (e.g. DAGraphEnsForecaster /
GraphEnsForecaster models with an internal noise injector). Follows the
operational aifs-ens contract: each invocation produces ONE stochastic member;
an external loop over ``member`` (e.g. an evaluation-suite ``ensemble.loop`` or
a shell loop) builds the ensemble. The member number is routed to the outputs
via GRIB ``eps``/``number`` keys or a ``{member}`` placeholder in output paths.

Seed discipline (mirrors DAGraphEnsForecaster training, where all members are
tiled from ONE analysis):

- The DA cycling phase is seeded with ``base_seed`` — identical for every
  member run, so all members share a bit-identical analysis.
- At the DA -> forecast transition the RNG is re-seeded with a member-specific
  seed, so members diverge only in the forecast via the model's noise injector.
- ``independent_da: true`` seeds everything per-member instead (an ensemble of
  DAs — spread includes DA-phase noise; deliberate deviation from training).
- ``base_seed: null`` disables seeding entirely (non-reproducible members).

Configuration example::

    runner: da_ens_cycling
    da_cycling:
      da_cycles: 4
      observation_sources: { ... }
    da_ens_cycling:
      member: 0            # this run's member number (loop externally)
      base_seed: 42
      independent_da: false

With ``da_cycling.da_cycles: 0`` this degenerates to a seeded default-runner
member (the plain GraphEnsForecaster / aifs-ens pattern).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from anemoi.inference.lazy import torch
from anemoi.inference.runners.da_cycling import DACyclingRunner

from . import runner_registry

if TYPE_CHECKING:
    from anemoi.inference.config.run import RunConfiguration

LOG = logging.getLogger(__name__)


def _substitute_member(obj: Any, member: int) -> Any:
    """Replace ``{member}`` placeholders in string values of a config structure."""
    if isinstance(obj, str):
        return obj.replace("{member}", f"{member:02d}") if "{member}" in obj else obj
    if isinstance(obj, dict):
        return {k: _substitute_member(v, member) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_member(v, member) for v in obj]
    return obj


@runner_registry.register("da_ens_cycling")
class DAEnsCyclingRunner(DACyclingRunner):
    """DA cycling runner for ensemble checkpoints: one seeded member per invocation."""

    def __init__(self, config: "RunConfiguration") -> None:
        super().__init__(config)

        ens_config = getattr(config, "da_ens_cycling", None) or {}
        if not isinstance(ens_config, dict):
            ens_config = dict(ens_config)

        self.member: int = int(ens_config.get("member", 0) or 0)
        base_seed = ens_config.get("base_seed", None)
        self.base_seed: int | None = int(base_seed) if base_seed is not None else None
        self.independent_da: bool = bool(ens_config.get("independent_da", False))

        LOG.info(
            "DA Ens Cycling Runner: member=%d, base_seed=%s, independent_da=%s",
            self.member,
            self.base_seed,
            self.independent_da,
        )

        # Post-DA forecast calls are warm-started, not cold starts: ensemble
        # models consume fcstep (capped at 1) as an input channel, and training
        # derives it from the forward-call count, which includes the DA cycles.
        self._fcstep_offset = self.da_cycles

    @property
    def _member_seed(self) -> int | None:
        """Member-specific seed for the forecast phase."""
        if self.base_seed is None:
            return None
        return self.base_seed + 1000 * (self.member + 1)

    def execute(self) -> None:
        """Execute one member run: template outputs, seed, then run DA + forecast."""
        # Route this member's output to its own destination when the config
        # uses a {member} placeholder (e.g. path: forecast_m{member}.nc).
        if getattr(self.config, "output", None) is not None:
            self.config.output = _substitute_member(self.config.output, self.member)

        if self.base_seed is not None:
            # Same seed for every member run unless independent_da: the DA
            # cycling phase then produces a bit-identical shared analysis,
            # matching training (one analysis, members tiled from it).
            # Without DA cycles there is no shared phase — seed per member
            # directly (the _on_analysis_ready hook never fires then).
            if self.da_cycles == 0 or self.independent_da:
                da_seed = self._member_seed
            else:
                da_seed = self.base_seed
            LOG.info("Seeding DA phase with %d (member %d)", da_seed, self.member)
            torch.manual_seed(da_seed)

        super().execute()

    def _on_analysis_ready(self) -> None:
        """Re-seed per member at the DA -> forecast transition.

        The analysis is shared across members (same DA seed); from here on the
        RNG differs per member, so ensemble spread comes exclusively from the
        forecast-phase noise draws — mirroring the training setup.
        """
        if self.base_seed is not None:
            LOG.info("Seeding forecast phase with %d (member %d)", self._member_seed, self.member)
            torch.manual_seed(self._member_seed)
