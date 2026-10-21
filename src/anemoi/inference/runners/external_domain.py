# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Any

import numpy as np

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.lazy import torch

from ..checkpoint import Checkpoint
from ..decorators import main_argument
from ..runner import RunnerClasses
from ..runners.default import DefaultRunner
from . import runner_registry
from .external_graph import update_state_dict

LOG = logging.getLogger(__name__)


def _rename(obj: Any, renames: dict, remove: set) -> Any:
    """Recursively rename names appearing anywhere in `obj` according to `renames`.

    This covers dict keys, list elements, and plain string values. Dict keys and
    list elements present in `remove` are dropped entirely (used to discard other, unused training
    domains).

    Parameters
    ----------
    obj : Any
        The (sub-)structure to rename. Will act non-trivially only on dicts and lists and strings.
    renames : dict
        Mapping of old name -> new name, applied to dict keys, list elements, and string leaf values.
    remove : set
        Names to drop entirely as dict keys or list elements (not applied to leaf values).

    Returns
    -------
    Any
        A new structure with names renamed/dropped as described above.
    """
    if isinstance(obj, dict):
        renamed = {}
        for key, value in obj.items():
            if key in remove:
                continue
            renamed[renames.get(key, key)] = _rename(value, renames, remove)
        return renamed
    if isinstance(obj, list):
        renamed_list = []
        for item in obj:
            if isinstance(item, str) and item in remove:
                continue
            renamed_list.append(_rename(item, renames, remove))
        return renamed_list
    if isinstance(obj, str):
        return renames.get(obj, obj)
    return obj


def _adapt_metadata(metadata: dict, new_domain: str, hidden_name: str) -> tuple[dict, str, set]:
    """Adapt metadata (from a checkpoint) to use of `new_domain` during inference.

    This involves renaming of the training dataset names to `new_domain`, dropping redundant training dataset names
    and renaming the hidden mesh node type for the new domain.

    Parameters
    ----------
    metadata : dict
        Raw checkpoint metadata (as returned by `anemoi.utils.checkpoints.load_metadata`).
    new_domain : str
        Name of the new domain to introduce.
    hidden_name : str
        Name of the hidden mesh node type (in the external graph) connected to `new_domain`.

    Returns
    -------
    tuple[dict, str, set]
        The renamed metadata, the name of the training domain used as a template for `new_domain`,
        and the set of other training domain names dropped during renaming.
    """
    old_dataset_names = list(metadata["metadata_inference"]["dataset_names"])
    template_dataset = old_dataset_names[0]
    other_domains = set(old_dataset_names[1:])

    # "data" is a legacy default dataset name used by single-domain checkpoints but also coincides with common config keys.
    assert template_dataset != "data", (
        "ExternalDomainRunner requires a genuinely multi-domain checkpoint with explicit dataset "
        "names; the training dataset name to replace must not be the generic default name 'data'."
    )

    # TODO: might be better to get this out of the checkpoint graph? [nodes connected to template_dataset]]
    old_hidden_nodes_name = metadata["config"]["model"].get("model", {}).get("hidden_nodes_name")
    assert isinstance(old_hidden_nodes_name, dict), (
        "Expected `hidden_nodes_name` to be a dict keyed by domain name for a multi-domain "
        f"checkpoint, got {old_hidden_nodes_name!r} instead."
    )
    old_hidden_for_template = old_hidden_nodes_name.get(template_dataset, "hidden")

    renames = {template_dataset: new_domain}
    if old_hidden_for_template is not None:
        renames[old_hidden_for_template] = hidden_name

    metadata = _rename(metadata, renames, other_domains)
    assert metadata["metadata_inference"]["dataset_names"] == [new_domain], (
        f"Expected a single dataset '{new_domain}' after renaming, "
        f"got {metadata['metadata_inference']['dataset_names']}."
    )
    assert metadata["config"]["model"]["model"].get("hidden_nodes_name") == {new_domain: hidden_name}, (
        f"Expected hidden_nodes_name to be renamed to {{'{new_domain}': '{hidden_name}'}}, "
        f"got {metadata['config']['model']['model'].get('hidden_nodes_name')}."
    )

    return metadata, template_dataset, other_domains


def _adapt_state_dict(state_dict: dict, renames: dict, other_domains: set) -> dict:
    """Rename names in `renames` wherever they appear as a whole `.`-separated path segment in
    `state_dict` keys (e.g. `pre_processors.<dataset_name>.*`, `model.residual.<dataset_name>.*`).

    Parameters
    ----------
    state_dict : dict
        The model's state dict (not mutated; a new dict is returned).
    renames : dict
        Mapping of old name -> new name, applied to each `.`-separated path segment.
    other_domains : set
        Names of the other training domains to drop entirely.

    Returns
    -------
    dict
        A new state dict with the relevant keys renamed/dropped.
    """
    renamed = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if any(part in other_domains for part in parts):
            continue
        new_key = ".".join(renames.get(part, part) for part in parts)
        renamed[new_key] = value
    return renamed


def _adapt_supporting_arrays(new_domain: str, graph: Any) -> dict:
    """Build the per-dataset `supporting_arrays` entry for `new_domain`.

    Parameters
    ----------
    new_domain : str
        Name of the new domain to introduce.
    graph : Any
        The external graph, with node coordinates (in radians) for `new_domain` at `graph[new_domain].x`.

    Returns
    -------
    dict
        Supporting arrays with a single `new_domain` entry, with up to date `latitudes`/`longitudes`.
    """
    # TODO(dieter): we need to add output_mask, and possible other graph based supporting arrays.
    coords = np.rad2deg(graph[new_domain].x.detach().cpu().numpy()).astype(np.float64)
    return {new_domain: {"latitudes": coords[:, 0], "longitudes": coords[:, 1]}}


@runner_registry.register("external_domain")
@main_argument("graph")
class ExternalDomainRunner(DefaultRunner):
    """Runner where the 'domain' is replaced by an externally provided one."""

    def __init__(self, config: RunConfiguration, graph: str) -> None:
        self.graph_path = graph

        if isinstance(config.input, dict):
            assert len(config.input) == 1, (
                "ExternalDomainRunner currently only supports replacing a single domain, "
                f"but config.input has keys {list(config.input)}."
            )
            new_domain = next(iter(config.input))
        else:
            new_domain = "domain"
        self.new_domain = new_domain

        assert new_domain in self.graph.node_types, (
            f"Domain '{new_domain}' (from config.input) not found in the external graph's node types "
            f"{sorted(self.graph.node_types)}."
        )
        hidden_candidates = {dst for (src, _, dst) in self.graph.edge_types if src == new_domain and dst != new_domain}
        assert len(hidden_candidates) == 1, (
            f"Could not uniquely infer the hidden mesh name for domain '{new_domain}' from the external graph's "
            f"edges. Found candidate destination node types {sorted(hidden_candidates)} for edges starting at "
            f"'{new_domain}'."
        )
        self.hidden_name = hidden_candidates.pop()
        LOG.info("Inferred hidden mesh name '%s' for new domain '%s'.", self.hidden_name, new_domain)

        # Local copies to use inside the nested `_DomainRenamingCheckpoint` class below, where
        # `self` refers to the checkpoint instance rather than this runner.
        hidden_name = self.hidden_name
        graph = self.graph

        class _DomainRenamingCheckpoint(Checkpoint):
            @cached_property
            def _raw_metadata(self) -> tuple[dict, dict]:
                metadata, _ = super()._raw_metadata
                metadata, self.template_dataset, self.other_domains = _adapt_metadata(metadata, new_domain, hidden_name)
                supporting_arrays = _adapt_supporting_arrays(new_domain, graph)
                return metadata, supporting_arrays

        super().__init__(config, classes=RunnerClasses(checkpoint=_DomainRenamingCheckpoint))
        self.template_dataset = self.checkpoint.template_dataset
        self.other_domains = self.checkpoint.other_domains
        self.renames = {self.template_dataset: self.new_domain}

    def _apply_renames(self, obj: Any) -> Any:
        """Shorthand for `_rename(obj, self.renames, self.other_domains)`."""
        return _rename(obj, self.renames, self.other_domains)

    @cached_property
    def graph(self) -> Any:
        graph_path = self.graph_path
        assert os.path.isfile(
            graph_path
        ), f"No graph found at {graph_path}. An external graph needs to be specified in the config file for this runner."
        LOG.info("Loading external graph from path %s.", graph_path)
        return torch.load(graph_path, map_location="cpu", weights_only=False)

    @cached_property
    def model(self) -> Any:
        """Get model adapted to the new domain.

        Rebuild the model using the external graph and the renamed, new-domain model config.
        Then inject model weights from checkpoint.

        """
        device = self.device
        self.device = "cpu"
        model_instance = super().model
        state_dict_ckpt = deepcopy(model_instance.state_dict())

        renamed_state_dict = _adapt_state_dict(state_dict_ckpt, self.renames, self.other_domains)

        # these are needed to rebuild the model, statistics will be overwritten by state_dict though
        model_instance.data_indices = self._apply_renames(model_instance.data_indices)
        model_instance.statistics = self._apply_renames(model_instance.statistics)
        model_instance.statistics_tendencies = self._apply_renames(model_instance.statistics_tendencies)

        # Rebuild the model with the new graph and the already domain-renamed model config.
        model_instance.graph_data = self.graph
        model_instance.config = self.checkpoint._metadata._config
        model_instance._build_model()

        # Reinstate the weights and normalizer statistics from the checkpoint.
        model_instance = update_state_dict(
            model_instance, renamed_state_dict, keywords=["bias", "weight", "processors.normalizer"]
        )

        LOG.info(
            "Successfully built model for new domain '%s' with external graph and reassigned model weights!",
            self.new_domain,
        )
        self.device = device
        return model_instance.to(self.device)
