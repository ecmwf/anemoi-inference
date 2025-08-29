# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Literal

import earthkit.data as ekd
from anemoi.utils.checkpoints import load_metadata
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.forcings import Forcings
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.inference.types import State

from .metadata import Metadata
from .metadata import Variable

LOG = logging.getLogger(__name__)


def _download_huggingfacehub(huggingface_config: Any) -> str:
    """Download model from huggingface.

    Parameters
    ----------
    huggingface_config : dict or str
        Configuration for downloading from huggingface.

    Returns
    -------
    str
        Path to the downloaded model.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Could not import `huggingface_hub`, please run `pip install huggingface_hub`.") from e

    if isinstance(huggingface_config, str):
        huggingface_config = {"repo_id": huggingface_config}

    if "filename" in huggingface_config:
        return str(hf_hub_download(**huggingface_config))

    repo_path = Path(snapshot_download(**huggingface_config))
    ckpt_files = list(repo_path.glob("*.ckpt"))

    if len(ckpt_files) == 1:
        return str(ckpt_files[0])
    else:
        raise ValueError(
            f"None or Multiple ckpt files found in repo, {ckpt_files}.\nCannot pick one to load, please specify `filename`."
        )


class Checkpoint:
    """Represents an inference checkpoint."""

    def __init__(
        self,
        source: str | Metadata | dict[str, Any],
        *,
        patch_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Checkpoint.

        Parameters
        ----------
        path : str
            The path to the checkpoint.
        patch_metadata : Optional[Dict[str, Any]], optional
            Metadata to patch the checkpoint with, by default None.
        """
        self._source = source
        self.patch_metadata = patch_metadata

    def __repr__(self) -> str:
        """Represent the Checkpoint as a string.

        Returns
        -------
        str
            String representation of the Checkpoint.
        """
        return f"Checkpoint({self.path})"

    @cached_property
    def path(self) -> str:
        """Get the path to the checkpoint."""
        import json

        try:
            path = json.loads(self._source)
        except Exception:
            path = self._source

        if isinstance(path, (Path, str)):
            return str(path)
        elif isinstance(path, dict):
            if "huggingface" in path:
                return _download_huggingfacehub(path["huggingface"])
            pass
        raise TypeError(f"Cannot parse model path: {path}. It must be a path or dict")

    @cached_property
    def _metadata(self) -> Metadata:
        """Get the metadata."""

        if isinstance(self._source, Metadata):
            return self._source

        try:
            result = Metadata(*load_metadata(self.path, supporting_arrays=True))
        except Exception as e:
            LOG.warning("Version does not support `supporting_arrays` (%s)", e)
            result = Metadata(load_metadata(self.path))

        if self.patch_metadata:
            LOG.warning("Patching metadata with %r", self.patch_metadata)
            result.patch(self.patch_metadata)

        return result

    ###########################################################################
    # Forwards used by the runner
    # We do not want to expose the metadata object directly
    # We do not use `getattr` to avoid exposing all methods and make debugging
    # easier
    ###########################################################################

    @property
    def timestep(self) -> Any:
        """Get the timestep."""
        return self._metadata.timestep

    @property
    def input_explicit_times(self) -> Any:
        """Get the input explicit times from metadata."""
        return self._metadata.input_explicit_times

    @property
    def target_explicit_times(self) -> Any:
        """Get the target explicit times."""
        return self._metadata.target_explicit_times

    @property
    def data_frequency(self) -> Any:
        """Get the data frequency."""
        return self._metadata._config_data.frequency

    @property
    def precision(self) -> Any:
        """Get the precision."""
        return self._metadata.precision

    @property
    def number_of_grid_points(self) -> Any:
        """Get the number of grid points."""
        return self._metadata.number_of_grid_points

    @property
    def number_of_input_features(self) -> Any:
        """Get the number of input features."""
        return self._metadata.number_of_input_features

    @property
    def variable_to_input_tensor_index(self) -> Any:
        """Get the variable to input tensor index."""
        return self._metadata.variable_to_input_tensor_index

    @property
    def model_computed_variables(self) -> Any:
        """Get the model computed variables."""
        return self._metadata.model_computed_variables

    @property
    def typed_variables(self) -> dict[str, Variable]:
        """Get the typed variables."""
        return self._metadata.typed_variables

    @property
    def diagnostic_variables(self) -> Any:
        """Get the diagnostic variables."""
        return self._metadata.diagnostic_variables

    @property
    def prognostic_variables(self) -> Any:
        """Get the prognostic variables."""
        return self._metadata.prognostic_variables

    @property
    def prognostic_output_mask(self) -> Any:
        """Get the prognostic output mask."""
        return self._metadata.prognostic_output_mask

    @property
    def prognostic_input_mask(self) -> Any:
        """Get the prognostic input mask."""
        return self._metadata.prognostic_input_mask

    @property
    def output_tensor_index_to_variable(self) -> Any:
        """Get the output tensor index to variable."""
        return self._metadata.output_tensor_index_to_variable

    @property
    def accumulations(self) -> Any:
        """Get the accumulations."""
        return self._metadata.accumulations

    @property
    def latitudes(self) -> Any:
        """Get the latitudes."""
        return self._metadata.latitudes

    @property
    def longitudes(self) -> Any:
        """Get the longitudes."""
        return self._metadata.longitudes

    @property
    def grid_points_mask(self) -> Any:
        """Get the grid points mask."""
        return self._metadata.grid_points_mask

    @cached_property
    def sources(self) -> list["SourceCheckpoint"]:
        """Get the sources."""
        return [SourceCheckpoint(self, _) for _ in self._metadata.sources(self.path)]

    def default_namer(self, *args: Any, **kwargs: Any) -> Callable[[ekd.Field, Any], str]:
        """Return a callable that can be used to name fields.

        Parameters
        ----------
        *args : Any
            Additional arguments.

        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Callable
            The namer that was used to create the training dataset.
        """
        return self._metadata.default_namer(*args, **kwargs)

    def report_error(self) -> None:
        """Report an error."""
        self._metadata.report_error()

    def validate_environment(
        self,
        *,
        all_packages: bool = False,
        on_difference: Literal["warn", "error", "ignore", "return"] = "warn",
        exempt_packages: list[str] | None = None,
    ) -> bool | str:
        """Validate the environment.

        Parameters
        ----------
        all_packages : bool, optional
            Check all packages in the environment (True) or just anemoi's (False), by default False.
        on_difference : Literal['warn', 'error', 'ignore', 'return'], optional
            What to do on difference, by default "warn"
        exempt_packages : list[str], optional
            List of packages to exempt from the check, by default EXEMPT_PACKAGES

        Returns
        -------
        Union[bool, str]
            boolean if `on_difference` is not 'return', otherwise formatted text of the differences
            True if environment is valid, False otherwise
        """
        return self._metadata.validate_environment(
            all_packages=all_packages, on_difference=on_difference, exempt_packages=exempt_packages
        )

    def open_dataset(
        self,
        *,
        use_original_paths: bool | None = None,
        from_dataloader: Any | None = None,
    ) -> Any:
        """Open the dataset.

        Parameters
        ----------
        use_original_paths : bool, optional
            Whether to use the original paths, by default None.
        from_dataloader : Any, optional
            The dataloader to use, by default None.

        Returns
        -------
        Any
            The opened dataset.
        """
        return self._metadata.open_dataset(use_original_paths=use_original_paths, from_dataloader=from_dataloader)

    def open_dataset_args_kwargs(
        self, *, use_original_paths: bool, from_dataloader: Any | None = None
    ) -> tuple[Any, Any]:
        """Get arguments and keyword arguments for opening the dataset.

        Parameters
        ----------
        use_original_paths : bool
            Whether to use original paths.
        from_dataloader : Optional[Any], optional
            Data loader, by default None.

        Returns
        -------
        Tuple[Any, Any]
            Arguments and keyword arguments for opening the dataset.
        """
        return self._metadata.open_dataset_args_kwargs(
            use_original_paths=use_original_paths,
            from_dataloader=from_dataloader,
        )

    def constant_forcings_inputs(self, runner: Any, input_state: State) -> list[Forcings]:
        """Get constant forcings inputs.

        Parameters
        ----------
        runner : Any
            The runner.
        input_state : State
            The input state.

        Returns
        -------
        List[Forcings]
            The constant forcings inputs.
        """
        return self._metadata.constant_forcings_inputs(runner, input_state)

    def dynamic_forcings_inputs(self, runner: Any, input_state: State) -> list[Forcings]:
        """Get dynamic forcings inputs.

        Parameters
        ----------
        runner : Any
            The runner.
        input_state : State
            The input state.

        Returns
        -------
        List[Forcings]
            The dynamic forcings inputs.
        """
        return self._metadata.dynamic_forcings_inputs(runner, input_state)

    def boundary_forcings_inputs(self, runner: Any, input_state: State) -> list[Forcings]:
        """Get boundary forcings inputs.

        Parameters
        ----------
        runner : Any
            The runner.
        input_state : State
            The input state.

        Returns
        -------
        List[Forcings]
            The boundary forcings inputs.
        """
        return self._metadata.boundary_forcings_inputs(runner, input_state)

    def name_fields(self, fields: Any, namer: Callable[..., str] | None = None) -> Any:
        """Name fields.

        Parameters
        ----------
        fields : Any
            The fields to name.
        namer : Optional[Callable[...,str]], optional
            The namer, by default None.

        Returns
        -------
        Any
            The named fields.
        """
        return self._metadata.name_fields(fields, namer=namer)

    def sort_by_name(
        self, fields: ekd.FieldList, *args: Any, namer: Callable[..., str] | None = None, **kwargs: Any
    ) -> ekd.FieldList:
        """Sort fields by name.

        Parameters
        ----------
        fields : ekd.FieldList
            The fields to sort.
        *args : Any
            Additional arguments.
        namer : Optional[Callable[...,str]], optional
            The namer, by default None.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ekd.FieldList
            The sorted fields.
        """
        return self._metadata.sort_by_name(fields, *args, namer=namer, **kwargs)

    def print_indices(self) -> None:
        """Print the indices."""
        return self._metadata.print_indices()

    def variable_categories(self) -> Any:
        """Get the variable categories.

        Returns
        -------
        Any
            The variable categories.
        """
        return self._metadata.variable_categories()

    def load_supporting_array(self, name: str) -> Any:
        """Load a supporting array.

        Parameters
        ----------
        name : str
            The name of the supporting array.

        Returns
        -------
        Any
            The supporting array.
        """
        return self._metadata.load_supporting_array(name)

    @property
    def supporting_arrays(self) -> Any:
        """Get the supporting arrays."""
        return self._metadata.supporting_arrays

    ###########################################################################

    @cached_property
    def lagged(self) -> list[datetime.timedelta]:
        """Return the list of steps for the `multi_step_input` fields."""
        result = list(range(0, self._metadata.multi_step_input))

        result = [-s * self._metadata.timestep for s in result]
        return sorted(result)

    @property
    def multi_step_input(self) -> Any:
        """Get the multi-step input."""
        return self._metadata.multi_step_input

    def print_variable_categories(self) -> None:
        """Print the variable categories."""
        return self._metadata.print_variable_categories()

    ###########################################################################
    # Data retrieval
    ###########################################################################

    def variables_from_input(self, *, include_forcings: bool) -> Any:
        """Get variables from input.

        Parameters
        ----------
        include_forcings : bool
            Whether to include forcings.

        Returns
        -------
        Any
            The variables from input.
        """
        return self._metadata.variables_from_input(include_forcings=include_forcings)

    @property
    def grid(self) -> Any:
        """Get the grid."""
        return self._metadata.grid

    @property
    def area(self) -> Any:
        """Get the area."""
        return self._metadata.area

    def mars_by_levtype(self, levtype: str) -> Any:
        """Get MARS requests by level type.

        Parameters
        ----------
        levtype : str
            The level type.

        Returns
        -------
        Any
            The MARS requests by level type.
        """
        return self._metadata.mars_by_levtype(levtype)

    def mars_requests(
        self,
        *,
        variables: list[str],
        dates: list[Date],
        use_grib_paramid: bool = False,
        always_split_time: bool = False,
        patch_request: Callable[[DataRequest], DataRequest] | None = None,
        **kwargs: Any,
    ) -> list[DataRequest]:
        """Generate MARS requests for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            The list of variables.
        dates : List[Any]
            The list of dates.
        use_grib_paramid : bool, optional
            Whether to use GRIB paramid, by default False.
        always_split_time : bool, optional
            Whether to always split time, by default False.
        patch_request : Optional[Callable], optional
            A callable to patch the request, by default None.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        List[DataRequest]
            The list of MARS requests.
        """
        from anemoi.utils.grib import shortname_to_paramid
        from earthkit.data.utils.availability import Availability

        assert variables, "No variables provided"

        if not isinstance(dates, (list, tuple)):
            dates = [dates]

        dates = [to_datetime(d) for d in dates]

        assert dates, "No dates provided"

        result: list[DataRequest] = []

        DEFAULT_KEYS = ("class", "expver", "type", "stream", "levtype")
        DEFAULT_KEYS_AND_TIME = ("class", "expver", "type", "stream", "levtype", "time")

        # ECMWF operational data has stream oper for 00 and 12 UTC and scda for 06 and 18 UTC
        # The split oper/scda is a bit special
        KEYS = {("oper", "fc"): DEFAULT_KEYS_AND_TIME, ("scda", "fc"): DEFAULT_KEYS_AND_TIME}

        requests = defaultdict(list)

        for r in self._metadata.mars_requests(variables=variables):
            for date in dates:

                r = r.copy()

                base = date
                step = str(r.get("step", 0)).split("-")[-1]
                step = int(step)
                base = base - datetime.timedelta(hours=step)

                r["date"] = base.strftime("%Y-%m-%d")
                r["time"] = base.strftime("%H%M")

                r.update(kwargs)  # We do it here so that the Availability can use that information

                if always_split_time:
                    keys = DEFAULT_KEYS_AND_TIME
                else:
                    keys = KEYS.get((r.get("stream"), r.get("type")), DEFAULT_KEYS)
                key = tuple(r.get(k) for k in keys)

                # Special case because of oper/scda

                requests[key].append(r)

        result = []
        for reqs in requests.values():

            compressed = Availability(reqs)
            for r in compressed.iterate():

                if not r:
                    continue

                changed = True
                while changed:
                    changed = False
                    for k, v in r.items():
                        if isinstance(v, tuple):
                            r[k] = list(v)
                            changed = True

                        if isinstance(v, (list, tuple)) and len(v) == 1:
                            r[k] = v[0]
                            changed = True

                # Convert all to lists
                for k, v in r.items():
                    if not isinstance(v, list):
                        r[k] = [v]

                # Patch BEFORE the shortname to paramid

                if patch_request:
                    r = patch_request(r)

                # Convert all to lists (again)
                for k, v in r.items():
                    if isinstance(v, tuple):
                        r[k] = list(*v)
                    if not isinstance(v, list):
                        r[k] = [v]

                if use_grib_paramid and "param" in r:
                    r["param"] = [shortname_to_paramid(_) for _ in r["param"]]

                # Simplify the request

                for k in list(r.keys()):
                    v = r[k]
                    if len(v) == 1:
                        v = v[0]

                    # Remove empty values for when tree is not fully defined
                    if v == "-":
                        r.pop(k)
                        continue
                    r[k] = v
                result.append(r)

        return result

    ###########################################################################
    # supporting arrays
    ###########################################################################

    @cached_property
    def _supporting_arrays(self) -> Any:
        """Get the supporting arrays."""
        return self._metadata._supporting_arrays

    @property
    def name(self) -> Any:
        """Get the name."""
        return self._metadata.name

    ###########################################################################
    # Misc
    ###########################################################################

    def provenance_training(self) -> Any:
        """Get the provenance of the training.

        Returns
        -------
        Any
            The provenance of the training.
        """
        return self._metadata.provenance_training()


class SourceCheckpoint(Checkpoint):
    """A checkpoint that represents a source."""

    def __init__(self, owner: Checkpoint, metadata: Any) -> None:
        """Initialize the SourceCheckpoint.

        Parameters
        ----------
        owner : Checkpoint
            The owner checkpoint.
        metadata : Any
            The metadata for the source checkpoint.
        """
        super().__init__(owner.path)
        self._owner = owner
        self._metadata = metadata

    def __repr__(self) -> str:
        """Represent the SourceCheckpoint as a string.

        Returns
        -------
        str
            String representation of the SourceCheckpoint.
        """
        return f"Source({self.name}@{self.path})"
