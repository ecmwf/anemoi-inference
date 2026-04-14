# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from typing import Any
from typing import Generator

import numpy as np

from ..config.run import RunConfiguration
from ..runners import create_runner
from ..types import State
from . import Command

LOG = logging.getLogger(__name__)


class InputComparisonConfiguration(RunConfiguration):
    """Subclass of RunConfiguration for input comparison, allowing for multiple inputs and dates."""

    input: list[str | dict[str, Any]]  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    date: datetime | dict[str, datetime] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    def to_datetime(cls, date: str | int | datetime | dict | None) -> datetime | None:
        if isinstance(date, dict):
            return {key: RunConfiguration.to_datetime(value) for key, value in date.items()}  # type: ignore
        return RunConfiguration.to_datetime(date)


def _plot_field_differences(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, name: str) -> None:
    """Plot the differences between fields using earthkit-plots, if available."""
    try:
        import earthkit.plots as ekp
    except ImportError:
        LOG.error("earthkit-plots is not installed. Skipping plot generation.")
        return

    chart = ekp.Map()

    z = field
    x = np.where(lon >= 180, lon - 360, lon)
    y = lat

    chart.contourf(z=z, x=x, y=y, interpolate=dict(distance_threshold=1))

    chart.coastlines()
    chart.gridlines()
    chart.legend()

    chart.title(name)
    chart.save(f"{name}.png")


def _check_field_coverage(states: list[State]) -> tuple[bool, set[str]]:
    """Check that all states have the same fields and return the common fields."""
    field_name_sets = [set(state["fields"].keys()) for state in states]
    common_fields = set.intersection(*field_name_sets)
    all_fields = set.union(*field_name_sets)
    consistent = True
    for field in sorted(all_fields - common_fields):
        LOG.warning("Field '%s' is not present in all states.", field)
        consistent = False
    return consistent, common_fields


def _check_nans(field: str, arrays: list[np.ndarray], states: list[State], plot_difference: bool) -> bool:
    """Check that the NaN positions are consistent across states."""
    nan_masks = [np.isnan(arr) for arr in arrays]
    nan_counts = [int(mask.sum()) for mask in nan_masks]
    consistent = True

    if len(set(nan_counts)) > 1:
        LOG.warning("Field '%s': NaN counts differ across states: %s.", field, nan_counts)
        consistent = False

    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) == 1:
        for i in range(1, len(nan_masks)):
            if not np.array_equal(nan_masks[0], nan_masks[i]):
                LOG.warning("Field '%s': NaN positions differ between state 0 and state %d.", field, i)
                consistent = False
                if plot_difference:
                    _plot_field_differences(
                        (nan_masks[0] != nan_masks[i])[0],
                        states[0]["latitudes"],
                        states[0]["longitudes"],
                        f"{field}_nan_mask_state0_vs_state{i}",
                    )
    else:
        LOG.warning("Field '%s': array shapes differ across states (%s); cannot compare NaN positions.", field, shapes)

    return consistent


def _check_exact(field: str, arrays: list[np.ndarray], states: list[State], plot_difference: bool) -> bool:
    """Check that the arrays are exactly equal across states."""
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) != 1:
        LOG.warning("Field '%s': array shapes differ across states (%s); cannot do exact comparison.", field, shapes)
        return False

    consistent = True
    for i in range(1, len(arrays)):
        if not np.array_equal(arrays[0], arrays[i], equal_nan=True):
            LOG.warning("Field '%s': values differ exactly between state 0 and state %d.", field, i)
            consistent = False
            if plot_difference:
                _plot_field_differences(
                    (arrays[0] - arrays[i])[0],
                    states[0]["latitudes"],
                    states[0]["longitudes"],
                    f"{field}_exact_difference_state0_vs_state{i}",
                )
    return consistent


def _check_statistics(
    field: str,
    arrays: list[np.ndarray],
    states: list[State],
    std_relative_threshold: float,
    range_relative_threshold: float,
    plot_difference: bool,
) -> bool:
    """Check that the statistics of the arrays are consistent across states."""
    means = [float(np.nanmean(arr)) for arr in arrays]
    stds = [float(np.nanstd(arr)) for arr in arrays]
    ranges = [float(np.nanmax(arr) - np.nanmin(arr)) for arr in arrays]
    consistent = True

    for i in range(1, len(arrays)):
        std_tolerance = max(stds[0], stds[i])
        if std_tolerance > 0 and abs(means[0] - means[i]) > std_tolerance:
            LOG.warning(
                "Field '%s': mean of state %d (%.4g) lies outside one std (%.4g) of state 0 mean (%.4g).",
                field,
                i,
                means[i],
                std_tolerance,
                means[0],
            )
            if plot_difference:
                _plot_field_differences(
                    (arrays[0] - arrays[i])[0],
                    states[0]["latitudes"],
                    states[0]["longitudes"],
                    f"{field}_mean_difference_state0_vs_state{i}",
                )
            consistent = False

        std_max = max(stds[0], stds[i])
        if std_max > 0 and abs(stds[0] - stds[i]) / std_max > std_relative_threshold:
            LOG.warning(
                "Field '%s': std values differ by more than %.0f%% between state 0 (%.4g) and state %d (%.4g).",
                field,
                std_relative_threshold * 100,
                stds[0],
                i,
                stds[i],
            )
            if plot_difference:
                _plot_field_differences(
                    (arrays[0] - arrays[i])[0],
                    states[0]["latitudes"],
                    states[0]["longitudes"],
                    f"{field}_std_difference_state0_vs_state{i}",
                )
            consistent = False

        range_max = max(ranges[0], ranges[i])
        if range_max > 0 and abs(ranges[0] - ranges[i]) / range_max > range_relative_threshold:
            LOG.warning(
                "Field '%s': range values differ by more than %.0f%% between state 0 (%.4g) and state %d (%.4g).",
                field,
                range_relative_threshold * 100,
                ranges[0],
                i,
                ranges[i],
            )
            if plot_difference:
                _plot_field_differences(
                    (arrays[0] - arrays[i])[0],
                    states[0]["latitudes"],
                    states[0]["longitudes"],
                    f"{field}_range_difference_state0_vs_state{i}",
                )
            consistent = False

    return consistent


def compare_states(
    *states: State,
    std_relative_threshold: float = 0.5,
    range_relative_threshold: float = 0.5,
    exact: bool = False,
    plot_difference: bool = False,
) -> bool:
    """Compare multiple states.

    It is not expected that the states are even of the same date, which may result in
    some red herrings.
    This will only check the fields within the state, and not the metadata. The checks are as follows:

    1. Position and count of nans (always).
    2. If ``exact=False`` (default):

       a. Mean: the mean of each state must lie within one standard deviation of the other.
          Primarily to check that the units are correct.
       b. Standard deviation: ``|std_a - std_b| / max(std_a, std_b) < std_relative_threshold``.
       c. Range (max - min): ``|range_a - range_b| / max(range_a, range_b) < range_relative_threshold``.

    3. If ``exact=True``: element-wise equality (NaNs treated as equal). Only valid when
       states have the same shape; most meaningful when states are from the same date.

    Parameters
    ----------
    *states : State
        The states to compare.
    std_relative_threshold : float, optional
        Maximum allowed relative difference between std values across states,
        in the range ``[0, 1]``. Defaults to 0.5.
    range_relative_threshold : float, optional
        Maximum allowed relative difference between range (max - min) values across states,
        in the range ``[0, 1]``. Defaults to 0.5.
    exact : bool, optional
        If True, perform element-wise exact comparison instead of statistical checks.
        Defaults to False.
    plot_difference : bool, optional
        Whether to plot the differences between states using earthkit-plots. Requires earthkit-plots to be installed. Defaults to False.

    Returns
    -------
    bool
        True if all states are consistent, False otherwise.
    """
    if len(states) < 2:
        LOG.error("compare_states called with fewer than 2 states; nothing to compare.")
        return True

    coverage_ok, common_fields = _check_field_coverage(list(states))
    consistent = coverage_ok

    for field in sorted(common_fields):
        arrays = [state["fields"][field] for state in states]
        consistent &= _check_nans(field, arrays, list(states), plot_difference)
        if exact:
            consistent &= _check_exact(field, arrays, list(states), plot_difference)
        else:
            consistent &= _check_statistics(
                field, arrays, list(states), std_relative_threshold, range_relative_threshold, plot_difference
            )

    return consistent


class InputComparison(Command):
    """Compare multiple input states for consistency."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("--defaults", action="append", help="Sources of default values.")
        command_parser.add_argument(
            "config",
            help="Path to config file. Can be omitted to pass config with overrides and defaults.",
        )
        command_parser.add_argument("overrides", nargs="*", help="Overrides as key=value")
        command_parser.add_argument(
            "--std-relative-threshold",
            type=float,
            default=0.5,
            help="Maximum relative difference between std values across states (0-1) before flagging a units mismatch. Default: 0.5.",
        )
        command_parser.add_argument(
            "--range-relative-threshold",
            type=float,
            default=0.5,
            help="Maximum relative difference between range (max - min) values across states (0-1) before flagging a units mismatch. Default: 0.5.",
        )
        command_parser.add_argument(
            "--exact",
            action="store_true",
            help="Perform element-wise exact comparison instead of statistical checks. Most meaningful when states are from the same date.",
        )
        command_parser.add_argument(
            "--plot-differences",
            action="store_true",
            help="Whether to plot the differences between states using earthkit-plots. Requires earthkit-plots to be installed. Default: False.",
        )

    def _iterate_configs(self, config: InputComparisonConfiguration) -> Generator[RunConfiguration, None, None]:
        """Iterate over the input configurations in the input comparison configuration.

        Parameters
        ----------
        config : InputComparisonConfiguration
            The input comparison configuration.

        Returns
        -------
        Generator[RunConfiguration, None, None]
            The input configurations in the input comparison configuration.
        """
        for input in config.input:
            if isinstance(config.date, dict):
                date = config.date.get(input if isinstance(input, str) else next(iter(input.keys())), None)
            else:
                date = config.date
            run_config = config.model_dump()
            run_config["date"] = date
            run_config["input"] = input

            yield RunConfiguration(**run_config)

    def get_states(self, config: InputComparisonConfiguration) -> Generator[State, None, None]:
        """Get the states from the input comparison configuration.

        Parameters
        ----------
        config : InputComparisonConfiguration
            The input comparison configuration.

        Returns
        -------
        Generator[State, None, None]

            The states from the input comparison configuration.
        """
        for run_config in self._iterate_configs(config):
            runner = create_runner(run_config)

            # Taken from runners/default.py
            # TODO: this should be refactored to avoid code duplication, but for now it is easier to just copy the relevant code
            prognostic_input = runner.create_prognostics_input()
            LOG.info(f"📥 Prognostic input: {prognostic_input}")
            prognostic_state = prognostic_input.create_input_state(date=runner.config.date)

            constants_input = runner.create_constant_coupled_forcings_input()
            LOG.info(f"📥 Constant forcings input: {constants_input}")
            constants_state = constants_input.create_input_state(date=runner.config.date)

            forcings_input = runner.create_dynamic_forcings_input()
            LOG.info(f"📥 Dynamic forcings input: {forcings_input}")
            forcings_state = forcings_input.create_input_state(date=runner.config.date)

            input_state = runner._combine_states(
                prognostic_state,
                constants_state,
                forcings_state,
            )
            yield input_state

    def run(self, args: Namespace) -> None:
        """Run the input comparison command.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        if "=" in args.config:
            args.overrides.append(args.config)
            args.config = {}

        config = InputComparisonConfiguration.load(
            args.config,
            args.overrides,
            defaults=args.defaults,
        )
        comparison_result = compare_states(
            *self.get_states(config),
            std_relative_threshold=args.std_relative_threshold,
            range_relative_threshold=args.range_relative_threshold,
            exact=args.exact,
            plot_difference=args.plot_differences,
        )

        if comparison_result:
            LOG.info("✅ The input states are consistent.")
        else:
            LOG.warning("❌ The input states are not consistent. ❌")


command = InputComparison
