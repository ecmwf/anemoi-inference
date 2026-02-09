# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from contextlib import AbstractContextManager
from contextlib import contextmanager
from contextlib import nullcontext
from typing import Any
from typing import Callable
from typing import Generator
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

from anemoi.inference.lazy import torch

LOG = logging.getLogger(__name__)

PRE_POST = dict[Literal["pre", "post"], bool]


class RecordOptions(BaseModel):
    """Torch debug mode options to record.

    Includes options straight from `torch.utils._debug_mode.DebugMode`,
    as well as custom options that are implemented using custom
    dispatch hooks in the _DispatchHooks class.

    For those implemented using custom dispatch hooks, the option can be
    set to a boolean to enable/disable the hook, or a dictionary
    with "pre" and "post" keys to specify whether to record the information
    before or after the event.

    .i.e setting
    ```"max_value": {"pre": True, "post": False}```
      will record the maximum value of the inputs before each event, but not the outputs after each event.
    """

    torchfunction: bool = False
    """Record the torch function calls and their arguments."""
    faketensor: bool = False
    """Record on fake tensors"""
    realtensor: bool = True
    """Record on real tensors"""
    localtensor: bool = True
    """Record on local tensors."""
    tensor_attributes: list[str] | None = None
    """Record the specified tensor attributes."""
    nn_module: bool = True
    """Record the nn module calls and their arguments."""
    stack_trace: bool = True
    """Record the stack trace for each recorded event."""
    output: bool = True
    """Records call outputs in logs (e.g. for __torch_dispatch__, __torch_function__, redistribute_input)"""
    ids: bool = True
    """Record the ids of each recorded event."""
    profiler_context: bool = False
    """Record the profiler context for each recorded event."""

    max_value: bool | PRE_POST = False
    """Record the maximum value of each tensor in the inputs/outputs of each event."""
    memory: bool | PRE_POST = False
    """Record the memory usage after each event."""
    nan_inf: bool | PRE_POST = False
    """Record whether any tensor in the inputs or outputs of each event contains NaN or Inf values."""
    is_contiguous: bool | PRE_POST = False
    """Record whether each tensor in the inputs/outputs of each event is contiguous."""

    def to_kwargs(self) -> dict[str, Any]:
        NOT_TORCH_OPTIONS = {"max_value", "memory", "nan_inf", "is_contiguous"}
        if not torch.__version__ >= "2.10.1":
            NOT_TORCH_OPTIONS.add("localtensor")
        dump = self.model_dump()
        return {f"record_{k}": v for k, v in dump.items() if k not in NOT_TORCH_OPTIONS}


class DebugOptions(BaseModel):
    """Options for debugging the model."""

    model_config = ConfigDict(extra="forbid")

    record: RecordOptions = RecordOptions()
    """Recording options for the debug mode."""
    tensor_hashes: bool = False
    """Whether to hash the inputs and outputs of each recorded event."""
    hash_function: str = "norm"
    """The hash function to use when hashing tensors. Can be 'norm' or 'hash_tensor'."""

    output: Literal["print", "log"] | str = "print"
    """Where to output the debug information. Can be 'print', 'log', or a file path."""


def _output_debug_info(info: str, options: DebugOptions) -> None:
    if options.output == "print":
        print(info)
    elif options.output == "log":
        LOG.debug(info)
    else:
        with open(options.output, "w") as f:
            f.write(info + "\n")


class _DispatchHooks(AbstractContextManager):
    """Custom dispatch hooks to record additional information not captured by the default debug mode options."""

    def __init__(self, options: DebugOptions):
        self.options = options

    @staticmethod
    def _cast_tensor(t: torch.Tensor) -> torch.Tensor:
        """Cast a tensor to a floating point type for the debug checks."""
        if not (t.is_floating_point() or t.is_complex()):
            return t.float()
        t = t.contiguous()
        if t.is_complex():
            t_float = t.to(dtype=torch.complex128)
        else:
            cast_dtype = torch.float64
            if t.device.type == "mps":
                cast_dtype = torch.float32  # mps doesn't support float64
            t_float = t.to(dtype=cast_dtype)
        return t_float

    @staticmethod
    def any_nan_or_inf(t: torch.Tensor) -> bool:
        """Check if a tensor contains any NaN or Inf values."""
        t = _DispatchHooks._cast_tensor(t)
        return bool(t.isnan().any().item() or t.isinf().any().item())

    @staticmethod
    def max_value(t: torch.Tensor) -> float | None:
        """Get the maximum absolute value of a tensor."""
        t = _DispatchHooks._cast_tensor(t)
        if t.numel() == 0:
            return None
        return t.abs().max().item()

    @staticmethod
    def memory_usage(t: torch.Tensor) -> dict:
        """Get the current and peak memory usage for the device of the tensor."""
        MB = 1024 * 1024.0
        mem = 0.0
        peak = 0.0

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / MB
            peak = torch.cuda.max_memory_allocated() / MB
            torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch.mps, "is_available") and torch.mps.is_available():
            mem = torch.mps.current_allocated_memory() / MB
            peak = torch.mps.driver_allocated_memory() / MB

        return {"mem": f"{mem:.3f} MB", "peak": f"{peak:.3f} MB"}

    @staticmethod
    def _dispatch_handler(
        fn: Callable[[torch.Tensor], Any], *, step: Literal["pre", "post"] = "post", name: str | None = None
    ) -> Callable[[Callable, Any, Any, Any, Any], dict | None]:
        """Map a function that takes a tensor and returns a dictionary to a dispatch hook that applies the function to all tensors in the inputs or outputs of a recorded event."""
        from torch.utils._pytree import tree_map

        name = name or fn.__name__

        def pre_hook(func: Callable, types, args, kwargs, call) -> dict | None:
            if "empty" in str(func) or "profiler" in str(func):
                return None

            return {name: tree_map(lambda x: fn(x) if isinstance(x, torch.Tensor) else None, (args, kwargs))}

        def post_hook(func: Callable, types, args, kwargs, result) -> dict | None:
            if "empty" in str(func) or "profiler" in str(func):
                return None
            return {name: tree_map(lambda x: fn(x) if isinstance(x, torch.Tensor) else None, result)}

        if step == "pre":
            return pre_hook
        return post_hook

    FUNC_MAP = {
        "max_value": max_value,
        "memory": memory_usage,
        "nan_inf": any_nan_or_inf,
        "contiguous": lambda t: t.is_contiguous(),
    }

    def get_hooks(self) -> list[AbstractContextManager]:
        """Get the list of dispatch hooks to use based on the options."""

        from torch.utils._debug_mode import DebugMode

        dispatch_hooks = []

        for option_name, log_hook in self.FUNC_MAP.items():
            if option := getattr(self.options.record, option_name, False):
                if isinstance(option, bool):
                    dispatch_hooks.append(DebugMode.dispatch_hooks(log_hook=log_hook))
                elif isinstance(option, dict):
                    for step, enabled in option.items():
                        if enabled:
                            dispatch_hooks.append(
                                DebugMode.dispatch_hooks(
                                    log_hook=self._dispatch_handler(log_hook, step=step, name=f"{option_name}_{step}")
                                )
                            )
                else:
                    raise TypeError(
                        f"Invalid type for option {option_name}: {type(option)}, expected bool or dict[str, bool]"
                    )
        return dispatch_hooks

    def __enter__(self) -> Any:
        self.hooks = self.get_hooks()
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for hook in self.hooks:
            hook.__exit__(exc_type, exc_value, traceback)


@contextmanager
def debug_torch(options: bool | dict[str, Any] | DebugOptions) -> Generator[None, None, None]:
    """Context manager to run PyTorch in debug mode with the specified options.

    Parameters
    ----------
    options: bool | dict[str, Any] | DebugOptions
        If False, the context manager does nothing.
        If True, the context manager runs PyTorch in debug mode with default options.
        If a DebugOptions object is provided, the context manager runs PyTorch in debug mode with the specified options.
        If a dictionary is provided, it is used to create a DebugOptions object.

    Yields
    ------
    None

    Example
    -------
    >>> with debug_torch(True):
    ...     # PyTorch code here will run in debug mode with default options.
    >>> with debug_torch({"record": {"torchfunction": True, "max_value": {"pre": True, "post": False}}}):
    ...     # PyTorch code here will run in debug mode with torch function calls recorded and the maximum value of inputs recorded before each event.
    """
    if isinstance(options, bool):
        if not options:
            yield
            return
        else:
            options = DebugOptions()

    if torch.__version__ < "2.10":
        raise RuntimeError("Debug mode requires PyTorch 2.10 or higher.")

    if isinstance(options, dict):
        options = DebugOptions(**options)

    from torch.utils._debug_mode import DebugMode

    log_hashes = (
        DebugMode.log_tensor_hashes(hash_fn=options.hash_function, hash_inputs=True)
        if options.tensor_hashes
        else nullcontext()
    )
    dispatch_hooks = _DispatchHooks(options)

    with DebugMode(**options.record.to_kwargs()) as dm, log_hashes, dispatch_hooks:
        try:
            yield
        finally:
            _output_debug_info(dm.debug_string(), options)
