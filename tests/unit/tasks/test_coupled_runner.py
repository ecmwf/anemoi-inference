# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Any
from unittest.mock import patch

import numpy as np

from anemoi.inference.config.couple import CoupleConfiguration
from anemoi.inference.task import Task
from anemoi.inference.tasks import create_task
from anemoi.inference.tasks.runner import CoupledRunner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing import files_for_tests
from anemoi.inference.testing import float_hash
from anemoi.inference.transports import create_transport
from anemoi.inference.transports.threads import ThreadsTransport
from anemoi.inference.types import State

# The atmos and ocean test checkpoints have a 24h timestep, so 72h gives three steps and two exchanges.
LEAD_TIME = "72h"
NUMBER_OF_EXCHANGES = 2

# Constants are sent from the initial state, so they are not filled by the no-model runner.
CONSTANTS = {"lsm"}


@fake_checkpoints
def test_coupled_runner_execute_no_model() -> None:
    config = CoupleConfiguration.load(files_for_tests("unit/configs/coupled.yaml"))
    global_config = {"runner": {"no-model": {"output_mode": "hash"}}, "lead_time": LEAD_TIME}

    tasks = {name: create_task(name, action, global_config=global_config) for name, action in config.tasks.items()}
    transport = create_transport(config.transport, config.couplings, tasks)

    sent: dict[tuple[str, str, int], State] = {}
    received: set[tuple[str, str, int]] = set()
    inputs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    original_send = ThreadsTransport.send
    original_receive = ThreadsTransport.receive
    original_predict_step = CoupledRunner.predict_step

    # Record what was passed into ThreadsTransport.send and ThreadsTransport.receive so the test can validate
    def recording_send(self: ThreadsTransport, sender: Task, target: Task, state: State, tag: int) -> None:
        sent[sender.name, target.name, tag] = state
        original_send(self, sender, target, state, tag)

    def recording_receive(self: ThreadsTransport, receiver: Task, source: Task, tag: int) -> State:
        state = original_receive(self, receiver, source, tag)
        received.add((source.name, receiver.name, tag))
        return state

    # Record the last input step of each model call, keyed by task and forecast step
    def recording_predict_step(self: CoupledRunner, model: Any, input_tensors_torch: dict, **kwargs: Any) -> dict:
        ((dataset, tensor),) = input_tensors_torch.items()
        index = self.checkpoint.multi_dataset_metadata[dataset].variable_to_input_tensor_index
        inputs[self.coupled_input.task.name, kwargs["fcstep"]] = {
            variable: tensor[0, -1, :, i].cpu().numpy().copy() for variable, i in index.items()
        }
        return original_predict_step(self, model, input_tensors_torch, **kwargs)

    with (
        patch.object(ThreadsTransport, "send", recording_send),
        patch.object(ThreadsTransport, "receive", recording_receive),
        patch.object(CoupledRunner, "predict_step", recording_predict_step),
    ):
        transport.start()
        transport.wait()

    expected_variables = {}
    for coupling in config.couplings:
        ((direction, variables),) = coupling.items()
        source, target = (name.strip() for name in direction.split("->"))
        expected_variables[source, target] = set(variables)

    # Confirm that config matches what was actually called
    expected_messages = {(*pair, tag) for pair in expected_variables for tag in range(NUMBER_OF_EXCHANGES)}
    assert set(sent) == expected_messages
    assert received == expected_messages

    for (sender, target, tag), state in sent.items():
        assert set(state["fields"]) == expected_variables[sender, target]
        assert state["date"] == datetime.datetime(2022, 1, 3 + tag)

        # A state sent at tag n is used as forcing in the input of the target's forecast step n + 1
        for variable in expected_variables[sender, target] - CONSTANTS:
            expected = float_hash(variable, state["date"])
            message = f"{sender} -> {target}: {variable}"
            np.testing.assert_allclose(state["fields"][variable], expected, rtol=1e-6, err_msg=message)
            np.testing.assert_allclose(inputs[target, tag + 1][variable], expected, rtol=1e-6, err_msg=message)
