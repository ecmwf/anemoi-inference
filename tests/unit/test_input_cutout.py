import numpy as np

from anemoi.inference.inputs.cutout import _mask_and_combine_states


def test_mask_and_combine_states():
    states = [{"a": np.arange(5).astype(float)}, {"a": np.arange(5, 10).astype(float)}, {"a": np.arange(10, 15)}]
    masks = [np.zeros(5).astype(bool) for _ in states]
    masks[0][[1, 2]] = True
    masks[1][[2, 3]] = True
    masks[2][[2, 4]] = True
    combined_state = {k: states[0][k][:] for k in states[0]}
    combined_mask = masks[0]
    for k in range(1, len(states)):
        mask = masks[k]
        new_state = states[k]
        combined_state = _mask_and_combine_states(combined_state, new_state, combined_mask, mask, ["a"])
        combined_mask = slice(0, None)

    assert combined_state["a"].shape[0] == 6
    assert (
        combined_state["a"]
        == np.array(
            [
                states[0]["a"][1],
                states[0]["a"][2],
                states[1]["a"][2],
                states[1]["a"][3],
                states[2]["a"][2],
                states[2]["a"][4],
            ]
        )
    ).all()
