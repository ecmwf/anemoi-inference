# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.inference.testing import fake_checkpoints


@fake_checkpoints
def test_checkpoint() -> None:
    """Test the Checkpoint class.

    Returns
    -------
    None
    """
    from anemoi.inference.checkpoint import Checkpoint

    c = Checkpoint("simple.chkpt")
    c.variables_from_input


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
