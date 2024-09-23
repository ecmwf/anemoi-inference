import numpy as np
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from anemoi.inference.condition import Condition


@given(shape=npst.array_shapes(min_dims=2), data=st.data())
def test_from_numpy(shape, data):
    # Test condition creation from numpy arrays
    data_strategy = npst.arrays(
        dtype=np.float32,
        shape=shape,
        elements=dict(allow_nan=False, allow_infinity=False),
    )
    data_array = data.draw(data_strategy)

    var_strategy = npst.arrays(
        dtype=str,
        shape=shape[0],
        elements=npst.byte_string_dtypes(min_len=1),
        unique=True,
    )
    var_array = data.draw(var_strategy)
    condition = Condition.from_numpy(data_array, var_array)

    assume(not np.isnan(data_array).any())
    assume(np.isfinite(data_array).all())
    assert np.allclose(condition.to_array(var_array), data_array)

    # Generate a permutation of indices
    permutation = np.random.permutation(shape[0])

    # Apply the permutation to both arrays
    new_data_array = data_array[permutation]
    new_var_array = var_array[permutation]

    assert np.allclose(condition.to_array(new_var_array), new_data_array)
