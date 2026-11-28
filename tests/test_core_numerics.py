import math

import numpy as np
import pytest

from core.numerics import as_finite_real, as_real_float_array


def test_real_array_conversion_preserves_numeric_values_and_nonfinite_slots():
    values = as_real_float_array([[1, 2], [np.nan, 4.5]], name="sample")
    assert values.dtype == np.float64
    np.testing.assert_equal(values, [[1.0, 2.0], [np.nan, 4.5]])


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1.0 + 2.0j]),
        np.asarray([True, False]),
        np.asarray(["1.0"]),
        np.asarray([2**53 + 1], dtype=np.int64),
        [10**400],
        [[1.0], [1.0, 2.0]],
    ],
)
def test_real_array_conversion_rejects_lossy_or_unrepresentable_inputs(value):
    with pytest.raises(ValueError):
        as_real_float_array(value, name="sample")


@pytest.mark.parametrize("value", [True, "0.5", 2**53 + 1, 10**400, 1.0 + 0.0j, float("inf")])
def test_finite_real_conversion_rejects_implicit_or_nonfinite_scalars(value):
    with pytest.raises(ValueError):
        as_finite_real(value, name="parameter")


def test_finite_real_conversion_accepts_integer_and_floating_scalars():
    assert as_finite_real(2, name="integer") == 2.0
    assert as_finite_real(np.float64(0.25), name="float") == 0.25
    assert math.isfinite(as_finite_real(-1.5, name="negative"))
