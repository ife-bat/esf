"""Tests for esf.utils.converters."""

import numpy as np
import pytest
import uncertainties

from esf.settings.parameters import ureg
from esf.utils.converters import (
    to_hours,
    to_numpy_float64,
    to_numpy_floats64,
    to_seconds,
)


class TestTimeConverters:
    def test_to_seconds_default_unit_is_seconds(self):
        assert to_seconds(42.0) == pytest.approx(42.0)

    def test_to_seconds_from_hours(self):
        assert to_seconds(2.0, unit="hours") == pytest.approx(7200.0)

    def test_to_seconds_from_days(self):
        assert to_seconds(1.0, unit="days") == pytest.approx(86_400.0)

    def test_to_hours_from_seconds(self):
        assert to_hours(7200.0, unit="seconds") == pytest.approx(2.0)


class TestToNumpyFloat64:
    def test_plain_float(self):
        result = to_numpy_float64(1.5)
        assert isinstance(result, np.float64)
        assert result == pytest.approx(1.5)

    def test_ufloat_takes_nominal_value(self):
        u = uncertainties.ufloat(2.5, 0.3)
        assert to_numpy_float64(u) == pytest.approx(2.5)

    def test_pint_quantity_takes_magnitude(self):
        q = ureg.Quantity(3.5, "seconds")
        assert to_numpy_float64(q) == pytest.approx(3.5)

    def test_collections(self):
        as_tuple = to_numpy_floats64((1, uncertainties.ufloat(2.0, 0.1)))
        assert isinstance(as_tuple, tuple)
        np.testing.assert_allclose(as_tuple, (1.0, 2.0))

        as_list = to_numpy_floats64([1, 2])
        assert isinstance(as_list, list)

        as_dict = to_numpy_floats64({"a": 1, "b": uncertainties.ufloat(2.0, 0.1)})
        assert as_dict["b"] == pytest.approx(2.0)
