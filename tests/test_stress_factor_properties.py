"""Property-based tests for the stress-factor relations (round-4 housekeeping).

Complements the point tests in test_base_models.py with invariants that must
hold across the whole parameter range:

- the normalized multipliers (SoC, high-SoC, temperature) are exactly 1 at the
  reference condition and strictly positive everywhere;
- each relation is monotonic in its stress variable with the sign of its rate
  constant.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from esf.models.base_models import (
    exponential_dod_k_relation,
    exponential_high_soc_k_relation,
    exponential_soc_k_relation,
    exponential_temperature_k_relation,
)

# small helpers -----------------------------------------------------------
k_soc = st.floats(min_value=0.05, max_value=5.0)
k_signed = st.floats(min_value=0.05, max_value=5.0).flatmap(
    lambda v: st.sampled_from([v, -v])
)
soc_ref = st.floats(min_value=0.2, max_value=0.8)
temp_ref = st.floats(min_value=273.15, max_value=318.15)


def _is_strictly_increasing(y):
    return np.all(np.diff(y) > 0)


def _is_strictly_decreasing(y):
    return np.all(np.diff(y) < 0)


class TestReferenceAnchor:
    """Normalized multipliers equal 1 at the reference condition (S(x_ref)=1)."""

    @given(k=k_signed, x_ref=soc_ref)
    def test_soc_is_one_at_reference(self, k, x_ref):
        out = exponential_soc_k_relation(np.array([x_ref]), k, x_ref)
        assert out[0] == 1.0

    @given(k=k_signed, x_ref=temp_ref)
    def test_temperature_is_one_at_reference(self, k, x_ref):
        out = exponential_temperature_k_relation(np.array([x_ref]), k, x_ref)
        assert out[0] == 1.0

    @given(k=k_signed, x_ref=st.floats(min_value=0.6, max_value=0.95))
    def test_high_soc_is_one_at_reference(self, k, x_ref):
        out = exponential_high_soc_k_relation(np.array([x_ref]), k, x_ref)
        assert out[0] == 1.0


class TestPositivity:
    @given(k=k_signed, x_ref=soc_ref)
    def test_soc_factor_is_positive(self, k, x_ref):
        x = np.linspace(0.0, 1.0, 25)
        assert np.all(exponential_soc_k_relation(x, k, x_ref) > 0)

    @given(k=k_signed, x_ref=temp_ref)
    def test_temperature_factor_is_positive(self, k, x_ref):
        x = np.linspace(273.15, 333.15, 25)
        assert np.all(exponential_temperature_k_relation(x, k, x_ref) > 0)


class TestMonotonicity:
    @given(k=k_soc, x_ref=soc_ref)
    def test_soc_increases_with_positive_k(self, k, x_ref):
        x = np.linspace(0.0, 1.0, 30)
        assert _is_strictly_increasing(exponential_soc_k_relation(x, k, x_ref))

    @given(k=k_soc, x_ref=soc_ref)
    def test_soc_decreases_with_negative_k(self, k, x_ref):
        x = np.linspace(0.0, 1.0, 30)
        assert _is_strictly_decreasing(exponential_soc_k_relation(x, -k, x_ref))

    @given(k=k_soc, x_ref=temp_ref)
    def test_temperature_increases_with_positive_k(self, k, x_ref):
        # factor k*(T-T_ref)*(T_ref/T) is strictly increasing in T (T>0)
        x = np.linspace(263.15, 333.15, 30)
        assert _is_strictly_increasing(exponential_temperature_k_relation(x, k, x_ref))

    @given(
        k1=st.floats(min_value=1e3, max_value=1e5),
        k2=st.floats(min_value=0.0, max_value=5.0),
    )
    def test_exponential_dod_increases_with_dod(self, k1, k2):
        # S = k1 * delta * exp(k2*delta) (k1>0, k2>=0) grows with DoD
        x = np.linspace(0.01, 1.0, 30)
        assert _is_strictly_increasing(
            exponential_dod_k_relation(x, k1, k2, k_3=0.0)
        )


class TestHighSoCClamp:
    @given(k=k_soc, x_ref=st.floats(min_value=0.6, max_value=0.9))
    @settings(max_examples=25)
    def test_clamped_to_one_below_reference(self, k, x_ref):
        # high-SoC stress only kicks in above the reference SoC
        x = np.linspace(0.0, x_ref - 1e-3, 20)
        assert np.allclose(exponential_high_soc_k_relation(x, k, x_ref), 1.0)

    @given(k=k_soc, x_ref=st.floats(min_value=0.6, max_value=0.9))
    @settings(max_examples=25)
    def test_at_least_one_above_reference(self, k, x_ref):
        x = np.linspace(x_ref + 1e-3, 1.0, 20)
        assert np.all(exponential_high_soc_k_relation(x, k, x_ref) >= 1.0)
