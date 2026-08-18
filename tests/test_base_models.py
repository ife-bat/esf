"""Numerical tests for the core math in esf.models.base_models.

These tests pin the current behaviour of the stress-factor relations and the
nonlinear (SEI) degradation models so that refactoring can be done safely.
"""

import numpy as np
import pytest

from esf.models import base_models as bm

ALPHA = 0.0575763976
BETA = 113.383941797


class TestNonlinearDegradationModel:
    def test_zero_at_x_zero(self):
        assert bm.nonlinear_degradation_model(0.0, ALPHA, BETA, 1e-4) == pytest.approx(
            0.0
        )

    def test_approaches_one_for_large_x(self):
        loss = bm.nonlinear_degradation_model(1e9, ALPHA, BETA, 1e-4)
        assert loss == pytest.approx(1.0)

    def test_matches_explicit_formula(self):
        x, rate, x_ref = 1500.0, 2e-4, 1.0
        expected = (
            1
            - ALPHA * np.exp(-x / x_ref * BETA * rate)
            - (1 - ALPHA) * np.exp(-x / x_ref * rate)
        )
        assert bm.nonlinear_degradation_model(
            x, ALPHA, BETA, rate, x_ref=x_ref
        ) == pytest.approx(expected)

    def test_alpha_zero_reduces_to_single_exponential(self):
        x, rate = 100.0, 1e-3
        expected = 1 - np.exp(-x * rate)
        assert bm.nonlinear_degradation_model(x, 0.0, BETA, rate) == pytest.approx(
            expected
        )

    def test_x_ref_scales_x(self):
        assert bm.nonlinear_degradation_model(
            86_400.0, ALPHA, BETA, 1e-4, x_ref=86_400.0
        ) == pytest.approx(bm.nonlinear_degradation_model(1.0, ALPHA, BETA, 1e-4))

    def test_monotonically_increasing(self):
        x = np.linspace(0, 5000, 200)
        loss = bm.nonlinear_degradation_model(x, ALPHA, BETA, 1e-4)
        assert np.all(np.diff(loss) > 0)

    def test_total_model_consistent_with_x_model(self):
        # nonlinear_total_degradation_model takes the accumulated linear
        # degradation f_d = x/x_ref * rate directly
        x, rate = 800.0, 1e-4
        f_d = x * rate
        assert bm.nonlinear_total_degradation_model(ALPHA, BETA, f_d) == pytest.approx(
            bm.nonlinear_degradation_model(x, ALPHA, BETA, rate)
        )

    def test_total_model_zero_loss_for_zero_degradation(self):
        assert bm.nonlinear_total_degradation_model(ALPHA, BETA, 0.0) == pytest.approx(
            0.0
        )

    def test_cal_and_cycle_wrappers_delegate(self):
        # nonlinear_cal_model uses x_ref = 86 400 s (1 day) by default
        t, rate = 3 * 86_400.0, 5e-4
        assert bm.nonlinear_cal_model(t, ALPHA, BETA, rate) == pytest.approx(
            bm.nonlinear_degradation_model(t, ALPHA, BETA, rate, x_ref=86_400)
        )
        n = 250.0
        assert bm.nonlinear_cycle_model(n, ALPHA, BETA, rate) == pytest.approx(
            bm.nonlinear_degradation_model(n, ALPHA, BETA, rate, x_ref=1.0)
        )


class TestResidualFunctions:
    def test_linear_loss_residuals_zero_at_consistent_input(self):
        rate, n, ref = 2e-4, 500.0, 1.0
        loss = bm.nonlinear_degradation_model(n, ALPHA, BETA, rate, x_ref=ref)
        res = bm.linear_loss_residuals(rate, n, ALPHA, BETA, loss, ref)
        assert res == pytest.approx(0.0)

    def test_residuals_solve_cycle_num_zero_at_consistent_input(self):
        rate, n = 2e-4, 500.0
        full_deg = bm.nonlinear_cycle_model(n, ALPHA, BETA, rate)
        res = bm.residuals_solve_cycle_num(n, ALPHA, BETA, rate, full_deg)
        assert res == pytest.approx(0.0)


class TestSocModels:
    def test_linear_is_offset_at_reference(self):
        assert bm.linear_soc_k_relation(0.5, k_1=2.0, k_2=0.3) == pytest.approx(0.3)

    def test_linear_formula(self):
        assert bm.linear_soc_k_relation(0.8, k_1=2.0, k_2=0.3) == pytest.approx(
            2.0 * (0.8 - 0.5) + 0.3
        )

    def test_exponential_is_one_at_reference(self):
        assert bm.exponential_soc_k_relation(np.array([0.5]), 1.0) == pytest.approx(1.0)

    def test_exponential_formula(self):
        k = 1.0249254361
        soc = np.array([0.2, 0.5, 0.9])
        expected = np.exp(k * (soc - 0.5))
        np.testing.assert_allclose(bm.exponential_soc_k_relation(soc, k), expected)

    def test_exponential_accepts_scalar(self):
        assert bm.exponential_soc_k_relation(0.7, 1.0) == pytest.approx(np.exp(0.2))

    def test_exponential_accepts_integer_arrays(self):
        # regression test: np.ones_like used to inherit the integer dtype and
        # silently truncate the result to whole numbers
        k = 1.0
        result = bm.exponential_soc_k_relation(np.array([0, 1]), k)
        np.testing.assert_allclose(result, np.exp(k * (np.array([0.0, 1.0]) - 0.5)))

    def test_high_soc_is_one_below_reference(self):
        soc = np.array([0.1, 0.5, 0.85])
        np.testing.assert_allclose(
            bm.exponential_high_soc_k_relation(soc, 1.39), np.ones(3)
        )

    def test_high_soc_is_exponential_above_reference(self):
        k = 1.391411435
        soc = np.array([0.9, 1.0])
        expected = np.exp(k * (soc - 0.85))
        np.testing.assert_allclose(
            bm.exponential_high_soc_k_relation(soc, k), expected
        )


class TestTemperatureModel:
    def test_is_one_at_reference(self):
        assert bm.exponential_temperature_k_relation(
            np.array([298.0]), 0.07
        ) == pytest.approx(1.0)

    def test_formula(self):
        k, x_ref = 0.0698782039, 298.15
        temp = np.array([278.15, 298.15, 318.15])
        expected = np.exp(k * (temp - x_ref) * (x_ref / temp))
        np.testing.assert_allclose(
            bm.exponential_temperature_k_relation(temp, k, x_ref=x_ref), expected
        )

    def test_increases_with_temperature(self):
        temp = np.linspace(273.15, 333.15, 20)
        sf = bm.exponential_temperature_k_relation(temp, 0.07, x_ref=298.15)
        assert np.all(np.diff(sf) > 0)

    def test_accepts_integer_kelvin(self):
        k, x_ref = 0.07, 298.0
        temp_int = np.array([288, 308])
        expected = np.exp(k * (temp_int - x_ref) * (x_ref / temp_int))
        np.testing.assert_allclose(
            bm.exponential_temperature_k_relation(temp_int, k, x_ref=x_ref), expected
        )


class TestTimeModels:
    def test_linear(self):
        assert bm.linear_time_k_relation(10.0, k_1=2.0, k_2=1.0) == pytest.approx(21.0)

    def test_linear_x_ref_scaling(self):
        assert bm.linear_time_k_relation(
            86_400.0, k_1=2.0, k_2=1.0, x_ref=86_400.0
        ) == pytest.approx(3.0)

    def test_proportional_is_linear_without_offset(self):
        assert bm.proportional_time_k_relation(10.0, k_1=2.0) == pytest.approx(20.0)
        assert bm.proportional_time_k_relation(0.0, k_1=2.0) == pytest.approx(0.0)

    def test_constant_returns_k_1(self):
        # Note: the docstring historically claimed k_2 is returned; the model
        # register varies k_1 and fixes k_2 = 0, so k_1 is the constant.
        assert bm.constant_time_k_relation(123.0, k_1=5e-10, k_2=0.0) == pytest.approx(
            5e-10
        )


class TestDodModels:
    def test_exponential_formula(self):
        k_1, k_2, k_3 = 4.2701e-06, 1.45764279, 0.0
        dod = np.array([0.1, 0.5, 1.0])
        expected = k_1 * dod * np.exp(k_2 * dod) + k_3
        np.testing.assert_allclose(
            bm.exponential_dod_k_relation(dod, k_1, k_2, k_3), expected
        )

    def test_exponential_zero_at_zero_dod(self):
        assert bm.exponential_dod_k_relation(0.0, 1e-4, 1.5, 0.0) == pytest.approx(0.0)

    def test_empirical_formula(self):
        k_1, k_2, k_3 = 173050.0, -0.501, -147740.0
        dod = np.array([0.2, 0.5, 1.0])
        expected = 1.0 / (k_1 * np.power(dod, k_2) + k_3)
        np.testing.assert_allclose(
            bm.empirical_dod_k_relation(dod, k_1, k_2, k_3), expected
        )

    def test_empirical_increases_with_dod(self):
        # deeper cycles cost more per cycle (with the default LMO parameters)
        k_1, k_2, k_3 = 173050.0, -0.501, -147740.0
        dod = np.linspace(0.1, 1.0, 30)
        sf = bm.empirical_dod_k_relation(dod, k_1, k_2, k_3)
        assert np.all(sf > 0)
        assert np.all(np.diff(sf) > 0)
