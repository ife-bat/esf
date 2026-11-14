"""Tests for the additional stress-factor forms and the C-rate factor.

The forms: an exponential temperature factor with a quadratic exponent
(calendar), a piecewise temperature factor with its minimum below the
reference (cycling), a logarithmic SoC factor (cycling) and a linear C-rate
factor (cycling, model set "with_rate"). Also covered: cycle_unit "EFC"
weights each rainflow cycle by its DoD whether or not half cycles are
weighted by their count.
"""

import numpy as np
import pytest

import esf
from esf.models.base_models import (
    exponential_quadratic_temperature_k_relation,
    linear_rate_k_relation,
    logarithmic_soc_k_relation,
    piecewise_temperature_k_relation,
)
from esf.settings.parameters import MODEL_SETS
from esf.simulations.degradation import (
    calculate_stress_factors,
    count_cycles,
    unpack_parameters,
)

T_REF = 298.15


class TestExponentialQuadraticTemperature:
    def test_is_one_at_reference(self):
        assert exponential_quadratic_temperature_k_relation(
            np.array([T_REF]), 6e-4, 0.066, T_REF
        )[0] == pytest.approx(1.0)

    def test_value(self):
        # 20 K above the reference: exp((6e-4 * 20 + 0.066) * 20)
        got = exponential_quadratic_temperature_k_relation(
            np.array([T_REF + 20.0]), 6e-4, 0.066, T_REF
        )[0]
        assert got == pytest.approx(np.exp((6e-4 * 20 + 0.066) * 20))

    def test_reduces_to_plain_exponential(self):
        x = np.array([T_REF - 15.0, T_REF + 15.0])
        got = exponential_quadratic_temperature_k_relation(x, 0.0, 0.05, T_REF)
        np.testing.assert_allclose(got, np.exp(0.05 * (x - T_REF)))

    def test_unit_independent(self):
        kelvin = exponential_quadratic_temperature_k_relation(
            np.array([308.15]), 6e-4, 0.066, 298.15
        )
        celsius = exponential_quadratic_temperature_k_relation(
            np.array([35.0]), 6e-4, 0.066, 25.0
        )
        np.testing.assert_allclose(kelvin, celsius)


class TestPiecewiseTemperature:
    ARGS = (1.0, 0.95, 0.04, 0.02, 10.0, T_REF)  # c_1, c_2, k_1, k_2, width, ref

    def test_values_at_the_joints(self):
        x = np.array([T_REF, T_REF - 10.0])
        np.testing.assert_allclose(
            piecewise_temperature_k_relation(x, *self.ARGS), [1.0, 0.95]
        )

    def test_continuous_at_the_joints(self):
        eps = 1e-9
        for joint in (T_REF, T_REF - 10.0):
            x = np.array([joint - eps, joint + eps])
            low, high = piecewise_temperature_k_relation(x, *self.ARGS)
            assert low == pytest.approx(high, abs=1e-8)

    def test_each_branch(self):
        x = np.array([T_REF + 20.0, T_REF - 5.0, T_REF - 20.0])
        expected = [
            1.0 * np.exp(0.04 * 20.0),
            0.95 + (1.0 - 0.95) * 0.5,
            0.95 * np.exp(0.02 * 10.0),
        ]
        np.testing.assert_allclose(
            piecewise_temperature_k_relation(x, *self.ARGS), expected
        )

    def test_grows_towards_both_ends(self):
        x = np.linspace(T_REF - 40.0, T_REF + 40.0, 81)
        s = piecewise_temperature_k_relation(x, *self.ARGS)
        assert x[np.argmin(s)] == pytest.approx(T_REF - 10.0)

    def test_no_overflow_far_from_the_reference(self):
        with np.errstate(over="raise"):
            s = piecewise_temperature_k_relation(
                np.array([T_REF - 5.0]), 1.0, 0.95, 50.0, 50.0, 10.0, T_REF
            )
        assert np.isfinite(s).all()


class TestLogarithmicSoc:
    def test_is_one_at_reference(self):
        assert logarithmic_soc_k_relation(np.array([0.5]), 0.55, 0.5)[0] == 1.0

    def test_value(self):
        got = logarithmic_soc_k_relation(np.array([0.8]), 0.55, 0.5)[0]
        assert got == pytest.approx(0.55 * np.log(0.8 / 0.5) + 1.0)

    def test_floored_at_zero(self):
        s = logarithmic_soc_k_relation(np.array([0.0, 0.01, 0.05]), 0.55, 0.5)
        np.testing.assert_array_equal(s, 0.0)


class TestLinearRate:
    def test_is_one_at_reference(self):
        assert linear_rate_k_relation(np.array([0.75]), 0.04, 0.75)[0] == 1.0

    def test_value(self):
        got = linear_rate_k_relation(np.array([1.5]), 0.04, 0.75)[0]
        assert got == pytest.approx(0.04 * 0.75 + 1.0)

    def test_floored_at_zero(self):
        assert linear_rate_k_relation(np.array([0.0]), 2.0, 1.0)[0] == 0.0


def _prms_with_rate(**overrides):
    prms = esf.get_example_params()
    prms.degradation_model = "with_rate"
    prms.rate_cycling_model = "Linear"
    prms.k_1_rate = 0.2
    prms.reference_rate = 1.0
    for key, value in overrides.items():
        setattr(prms, key, value)
    return prms


class TestRegistration:
    @pytest.mark.parametrize(
        "regime, factor, attribute, label, function",
        [
            ("Calendar", "temperature", "temperature_calendar_model",
             "ExponentialQuadratic", exponential_quadratic_temperature_k_relation),
            ("Cycling", "temperature", "temperature_cycling_model",
             "Piecewise", piecewise_temperature_k_relation),
            ("Cycling", "soc", "soc_cycling_model",
             "Logarithmic", logarithmic_soc_k_relation),
            ("Cycling", "rate", "rate_cycling_model",
             "Linear", linear_rate_k_relation),
        ],
    )
    def test_label_resolves_to_function(self, regime, factor, attribute, label, function):
        prms = esf.ESFParams(**{attribute: label})
        assert prms.stress_model_function(regime, factor) is function
        values = prms.stress_model_parameter_values(regime, factor)
        assert all(np.isfinite(float(v)) for v in values)

    def test_with_rate_model_set(self):
        assert MODEL_SETS["with_rate"]["Cycling"] == ["dod", "soc", "temperature", "rate"]

    def test_new_labels_survive_json_round_trip(self, tmp_path):
        prms = esf.ESFParams(
            temperature_calendar_model="ExponentialQuadratic",
            temperature_cycling_model="Piecewise",
            soc_cycling_model="Logarithmic",
            degradation_model="with_rate",
            k_quadratic_1_temperature_calendar=6e-4,
            c_piecewise_2_temperature_cycling=0.95,
            k_logarithmic_soc_cycling=0.55,
        )
        path = tmp_path / "prms.json"
        prms.save_json(path)
        restored = esf.ESFParams.load_json(path)
        assert restored.temperature_cycling_model == "Piecewise"
        assert restored.k_quadratic_1_temperature_calendar == 6e-4
        assert restored.c_piecewise_2_temperature_cycling == 0.95
        assert restored.k_logarithmic_soc_cycling == 0.55


class TestRateInSimulation:
    def test_count_cycles_absolute_rate(self):
        # a symmetric triangle: the signed C-rate averages to about zero over a
        # cycle, the absolute C-rate does not
        t = np.arange(0.0, 7201.0, 1.0)
        c_rate = np.where(t < 3600.0, -0.4, 0.4)
        soc = 0.7 + np.cumsum(c_rate) / 3600.0
        temperature = np.full_like(t, T_REF)
        signed, *_ = count_cycles(t, soc, c_rate, temperature, peak_finder_resolution=0.01)
        absolute, *_ = count_cycles(
            t, soc, c_rate, temperature, peak_finder_resolution=0.01, absolute_rate=True
        )
        assert np.all(np.asarray(absolute) == pytest.approx(0.4))
        assert np.all(np.abs(np.asarray(signed)) < 0.4)

    def test_rate_factor_scales_the_cycling_stress(self):
        cycle = esf.drive_cycle_001(verbose=False)
        numbers = np.array([1.0])
        without = esf.drive_cycle_degradation_calculator(
            cycle, _prms_with_rate(k_1_rate=0.0), cycle_numbers=numbers
        )
        with_rate = esf.drive_cycle_degradation_calculator(
            cycle, _prms_with_rate(k_1_rate=0.2), cycle_numbers=numbers
        )
        # k = 0 makes the factor 1 everywhere; a non-zero k changes f_c but
        # leaves the calendar term alone
        assert with_rate["f_c"].iloc[0] != pytest.approx(without["f_c"].iloc[0])
        assert with_rate["f_t"].iloc[0] == pytest.approx(without["f_t"].iloc[0])

    def test_default_model_set_is_unchanged_by_the_rate_code(self):
        # the default set has no rate factor: the signed mean C-rate is still
        # what the cycle counter is asked for, and results do not depend on
        # the rate parameters
        cycle = esf.drive_cycle_001(verbose=False)
        prms = esf.get_example_params()
        other = esf.get_example_params()
        other.k_1_rate = 5.0
        numbers = np.array([1.0, 100.0])
        a = esf.drive_cycle_degradation_calculator(cycle, prms, cycle_numbers=numbers)
        b = esf.drive_cycle_degradation_calculator(cycle, other, cycle_numbers=numbers)
        np.testing.assert_array_equal(a["loss"].to_numpy(), b["loss"].to_numpy())


class TestEfcWeighting:
    """cycle_unit "EFC": the stress is per equivalent full cycle, so a rainflow
    cycle is weighted by its DoD with or without count weighting."""

    ARRAYS = dict(
        arr_time=np.array([100.0, 200.0]),
        arr_soc_mean=np.array([0.5, 0.5]),
        arr_temp=np.array([T_REF, T_REF]),
        arr_crate=np.array([0.5, 0.5]),
        arr_n=np.array([0.5, 1.0]),
        arr_dod=np.array([0.2, 0.4]),
    )

    def _f_cycling(self, cycle_unit, weight_by_cycle_count):
        prms = esf.get_example_params()
        prms.cycle_unit = cycle_unit
        keywords = unpack_parameters(prms, use_model_sets=True)
        f_cycling, *_ = calculate_stress_factors(
            **self.ARRAYS,
            use_model_sets=True,
            weight_by_cycle_count=weight_by_cycle_count,
            **keywords,
        )
        return f_cycling

    def test_efc_with_count_weighting(self):
        per_cycle = self._f_cycling("N", weight_by_cycle_count=False)
        got = self._f_cycling("EFC", weight_by_cycle_count=True)
        np.testing.assert_allclose(got, per_cycle * np.array([0.5 * 0.2, 1.0 * 0.4]))

    def test_efc_without_count_weighting_still_weights_by_dod(self):
        per_cycle = self._f_cycling("N", weight_by_cycle_count=False)
        got = self._f_cycling("EFC", weight_by_cycle_count=False)
        np.testing.assert_allclose(got, per_cycle * np.array([0.2, 0.4]))

    def test_cycle_number_unit_is_unchanged(self):
        per_cycle = self._f_cycling("N", weight_by_cycle_count=False)
        weighted = self._f_cycling("N", weight_by_cycle_count=True)
        np.testing.assert_allclose(weighted, per_cycle * np.array([0.5, 1.0]))
