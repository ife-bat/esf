"""Tests for the drive-cycle degradation pipeline (esf.simulations.degradation).

Covers each stage separately (unpacking, repetition, stress factors, linear
rates, loss) plus an end-to-end regression with pinned numbers so that the
planned refactors cannot silently change the model output.
"""

import numpy as np
import pandas as pd
import pytest

from esf.models.base_models import (
    exponential_dod_k_relation,
    exponential_soc_k_relation,
    exponential_temperature_k_relation,
    nonlinear_total_degradation_model,
    proportional_time_k_relation,
)
from esf.settings.parameters import get_example_params
from esf.simulations.cycles import drive_cycle_001
from esf.simulations.degradation import (
    calc_cycle_at_end_of_life,
    calc_f,
    calc_fd,
    calc_loss,
    calculate_linear_degradation_rates,
    calculate_stress_factors,
    count_cycles,
    drive_cycle_degradation_calculator,
    repeat_drive_cycle,
    unpack_drive_cycle_frame,
    unpack_parameters,
)


@pytest.fixture(scope="module")
def example_prms():
    return get_example_params()


@pytest.fixture(scope="module")
def drive_cycle_df():
    return drive_cycle_001(verbose=False)


@pytest.fixture(scope="module")
def counted_cycles(drive_cycle_df):
    t, soc, temperature, c_rate = unpack_drive_cycle_frame(drive_cycle_df)
    return count_cycles(t, soc, c_rate, temperature, peak_finder_resolution=0.01)


class TestUnpackDriveCycleFrame:
    def test_unpacks_columns_in_order(self):
        df = pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "soc": [1.0, 0.9],
                "temperature": [298.0, 298.0],
                "c-rate": [-1.0, -1.0],
            }
        )
        t, soc, temperature, c_rate = unpack_drive_cycle_frame(df)
        np.testing.assert_allclose(t, [0.0, 1.0])
        np.testing.assert_allclose(soc, [1.0, 0.9])
        np.testing.assert_allclose(temperature, 298.0)
        np.testing.assert_allclose(c_rate, -1.0)

    def test_drops_nan_rows(self):
        df = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "soc": [1.0, np.nan, 0.8],
                "temperature": [298.0, 298.0, 298.0],
                "c-rate": [-1.0, -1.0, -1.0],
            }
        )
        t, soc, *_ = unpack_drive_cycle_frame(df)
        assert len(t) == 2
        np.testing.assert_allclose(soc, [1.0, 0.8])


class TestRepeatDriveCycle:
    @pytest.fixture
    def small_cycle(self):
        return pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "soc": [1.0, 0.8, 0.6, 0.8],
                "temperature": [298.0] * 4,
                "c-rate": [-1.0, -1.0, 1.0, 1.0],
            }
        )

    def test_length_and_tiling(self, small_cycle):
        out = repeat_drive_cycle(small_cycle, repetitions=3)
        assert len(out) == 12
        np.testing.assert_allclose(out["soc"], np.tile(small_cycle["soc"], 3))
        np.testing.assert_allclose(out["c-rate"], np.tile(small_cycle["c-rate"], 3))

    def test_time_keeps_constant_step(self, small_cycle):
        out = repeat_drive_cycle(small_cycle, repetitions=3)
        np.testing.assert_allclose(np.diff(out["time"]), 1.0)
        assert out["time"].iloc[0] == pytest.approx(0.0)
        assert out["time"].iloc[-1] == pytest.approx(11.0)


class TestLinearDegradationRates:
    def test_formula(self):
        f_c = np.array([1e-5, 2e-5])
        f_t = np.array([1e-8, 3e-8])
        cycle_numbers = np.array([1.0, 10.0, 100.0])
        f_c_sum, f_t_sum, f_d = calculate_linear_degradation_rates(
            cycle_numbers, f_c, f_t
        )
        assert f_c_sum == pytest.approx(3e-5)
        assert f_t_sum == pytest.approx(4e-8)
        np.testing.assert_allclose(f_d, cycle_numbers * 3e-5 + 4e-8)

    def test_calc_fd_sums_both_contributions(self):
        assert calc_fd(np.array([1.0, 2.0]), np.array([0.5])) == pytest.approx(3.5)

    def test_calc_f_is_elementwise_product(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        c = np.array([5.0, 6.0])
        np.testing.assert_allclose(calc_f(a, b, c), [15.0, 48.0])


class TestCalcLoss:
    def test_matches_nonlinear_total_model(self, example_prms):
        f_d = np.array([0.0, 0.05, 0.2])
        expected = nonlinear_total_degradation_model(
            example_prms.sei_alpha, example_prms.sei_beta, f_d
        )
        np.testing.assert_allclose(calc_loss(f_d, example_prms), expected)

    def test_zero_degradation_gives_zero_loss(self, example_prms):
        assert calc_loss(np.array([0.0]), example_prms)[0] == pytest.approx(0.0)


class TestCalculateStressFactors:
    def test_default_path_matches_manual_product(self):
        # one full cycle at known conditions, explicit models and parameters
        arr_time = np.array([3600.0])
        arr_soc = np.array([0.6])
        arr_temp = np.array([308.15])
        arr_crate = np.array([1.0])
        arr_n = np.array([1.0])
        arr_dod = np.array([0.8])
        k_time, k_soc, k_temp = 4e-10, 1.02, 0.0699
        k_dod = (4.2701e-06, 1.45764279, 0.0)

        f_cyc, f_cal, cal_factors, cyc_factors = calculate_stress_factors(
            arr_time,
            arr_soc,
            arr_temp,
            arr_crate,
            arr_n,
            arr_dod,
            time_calendar_stress_model_parameters=(k_time,),
            soc_calendar_stress_model_parameters=(k_soc,),
            soc_cycling_stress_model_parameters=(k_soc,),
            temperature_calendar_stress_model_parameters=(k_temp,),
            temperature_cycling_stress_model_parameters=(k_temp,),
            dod_stress_model_parameters=k_dod,
        )

        s_time = proportional_time_k_relation(arr_time, k_time)
        s_soc = exponential_soc_k_relation(arr_soc, k_soc)
        s_temp = exponential_temperature_k_relation(arr_temp, k_temp)
        s_dod = exponential_dod_k_relation(arr_dod, *k_dod)

        np.testing.assert_allclose(f_cal, s_time * s_soc * s_temp)
        np.testing.assert_allclose(f_cyc, s_dod * s_soc * s_temp)

    def test_model_set_path_agrees_with_default_path(
        self, example_prms, counted_cycles
    ):
        # the two ways of resolving stress models from ESFParams must give the
        # same result (the high-SoC factor is 1.0 below the 0.85 threshold)
        arr_crate, arr_dod, arr_n, arr_soc_mean, arr_temp, arr_time = counted_cycles
        kw_with = unpack_parameters(example_prms, use_model_sets=True)
        kw_without = unpack_parameters(example_prms, use_model_sets=False)

        f_cyc_1, f_cal_1, *_ = calculate_stress_factors(
            arr_time,
            arr_soc_mean,
            arr_temp,
            arr_crate,
            arr_n,
            arr_dod,
            use_model_sets=True,
            **kw_with,
        )
        f_cyc_2, f_cal_2, *_ = calculate_stress_factors(
            arr_time,
            arr_soc_mean,
            arr_temp,
            arr_crate,
            arr_n,
            arr_dod,
            use_model_sets=False,
            **kw_without,
        )
        np.testing.assert_allclose(f_cyc_1, f_cyc_2)
        np.testing.assert_allclose(f_cal_1, f_cal_2)


class TestCycleAtEndOfLife:
    def test_round_trip(self, example_prms):
        f_c, f_t = 2.3e-4, 1.1e-8
        eol_soh = 0.8
        n_eol = calc_cycle_at_end_of_life(f_c, f_t, example_prms, eol_soh=eol_soh)
        assert n_eol > 0
        # the loss at the solved cycle number must equal the target loss
        _, _, f_d = calculate_linear_degradation_rates(
            np.array([n_eol]), np.array([f_c]), np.array([f_t])
        )
        loss = calc_loss(f_d, example_prms)[0]
        assert loss == pytest.approx(1 - eol_soh, abs=1e-9)


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def result(self, example_prms, drive_cycle_df):
        cycle_numbers = np.linspace(1, 1000, 5)
        return drive_cycle_degradation_calculator(
            drive_cycle_df, example_prms, cycle_numbers=cycle_numbers
        )

    def test_frame_layout(self, result):
        assert len(result) == 5
        for col in ["cycle_number", "f_c", "f_t", "f_d", "loss", "soh"]:
            assert col in result.columns

    def test_loss_is_increasing_and_bounded(self, result):
        assert np.all(np.diff(result["loss"]) > 0)
        assert np.all(result["loss"] >= 0)
        assert np.all(result["loss"] <= 1)

    def test_soh_is_one_minus_loss(self, result):
        np.testing.assert_allclose(result["soh"], 1.0 - result["loss"])

    def test_delta_loss_accumulates_to_loss(self, result):
        np.testing.assert_allclose(
            result["delta_loss"].cumsum(), result["loss"], rtol=1e-12
        )

    def test_temperature_stress_is_unity_at_reference(self, example_prms):
        # the drive cycle runs at 298.15 K, the example parameters reference
        # 298.15 K -> the temperature stress factor must be exactly 1
        function = example_prms.stress_model_function("Cycling", "temperature")
        values = example_prms.stress_model_parameter_values("Cycling", "temperature")
        np.testing.assert_allclose(function(np.array([298.15]), *values), 1.0)

    def test_pinned_regression_values(self, result):
        # pinned on 2026-07-10 (drive_cycle_001 + get_example_params); a change
        # here means the model output changed - make sure it is intentional.
        # Re-pinned after the kelvin units hardening: the example parameters
        # previously used a degC reference (25) against a drive cycle in
        # kelvin (298.15), inflating the temperature stress factor to ~4.9;
        # at reference temperature it is now correctly 1.0.
        assert result["f_c"].iloc[0] == pytest.approx(4.7300181987666376e-05, rel=1e-6)
        assert result["f_t"].iloc[0] == pytest.approx(2.2348812431982e-09, rel=1e-6)
        assert result["loss"].iloc[0] == pytest.approx(0.00037275, rel=1e-4)
        assert result["loss"].iloc[-1] == pytest.approx(0.10085456, rel=1e-4)
