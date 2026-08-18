"""Tests for the ``apply`` flag on the fit facades (F2 finish).

``apply=True`` (default) writes fitted values into ``prms``; ``apply=False``
runs the fit but leaves ``prms`` untouched, so a caller can read results with
``fitted_parameters()`` without the side effect.
"""

import numpy as np
import pandas as pd
import pytest

import esf
from esf.io.data import SampleData
from esf.models.base_models import (
    exponential_soc_k_relation,
    exponential_temperature_k_relation,
    nonlinear_cal_model,
)
from esf.settings.parameters import DataType


def _soc_rates():
    soc = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    return pd.DataFrame(
        {"SoC": soc, "deg_rate": 3.6e-5 * exponential_soc_k_relation(soc, 1.3)}
    )


def _temperature_rates():
    temperature = np.array([288.15, 298.15, 308.15, 318.15, 328.15])
    return pd.DataFrame(
        {
            "T": temperature,
            "deg_rate": 4.5e-4
            * exponential_temperature_k_relation(temperature, 0.0693, x_ref=298.15),
        }
    )


class TestStressFactorApply:
    def test_apply_false_leaves_prms_untouched(self):
        prms = esf.get_example_params()
        before = prms.k_soc_calendar
        stamp = prms.last_updated
        fit = esf.soc_stress_factor_fit(
            prms, _soc_rates(), data_type=DataType.CALENDAR_VS_SOC, apply=False
        )
        assert prms.k_soc_calendar == before
        assert prms.last_updated == stamp  # not even marked changed
        # results are still readable from the fit object
        assert fit.fitted_parameters()["k_soc_calendar"] == pytest.approx(1.3, rel=1e-4)

    def test_apply_true_is_the_default(self):
        prms = esf.get_example_params()
        esf.temperature_stress_factor_fit(
            prms, _temperature_rates(), data_type=DataType.CALENDAR_VS_TEMPERATURE
        )
        assert prms.k_temperature_calendar == pytest.approx(0.0693, rel=1e-4)

    def test_apply_false_then_apply_manually(self):
        prms = esf.get_example_params()
        before = prms.k_soc_calendar
        fit = esf.soc_stress_factor_fit(
            prms, _soc_rates(), data_type=DataType.CALENDAR_VS_SOC, apply=False
        )
        assert prms.k_soc_calendar == before
        # the caller decides to apply the results
        for name, value in fit.fitted_parameters().items():
            prms.set(name, value)
        assert prms.k_soc_calendar == pytest.approx(1.3, rel=1e-4)


class TestSeiApply:
    def _calendar_selection(self):
        t = np.concatenate([np.linspace(0.5, 30, 15), np.linspace(50, 3000, 25)])
        loss = nonlinear_cal_model(t * 86_400.0, 0.06, 110.0, 4.5e-4)
        frame = pd.DataFrame({"t": t, "SoH": 1 - loss})
        frame["T"] = 298.15
        data = SampleData()
        data.add_data(
            frame,
            data_type=DataType.CALENDAR_VS_TEMPERATURE,
            time_unit="days",
            temperature_unit="K",
        )
        data.calculate_life_fraction()
        return data.calendar_life_vs_temperature(filter_value=298.15, strict_mode=False)

    def test_apply_false_leaves_sei_untouched(self):
        prms = esf.get_example_params()
        alpha, beta = prms.sei_alpha, prms.sei_beta
        fit = esf.sei_fit_at_reference_conditions(
            prms,
            self._calendar_selection(),
            data_type=DataType.CALENDAR_VS_TEMPERATURE,
            apply=False,
        )
        assert prms.sei_alpha == alpha
        assert prms.sei_beta == beta
        assert fit.fitted_parameters()["sei_alpha"] == pytest.approx(0.06, rel=1e-3)


class TestTimeCalcApply:
    def test_apply_false_leaves_k_time_untouched(self):
        # build the rates frame the time calc consumes
        prms = esf.get_example_params()
        prms.sei_alpha, prms.sei_beta = 0.06, 110.0
        t = np.concatenate([np.linspace(0.5, 30, 15), np.linspace(50, 3000, 25)])
        frames = []
        for temperature in (288.15, 298.15, 308.15, 318.15):
            stress = float(
                exponential_temperature_k_relation(
                    np.array([temperature]), 0.0693, x_ref=298.15
                )[0]
            )
            loss = nonlinear_cal_model(t * 86_400.0, 0.06, 110.0, 4.5e-4 * stress)
            frame = pd.DataFrame({"t": t, "SoH": 1 - loss})
            frame["T"] = temperature
            frames.append(frame)
        data = SampleData()
        data.add_data(
            pd.concat(frames),
            data_type=DataType.CALENDAR_VS_TEMPERATURE,
            time_unit="days",
            temperature_unit="K",
        )
        data.calculate_life_fraction()
        selection = data.calendar_life_vs_temperature(strict_mode=False)
        rates = esf.degradation_rates_fit(
            prms, selection, data_type=DataType.CALENDAR_VS_TEMPERATURE
        )

        before = prms.k_1_time_calendar
        calc = esf.time_stress_factor_calc(
            prms, rates, data_type=DataType.CALENDAR_VS_TEMPERATURE, apply=False
        )
        assert prms.k_1_time_calendar == before
        assert calc.calculated_k_time == pytest.approx(
            4.5e-4 / prms.reference_calendar_time, rel=1e-6
        )
