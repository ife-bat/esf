"""Synthetic-data recovery tests for the fitting module.

Data are generated from the model equations with known parameters; the fits
must recover those parameters. This verifies the full fitting plumbing
(data selectors -> fit classes -> prms updates) numerically, not just that it
runs.
"""

import numpy as np
import pandas as pd
import pytest

from esf.io.data import SampleData
from esf.models.base_models import (
    exponential_soc_k_relation,
    exponential_temperature_k_relation,
    nonlinear_cal_model,
    nonlinear_cycle_model,
)
from esf.models.fitting import (
    NonlinearFit,
    SoCSFfit,
    TemperatureSFfit,
    degradation_rates_fit,
    sei_fit_at_reference_conditions,
    soc_stress_factor_fit,
    temperature_stress_factor_fit,
)
from esf.settings.parameters import DataType, get_example_params

# true parameters used to generate the synthetic data
ALPHA_TRUE = 0.06
BETA_TRUE = 110.0
RATE_PER_DAY_TRUE = 4.5e-4
K_SOC_TRUE = 1.3
K_TEMPERATURE_TRUE = 0.0693
REFERENCE_RATE = 3.6e-5

# sampling that resolves both the SEI transient (~1/(beta*rate) = 20 days)
# and the long-term stage
T_DAYS = np.concatenate([np.linspace(0.5, 30, 15), np.linspace(50, 3000, 25)])


def make_calendar_sample_data(temperatures=(25.0,), k_temperature=0.0):
    """Calendar aging curves at one or more temperatures (given in degC).

    The temperature stress is parameterized in kelvin (x_ref = 298.15 K)
    because the data selectors convert temperatures to kelvin, and the
    exponential temperature model is not invariant under a change of unit
    (the x_ref/x factor differs between degC and K parameterizations).
    """
    frames = []
    for temperature in temperatures:
        stress = float(
            exponential_temperature_k_relation(
                np.array([temperature + 273.15]), k_temperature, x_ref=298.15
            )[0]
        )
        loss = nonlinear_cal_model(
            T_DAYS * 86_400.0, ALPHA_TRUE, BETA_TRUE, RATE_PER_DAY_TRUE * stress
        )
        frame = pd.DataFrame({"t": T_DAYS, "SoH": 1 - loss})
        frame["T"] = temperature
        frames.append(frame)
    data = SampleData()
    data.add_data(
        pd.concat(frames),
        data_type=DataType.CALENDAR_VS_TEMPERATURE,
        time_unit="days",
        temperature_unit="degC",
    )
    data.calculate_life_fraction()
    return data


class TestStressFactorRecovery:
    soc = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    temperature = np.array([288.15, 298.15, 308.15, 318.15, 328.15])  # K

    def soc_rates_frame(self):
        rates = REFERENCE_RATE * exponential_soc_k_relation(self.soc, K_SOC_TRUE)
        return pd.DataFrame({"SoC": self.soc, "deg_rate": rates})

    def temperature_rates_frame(self):
        rates = REFERENCE_RATE * exponential_temperature_k_relation(
            self.temperature, K_TEMPERATURE_TRUE, x_ref=298.15
        )
        return pd.DataFrame({"T": self.temperature, "deg_rate": rates})

    def test_soc_calendar(self):
        prms = get_example_params()
        soc_stress_factor_fit(
            prms, self.soc_rates_frame(), data_type=DataType.CALENDAR_VS_SOC
        )
        assert prms.k_soc_calendar == pytest.approx(K_SOC_TRUE, rel=1e-4)

    def test_soc_cycling(self):
        prms = get_example_params()
        fit = SoCSFfit(self.soc_rates_frame(), prms, regime="cycling")
        fit.fit()
        assert prms.k_soc_cycling == pytest.approx(K_SOC_TRUE, rel=1e-4)

    def test_temperature_calendar(self):
        prms = get_example_params()
        temperature_stress_factor_fit(
            prms,
            self.temperature_rates_frame(),
            data_type=DataType.CALENDAR_VS_TEMPERATURE,
        )
        assert prms.k_temperature_calendar == pytest.approx(
            K_TEMPERATURE_TRUE, rel=1e-4
        )

    def test_temperature_cycling(self):
        prms = get_example_params()
        fit = TemperatureSFfit(self.temperature_rates_frame(), prms, regime="cycling")
        fit.fit()
        assert prms.k_temperature_cycling == pytest.approx(
            K_TEMPERATURE_TRUE, rel=1e-4
        )

    def test_fitted_parameters_read_side(self):
        prms = get_example_params()
        fit = SoCSFfit(self.soc_rates_frame(), prms, regime="calend")
        fit.fit()
        fitted = fit.fitted_parameters()
        # keyed by the ESFParams attribute name, recovers the true constant
        assert fitted["k_soc_calendar"] == pytest.approx(K_SOC_TRUE, rel=1e-4)
        assert fitted == fit.fitted_parameters()  # pure read, repeatable


class TestSeiRecovery:
    def test_calendar_at_reference_conditions(self):
        # regression note: before alpha_sei/beta_sei got physical bounds, this
        # crashed inside lmfit with "The array returned by a function changed
        # size between calls" (exp overflow while exploring unbounded space)
        data = make_calendar_sample_data()
        selection = data.calendar_life_vs_temperature(
            filter_value=25.0, strict_mode=False
        )
        prms = get_example_params()
        fit = sei_fit_at_reference_conditions(
            prms, selection, data_type=DataType.CALENDAR_VS_TEMPERATURE
        )
        assert prms.sei_alpha == pytest.approx(ALPHA_TRUE, rel=1e-3)
        assert prms.sei_beta == pytest.approx(BETA_TRUE, rel=1e-3)
        rate = fit.fit_result.get_fit().best_values["deg_per_time_unit"]
        assert rate == pytest.approx(RATE_PER_DAY_TRUE, rel=1e-3)

    def test_cycling_at_reference_conditions(self):
        alpha, beta, rate_per_cycle = 0.05, 120.0, 2e-4
        n = np.concatenate([np.linspace(1, 30, 12), np.linspace(50, 4000, 25)])
        loss = nonlinear_cycle_model(n, alpha, beta, rate_per_cycle)
        frame = pd.DataFrame({"N": n, "SoH": 1 - loss})
        frame["T"] = 25.0
        frame["SoC"] = 0.5
        data = SampleData()
        data.add_data(
            frame, data_type=DataType.CYCLE_VS_TEMPERATURE, temperature_unit="degC"
        )
        data.calculate_life_fraction()
        selection = data.cycle_life_vs_temperature(strict_mode=False)

        prms = get_example_params()
        fit = NonlinearFit(
            selection,
            prms,
            regime="cycling_vs_temperature",
            is_reference=True,
            update_degradation_parameter=True,
        )
        fit.fit()
        assert prms.sei_alpha == pytest.approx(alpha, rel=1e-3)
        assert prms.sei_beta == pytest.approx(beta, rel=1e-3)
        # the cycling degradation rate lands in the canonical field deg_per_cycle
        assert prms.deg_per_cycle == pytest.approx(rate_per_cycle, rel=1e-3)

    def test_fitted_parameters_read_side_does_not_mutate(self):
        # the read side reports the fitted values without touching prms
        n = np.concatenate([np.linspace(1, 30, 12), np.linspace(50, 4000, 25)])
        loss = nonlinear_cycle_model(n, 0.05, 120.0, 2e-4)
        frame = pd.DataFrame({"N": n, "SoH": 1 - loss})
        frame["T"] = 298.15
        frame["SoC"] = 0.5
        data = SampleData()
        data.add_data(frame, data_type=DataType.CYCLE_VS_TEMPERATURE)
        data.calculate_life_fraction()
        selection = data.cycle_life_vs_temperature(strict_mode=False)

        prms = get_example_params()
        # fix the SEI at the true generating values so the rate is recoverable
        prms.sei_alpha, prms.sei_beta = 0.05, 120.0
        deg_before = prms.deg_per_cycle
        # is_reference False and no degradation update: fit must not change prms
        fit = NonlinearFit(
            selection, prms, regime="cycling_vs_temperature", is_reference=False
        )
        fit.fit()
        assert prms.sei_alpha == 0.05
        assert prms.sei_beta == 120.0
        assert prms.deg_per_cycle == deg_before  # not applied (no update flag)
        fitted = fit.fitted_parameters()
        # fitted_parameters keys the rate by its ESFParams field name
        assert fitted["deg_per_cycle"] == pytest.approx(2e-4, rel=1e-3)


class TestDegradationRatesRecovery:
    """degradation_rates_fit extracts one rate per condition; chained with the
    stress-factor fit it must give back the temperature constant."""

    @pytest.fixture(scope="class")
    def rates_frame(self):
        temperatures = (15.0, 25.0, 35.0, 45.0)
        data = make_calendar_sample_data(
            temperatures=temperatures, k_temperature=K_TEMPERATURE_TRUE
        )
        selection = data.calendar_life_vs_temperature(strict_mode=False)
        prms = get_example_params()
        # SEI parameters are known (fitted at reference); fix them:
        prms.sei_alpha = ALPHA_TRUE
        prms.sei_beta = BETA_TRUE
        return degradation_rates_fit(
            prms, selection, data_type=DataType.CALENDAR_VS_TEMPERATURE
        )

    def test_rates_scale_with_temperature_stress(self, rates_frame):
        # the selector hands temperatures over in kelvin
        assert list(rates_frame.columns) == ["t_max", "T", "deg_rate"]
        assert rates_frame["T"].min() > 273.0
        expected = RATE_PER_DAY_TRUE * exponential_temperature_k_relation(
            rates_frame["T"].to_numpy(), K_TEMPERATURE_TRUE, x_ref=298.15
        )
        np.testing.assert_allclose(rates_frame["deg_rate"], expected, rtol=1e-3)

    def test_chained_temperature_stress_factor_fit(self, rates_frame):
        # with kelvin as the internal unit everywhere, the chained pipeline
        # (rates fit -> stress-factor fit) works without any manual unit flip
        prms = get_example_params()
        temperature_stress_factor_fit(
            prms, rates_frame, data_type=DataType.CALENDAR_VS_TEMPERATURE
        )
        assert prms.k_temperature_calendar == pytest.approx(
            K_TEMPERATURE_TRUE, rel=1e-3
        )
