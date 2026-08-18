"""Characterization tests for the DoD and time stress-factor paths (finding F7).

Both DoD fit paths are now implemented (see test_dod_fit.py): the
reference-conditions path and the non-reference stress-removal path. This
module keeps the schema guard (non-reference data must carry the extra
condition columns) and pins that the DoD *model* drives simulation. The time
stress factor is calculated from calendar degradation rates and is exact for
synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from esf.io.data import SampleData, example_sample_data
from esf.models.base_models import (
    exponential_temperature_k_relation,
    nonlinear_cal_model,
)
from esf.models.fitting import (
    DoDSFfit,
    degradation_rates_fit,
    time_stress_factor_calc,
)
from esf.settings.parameters import DataType, get_example_params
from esf.simulations.cycles import drive_cycle_001
from esf.simulations.degradation import drive_cycle_degradation_calculator


class TestDoDNonReferenceSchema:
    """The non-reference DoD fit needs the extra condition columns (T, SoC,
    t_cycle); without them it must raise a clear error rather than silently
    producing wrong numbers. The recovery behaviour itself lives in
    test_dod_fit.py."""

    def _dod_data(self):
        data = example_sample_data()
        data.calculate_life_fraction()
        selection = data.cycle_life_vs_dod(strict_mode=False)
        uid0 = selection["uid"].iloc[0]
        return selection[selection["uid"] == uid0].copy()

    def test_non_reference_without_columns_raises(self):
        prms = get_example_params()
        fit = DoDSFfit(
            self._dod_data(), prms, regime="cycling", is_at_reference=False
        )
        with pytest.raises(ValueError, match="t_cycle"):
            fit.fit()


class TestDoDModelDrivesSimulation:
    """Even though the DoD parameters cannot be fitted yet, they are wired into
    the degradation simulation (via the k_*_dod parameters)."""

    def test_k_1_dod_changes_simulated_loss(self):
        df = drive_cycle_001(verbose=False)
        cycle_numbers = np.linspace(1, 1000, 3)

        base = get_example_params()
        doubled = get_example_params()
        doubled.k_1_dod = base.k_1_dod * 2.0

        loss_base = drive_cycle_degradation_calculator(
            df, base, cycle_numbers=cycle_numbers
        )["loss"].iloc[-1]
        loss_doubled = drive_cycle_degradation_calculator(
            df, doubled, cycle_numbers=cycle_numbers
        )["loss"].iloc[-1]

        assert loss_base > 0
        assert not np.isclose(loss_base, loss_doubled)


class TestTimeStressFactorRecovery:
    """The time stress factor is k_t = deg_rate_per_time / reference_time.

    Built from synthetic calendar data whose per-temperature degradation rate
    is exactly rate * temperature_stress, the calculation must recover
    rate / reference_time (the temperature stress cancels out).
    """

    RATE_PER_DAY = 4.5e-4
    ALPHA, BETA = 0.06, 110.0
    K_TEMPERATURE = 0.0693

    def _rates_frame(self, prms):
        t_days = np.concatenate([np.linspace(0.5, 30, 15), np.linspace(50, 3000, 25)])
        frames = []
        for temperature in (288.15, 298.15, 308.15, 318.15):
            stress = float(
                exponential_temperature_k_relation(
                    np.array([temperature]), self.K_TEMPERATURE, x_ref=298.15
                )[0]
            )
            loss = nonlinear_cal_model(
                t_days * 86_400.0, self.ALPHA, self.BETA, self.RATE_PER_DAY * stress
            )
            frame = pd.DataFrame({"t": t_days, "SoH": 1 - loss})
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
        return degradation_rates_fit(
            prms, selection, data_type=DataType.CALENDAR_VS_TEMPERATURE
        )

    def test_recovers_rate_over_reference_time(self):
        prms = get_example_params()
        prms.sei_alpha, prms.sei_beta = self.ALPHA, self.BETA
        rates_frame = self._rates_frame(prms)

        calc = time_stress_factor_calc(
            prms, rates_frame, data_type=DataType.CALENDAR_VS_TEMPERATURE
        )

        expected = self.RATE_PER_DAY / prms.reference_calendar_time  # per second
        assert calc.calculated_k_time == pytest.approx(expected, rel=1e-6)
        assert prms.k_1_time_calendar == pytest.approx(expected, rel=1e-6)
