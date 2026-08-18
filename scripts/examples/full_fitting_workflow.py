"""The complete esf fitting workflow, end to end, on synthetic data.

This is the runnable companion to docs/fitting-architecture.md. It generates
aging data from known ("true") parameters, runs every stage of the two-stage
fitting procedure, and finally simulates a drive cycle with the fitted
parameters. Because the data are synthetic, every fitted constant can be
compared against the value that generated the data.

Run with:
    uv run python scripts/examples/full_fitting_workflow.py

The test suite executes main() to keep this example (and the document that
embeds it) from rotting.
"""

import numpy as np
import pandas as pd

import esf
from esf.models.base_models import (
    exponential_soc_k_relation,
    exponential_temperature_k_relation,
    nonlinear_cal_model,
    nonlinear_cycle_model,
)

# ---------------------------------------------------------------------------
# The "true" parameters used to generate the synthetic aging data.
# A successful fit must give these back.
# ---------------------------------------------------------------------------
ALPHA_TRUE = 0.06  # fraction of capacity consumed while the SEI film forms
BETA_TRUE = 110.0  # rate ratio between SEI formation and the rest of life
RATE_PER_DAY_TRUE = 4.5e-4  # linear degradation rate at reference conditions
RATE_PER_CYCLE_TRUE = 2.0e-4  # cycling counterpart
K_TEMPERATURE_TRUE = 0.0693  # temperature stress constant (kelvin)
K_SOC_TRUE = 1.04  # SoC stress constant

REFERENCE_TEMPERATURE = 298.15  # K
REFERENCE_SOC = 0.5

# sampling that resolves both the SEI transient (~1/(beta*rate) = 20 days)
# and the long-term stage
T_DAYS = np.concatenate([np.linspace(0.5, 30, 15), np.linspace(50, 3000, 25)])


def make_calendar_data():
    """Synthetic calendar aging: SoH(t) curves at several T and several SoC."""
    frames = []
    # temperature series (at reference SoC)
    for temperature in (288.15, 298.15, 308.15, 318.15):
        stress = float(
            exponential_temperature_k_relation(
                np.array([temperature]), K_TEMPERATURE_TRUE, x_ref=REFERENCE_TEMPERATURE
            )[0]
        )
        loss = nonlinear_cal_model(
            T_DAYS * 86_400.0, ALPHA_TRUE, BETA_TRUE, RATE_PER_DAY_TRUE * stress
        )
        frame = pd.DataFrame({"t": T_DAYS, "SoH": 1 - loss})
        frame["T"] = temperature
        frame["SoC"] = REFERENCE_SOC
        frame["subset"] = f"T{temperature:.0f}"
        frames.append(frame)
    data_vs_temperature = pd.concat(frames)

    frames = []
    # SoC series (at reference temperature)
    for soc in (0.2, 0.5, 0.8, 1.0):
        stress = float(
            exponential_soc_k_relation(np.array([soc]), K_SOC_TRUE)[0]
        )
        loss = nonlinear_cal_model(
            T_DAYS * 86_400.0, ALPHA_TRUE, BETA_TRUE, RATE_PER_DAY_TRUE * stress
        )
        frame = pd.DataFrame({"t": T_DAYS, "SoH": 1 - loss})
        frame["T"] = REFERENCE_TEMPERATURE
        frame["SoC"] = soc
        frame["subset"] = f"SoC{soc:.1f}"
        frames.append(frame)
    data_vs_soc = pd.concat(frames)

    return data_vs_temperature, data_vs_soc


def make_cycling_data():
    """Synthetic cycle aging at reference conditions: SoH(N)."""
    n = np.concatenate([np.linspace(1, 30, 12), np.linspace(50, 4000, 25)])
    loss = nonlinear_cycle_model(n, ALPHA_TRUE, BETA_TRUE, RATE_PER_CYCLE_TRUE)
    frame = pd.DataFrame({"N": n, "SoH": 1 - loss})
    frame["T"] = REFERENCE_TEMPERATURE
    frame["SoC"] = REFERENCE_SOC
    return frame


def main(verbose=False):
    # -----------------------------------------------------------------------
    # 1. Load the aging data into a SampleData container.
    #    add_data() records the units in the metadata; the selectors convert
    #    to the internal units (seconds, kelvin) on the way out.
    # -----------------------------------------------------------------------
    data_vs_temperature, data_vs_soc = make_calendar_data()

    calendar_data = esf.SampleData()
    calendar_data.add_data(
        data_vs_temperature,
        data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
        comment="synthetic calendar aging vs temperature",
        time_unit="days",
        temperature_unit="K",
    )
    calendar_data.add_data(
        data_vs_soc,
        data_type=esf.DataType.CALENDAR_VS_SOC,
        comment="synthetic calendar aging vs SoC",
        time_unit="days",
        temperature_unit="K",
    )
    calendar_data.calculate_life_fraction()  # adds the loss column L = 1 - SoH

    cycling_data = esf.SampleData()
    cycling_data.add_data(
        make_cycling_data(),
        data_type=esf.DataType.CYCLE_VS_TEMPERATURE,
        comment="synthetic cycle aging at reference conditions",
        temperature_unit="K",
    )
    cycling_data.calculate_life_fraction()

    # -----------------------------------------------------------------------
    # 2. Start from a parameter set. The fits write their results into it.
    # -----------------------------------------------------------------------
    prms = esf.ESFParams(
        battery_chemistry="synthetic example",
        reference_temperature=REFERENCE_TEMPERATURE,
        reference_soc=REFERENCE_SOC,
    )

    # -----------------------------------------------------------------------
    # 3. Stage one - the nonlinear (SEI) fit at reference conditions.
    #    Only data measured at the reference temperature and SoC are used;
    #    this pins alpha_sei, beta_sei, and the linear degradation rate.
    # -----------------------------------------------------------------------
    at_reference = calendar_data.calendar_life_vs_temperature(
        filter_value=REFERENCE_TEMPERATURE, strict_mode=False
    )
    esf.sei_fit_at_reference_conditions(
        prms, at_reference, data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
        verbose=verbose,
    )

    # the cycling counterpart pins deg_per_cycle (SEI values refit on
    # cycling data; both fits see the same true alpha/beta here)
    cycling_at_reference = cycling_data.cycle_life_vs_temperature(strict_mode=False)
    esf.sei_fit_at_reference_conditions(
        prms, cycling_at_reference, data_type=esf.DataType.CYCLE_VS_TEMPERATURE,
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 4. Stage two - extract one degradation rate per condition.
    #    The SEI parameters are now frozen; each temperature (or SoC) series
    #    is refitted for its linear rate only.
    # -----------------------------------------------------------------------
    selection = calendar_data.calendar_life_vs_temperature(strict_mode=False)
    rates_vs_temperature = esf.degradation_rates_fit(
        prms, selection, data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
        verbose=verbose,
    )
    # -> DataFrame with columns [t_max, T, deg_rate]

    selection = calendar_data.calendar_life_vs_soc(strict_mode=False)
    rates_vs_soc = esf.degradation_rates_fit(
        prms, selection, data_type=esf.DataType.CALENDAR_VS_SOC, verbose=verbose
    )
    # -> DataFrame with columns [t_max, SoC, deg_rate]

    # -----------------------------------------------------------------------
    # 5. Stage three - fit the stress-factor models to the rates.
    #    Each fit normalizes the rates by the rate at the reference value,
    #    then fits its stress model; results land in prms.
    # -----------------------------------------------------------------------
    esf.temperature_stress_factor_fit(
        prms, rates_vs_temperature,
        data_type=esf.DataType.CALENDAR_VS_TEMPERATURE, verbose=verbose,
    )
    esf.soc_stress_factor_fit(
        prms, rates_vs_soc, data_type=esf.DataType.CALENDAR_VS_SOC, verbose=verbose
    )
    # the time stress factor is calculated (not fitted): k_t = rate / t_ref
    esf.time_stress_factor_calc(
        prms, rates_vs_temperature,
        data_type=esf.DataType.CALENDAR_VS_TEMPERATURE, verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 6. Use the fitted parameters in a simulation.
    # -----------------------------------------------------------------------
    drive_cycle = esf.drive_cycle_001(verbose=False)  # temperature 298.15 K
    cycle_numbers = np.linspace(1, 1000, 5)
    simulated = esf.drive_cycle_degradation_calculator(
        drive_cycle, prms, cycle_numbers=cycle_numbers
    )

    # -----------------------------------------------------------------------
    # Report: fitted vs true
    # -----------------------------------------------------------------------
    k_time_true = RATE_PER_DAY_TRUE / prms.reference_calendar_time
    rows = [
        ("sei_alpha", prms.sei_alpha, ALPHA_TRUE),
        ("sei_beta", prms.sei_beta, BETA_TRUE),
        ("deg_per_time_unit (1/day)", prms.deg_per_time_unit, RATE_PER_DAY_TRUE),
        ("deg_per_cycle", prms.deg_per_cycle, RATE_PER_CYCLE_TRUE),
        ("k_temperature_calendar", prms.k_temperature_calendar, K_TEMPERATURE_TRUE),
        ("k_soc_calendar", prms.k_soc_calendar, K_SOC_TRUE),
        ("k_1_time_calendar (1/s)", prms.k_1_time_calendar, k_time_true),
    ]
    print()
    print(f"{'parameter':<28} {'fitted':>14} {'true':>14}")
    print("-" * 58)
    for name, fitted, true in rows:
        print(f"{name:<28} {fitted:>14.6g} {true:>14.6g}")
    print()
    print("simulated drive-cycle degradation (fitted parameters):")
    print(simulated[["cycle_number", "loss", "soh"]].to_string(index=False))

    # sanity check - the test suite runs this script and the recovery must
    # hold, otherwise the documentation example is lying
    for name, fitted, true in rows:
        relative_error = abs(fitted - true) / abs(true)
        assert relative_error < 1e-3, f"{name}: {fitted} != {true}"

    return prms, simulated


if __name__ == "__main__":
    main()
