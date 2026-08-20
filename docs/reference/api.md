# Public API

Everything below is importable straight from the package (`import esf`). It is
the supported surface (`esf.__all__`); anything imported from a submodule is
internal and may change without notice.

## Parameters & vocabulary

| Name | What it is |
|---|---|
| `ESFParams` | The parameter container (dataclass). `save_json` / `load_json` round-trip exactly. The DoD model form follows `battery_chemistry`. |
| `get_example_params()` | Parameters from the Xu et al. paper (Table I). The reproduction config. |
| `get_example_params_from_original_repo()` | An alternative parameterization from the original code (does *not* reproduce the paper figures). |
| `drive_cycle_002()` | A second example drive cycle, alongside `drive_cycle_001()`. |
| `dst_cycles_from_experimental_data()` | The digitised DST curves from the publication; defaults to the bundled data. |
| `dst_cycles_experimental_data_v_lims()` | The seven SoC windows `(soc_min, soc_max)` those curves were measured over. |
| `DataType` | Enum for data-set kinds (`CALENDAR_VS_TEMPERATURE`, `CYCLE_VS_DOD`, …). |
| `Regime` | Enum: `CALENDAR`, `CYCLING`, `OPERATIONAL`. |

## Data handling

| Name | What it is |
|---|---|
| `SampleData` | Container for aging-test data (for **fitting**). `add_data`, `calculate_life_fraction`, and the selectors (`calendar_life_vs_temperature`, `cycle_life_vs_dod`, …). |
| `OperationalData` | Container for a measured operating trace (for **prediction**). `from_field_dataframe`, `to_drive_cycle`, `predict`, `measured_soh`. |
| `example_sample_data()` | Ready-made `SampleData` from the bundled example data. |

## Fitting

| Name | What it is |
|---|---|
| `sei_fit_at_reference_conditions(prms, data, …)` | Stage 1 — the SEI envelope (`sei_alpha`, `sei_beta`) at reference conditions. |
| `degradation_rates_fit(prms, selection, …)` | Stage 2 — one linear rate per condition, with the envelope frozen. |
| `temperature_stress_factor_fit(prms, rates, …)` | Stage 3 — the temperature stress factor. |
| `soc_stress_factor_fit(prms, rates, …)` | Stage 3 — the SoC stress factor. |
| `dod_stress_factor_fit(prms, frame, …)` | Stage 3 — the DoD stress factor (chemistry-selected form; reference and non-reference paths). |
| `time_stress_factor_calc(prms, rates, …)` | Stage 3 — the time stress factor (a calculation, not a fit). |
| `NonlinearFit`, `NonlinearMultiFit` | The underlying fit classes (for advanced use). |

Every facade takes `apply=True` (write into `prms`) / `apply=False` (read via
`fitted_parameters()`), `verbose=`, and `parameter_overrides=`.

## Simulation

| Name | What it is |
|---|---|
| `drive_cycle_degradation_calculator(df, prms, cycle_numbers=…)` | Predict loss for a drive-cycle frame (`time`, `soc`, `c-rate`, `temperature`). |
| `drive_cycle_001(…)` | An example drive cycle. |
| `load_drive_cycle(…)` | Build a drive cycle from a CSV power profile. |
| `calc_cycle_at_end_of_life(f_c, f_t, prms, eol_soh=…)` | Cycles to a given end-of-life SoH. |
| `DSTCycleDeg(soc_min, soc_max, prms, temperature)` | The DST case-study simulator (`.soh`, `.cycle_numbers`). |

## Uncertainty propagation

| Name | What it is |
|---|---|
| `ParameterUncertainty` | Fitted values + joint covariance, keyed by `ESFParams` names. `report()`, `sample()`. |
| `ParameterEnsemble` | Samples a `ParameterUncertainty` into `ESFParams` copies. |
| `simulate_with_uncertainty(fn, uncertainty, prms, n=…)` | Run a simulation over the ensemble; return quantile bands. |

See [Uncertainty](../workflows/uncertainty.md).
