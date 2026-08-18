# Fitting parameters

Fitting is **staged**: because the linear rate `f` is a product of independent
stress factors, you cannot fit everything from one data set. You pin the SEI
envelope first, then extract one rate per condition, then fit each stress
factor from data that varies only that condition.

```
aging data ──▶ SampleData ──▶ 1. SEI fit ──▶ 2. rates fit ──▶ 3. stress-factor fits ──▶ ESFParams
```

This page is the practical recipe; the
[fitting architecture](../fitting-architecture.md) explains the design and the
contracts between stages in depth.

## 0. Prepare the data

`SampleData` holds aging measurements. Feed it plain DataFrames plus a
`DataType` and the units they arrive in; the *selectors* return flat frames in
the [internal units](../reference/units.md).

```python
import esf

data = esf.SampleData()
data.add_data(
    frame,                                          # columns: t, SoH, T, SoC[, subset]
    data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
    time_unit="days",
    temperature_unit="K",
)
data.calculate_life_fraction()                      # adds L = 1 - SoH
```

## 1. SEI fit at reference conditions

Uses only data at the reference temperature and SoC, where every stress factor
is 1 and `f` is the reference rate. This is the only stage where `sei_alpha`
and `sei_beta` vary.

```python
prms = esf.get_example_params()
at_reference = data.calendar_life_vs_temperature(filter_value=298.15, strict_mode=False)
esf.sei_fit_at_reference_conditions(
    prms, at_reference, data_type=esf.DataType.CALENDAR_VS_TEMPERATURE
)
```

## 2. One degradation rate per condition

With the envelope frozen, every temperature (or SoC) series is refit for its
linear rate only.

```python
rates = esf.degradation_rates_fit(
    prms,
    data.calendar_life_vs_temperature(strict_mode=False),
    data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
)
```

## 3. Stress-factor fits

Each stress-factor fit normalizes the rates by the reference rate
(`S = f / f_ref`) and fits its model function.

```python
esf.temperature_stress_factor_fit(prms, rates)
esf.soc_stress_factor_fit(prms, rates_vs_soc, data_type=esf.DataType.CALENDAR_VS_SOC)
esf.time_stress_factor_calc(prms, rates, data_type=esf.DataType.CALENDAR_VS_TEMPERATURE)
```

### The DoD stress factor

DoD is special: its data is **cycle life `N` to end-of-life vs DoD**, and the
model output *is* the per-cycle rate (it is not normalized to 1 at a reference
DoD). The model **form follows the chemistry** (`battery_chemistry`): empirical
for LMO, exponential for LFP, quadratic for NMC.

```python
prms = esf.ESFParams(battery_chemistry="NMC")   # -> quadratic DoD form
esf.dod_stress_factor_fit(prms, frame, data_type=esf.DataType.CYCLE_VS_DOD)
```

With `is_at_reference=False` the temperature, SoC and calendar-time stress
factors are stripped out first (Xu et al. eqs. 20/31), which needs per-point
`T`, `SoC` and `t_cycle` columns.

## Results, overrides, verbosity

- By default fits write into `ESFParams` (and `mark_changed()` timestamps it).
  Pass `apply=False` to leave `prms` untouched and read the result via
  `fit.fitted_parameters()`.
- Every fit accepts `verbose=True` (report + plots) and a `parameter_overrides`
  dict, e.g. `parameter_overrides={"x_ref": {"value": 1.0, "vary": False}}`.
  Unknown parameter names raise with the list of valid ones.

```python
esf.ESFParams  # save/load round-trips exactly via prms.save_json(...) / load_json(...)
```
