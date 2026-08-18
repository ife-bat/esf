# Quickstart

Two self-contained examples: simulate degradation for a drive cycle, and fit
parameters from aging data. Both use only the top-level `import esf` surface.

## Simulate degradation for a drive cycle

```python
import numpy as np
import esf

prms = esf.get_example_params()                    # parameters from the paper
drive_cycle = esf.drive_cycle_001(verbose=False)   # time / soc / c-rate / temperature frame

result = esf.drive_cycle_degradation_calculator(
    drive_cycle, prms, cycle_numbers=np.linspace(1, 1000, 20)
)
print(result[["cycle_number", "loss", "soh"]].tail())
```

Any drive cycle works as long as the frame has `time` (s), `soc` (0–1),
`c-rate`, and `temperature` (K) columns. `esf.load_drive_cycle` reads CSV power
profiles.

## Fit parameters from aging data

The staged procedure: fit the nonlinear SEI model at reference conditions,
extract per-condition degradation rates, then fit each stress factor.

```python
import esf

data = esf.SampleData()
data.add_data(                       # frame with t / SoH / T columns
    frame,
    data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
    time_unit="days",
    temperature_unit="K",
)
data.calculate_life_fraction()

prms = esf.get_example_params()

# 1) SEI parameters at reference conditions (298.15 K)
esf.sei_fit_at_reference_conditions(
    prms, data.calendar_life_vs_temperature(filter_value=298.15, strict_mode=False)
)

# 2) one degradation rate per temperature
rates = esf.degradation_rates_fit(
    prms,
    data.calendar_life_vs_temperature(strict_mode=False),
    data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
)

# 3) the temperature stress factor from those rates
esf.temperature_stress_factor_fit(prms, rates)

prms.save_json("my_parameters.json")   # reload with esf.ESFParams.load_json(...)
```

See [Fitting parameters](workflows/fitting.md) for the full three-stage flow,
and the [fitting architecture](fitting-architecture.md) for why it is staged.

## Predict from real field data

```python
import esf

op = esf.OperationalData.from_field_dataframe(field_frame)  # SoC/C-rate/T/SoH over time
result = op.predict(prms)
```

See [Predicting degradation](workflows/prediction.md).
