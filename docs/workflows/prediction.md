# Predicting degradation

Given a set of parameters (`ESFParams`), `esf` predicts capacity loss for an
operating profile. There are two entry points depending on the input shape.

## From a drive cycle

A drive cycle is a frame with `time` (s), `soc` (0–1), `c-rate`, and
`temperature` (K). `drive_cycle_degradation_calculator` runs rainflow cycle
counting → stress factors → nonlinear loss.

```python
import numpy as np
import esf

prms = esf.get_example_params()
drive_cycle = esf.drive_cycle_001(verbose=False)

result = esf.drive_cycle_degradation_calculator(
    drive_cycle, prms, cycle_numbers=np.linspace(1, 1000, 20)
)
result[["cycle_number", "loss", "soh"]].tail()
```

`cycle_numbers` selects how many repetitions of the provided cycle to report
loss at. `esf.load_drive_cycle` builds a drive cycle from a CSV power profile.

## From real field data

`OperationalData` is the input for prediction from a **measured operating
trace** — state of charge, C-rate, temperature and (optionally) measured SoH
sampled over time. It converts a raw field-data frame to the
[internal units](../reference/units.md) and hands off the drive-cycle frame.

```python
import esf

op = esf.OperationalData.from_field_dataframe(
    field_frame,                      # e.g. DateTime, SOC (%), Crate (C), Temp (DegC), SOH (%)
    column_map={
        "datetime": "DateTime",
        "soc": "SOC (%)",
        "c_rate": "Crate (C)",
        "temperature": "Temp (DegC)",
        "soh": "SOH (%)",
    },
)

prediction = op.predict(prms)         # loss over the trace
measured = op.measured_soh()          # measured curve, for validation
```

`from_field_dataframe` converts time to seconds (from a datetime column or a
numeric time column in a given unit), temperature to kelvin, and SoC/SoH to
fractions; C-rate is unchanged. Required columns are validated with a clear
error.

## The DST case study

`DSTCycleDeg` reproduces the paper's Dynamic Stress Test degradation curves for
a given SoC window and temperature:

```python
from esf.simulations.dst_cycle import DSTCycleDeg

c = DSTCycleDeg(soc_min=25, soc_max=100, prms=esf.get_example_params(), temperature=293.15)
c.soh          # state of health (%) vs c.cycle_numbers
```

Run with the paper's Table I constants, this reproduces all seven published DST
curves within a few tenths of a percentage point (see the end-to-end regression
in the test suite).
