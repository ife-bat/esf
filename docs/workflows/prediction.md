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
loss at. Both the cycling and the calendar contribution accumulate with the
repetitions: after `N` passes the linear degradation is `N * (f_c + f_t)`.

Two options control how the stress factors are evaluated:

- `calendar_mode="cycles"` (default) sums the calendar stress over the
  rainflow cycles, each with its own duration, mean SoC and mean temperature.
  `calendar_mode="profile"` evaluates it once for the whole drive cycle, from
  its duration and its time-averaged SoC and temperature (the formulation of
  Xu et al.).
- `weight_by_cycle_count=True` (default) multiplies each rainflow cycle's
  cycling stress by its count (0.5 for a half cycle), as in the cycle-ageing
  sum of Xu et al. `False` counts every rainflow cycle as a full cycle, which
  is what reproduces the paper's DST curves (see
  `development/dst-reproduction-reference.md`).

## From a battery power trace

Power versus time is not yet that frame. SoC and C-rate follow from the pack
energy and the sign of the power column, and the file has no temperature.
`drive_cycle_from_power_csv` takes those as arguments and returns time in
seconds, SoC as a fraction, C-rate (negative when discharging) and temperature
in kelvin. The defaults match a comma-separated file with columns
`Time (s)` and `Battery Power (W)`.

```python
drive_cycle = esf.drive_cycle_from_power_csv(
    "battery_power.csv",
    capacity_wh=200_000,       # pack energy, Wh
    initial_soc=0.9,
    temperature=25,            # °C
    positive="discharge",      # positive battery power leaves the pack
)
```

`drive_cycle_from_power` is the same conversion for arrays already in memory.
If the resulting SoC would leave 0–1, the call fails and reports the smallest
pack that would have stayed inside from that starting SoC. Measured terminal
power is used as recorded. Pass `source="traction"` only when the column is a
drivetrain request; `regen_efficiency` then scales the power returned to the
pack during regen and leaves discharge unchanged.

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

c = DSTCycleDeg(
    soc_min=25,
    soc_max=100,
    prms=esf.get_example_params(),
    temperature=293.15,
    degradation_kwargs={"weight_by_cycle_count": False},
)
c.soh          # state of health (%) vs c.cycle_numbers
```

Run with the paper's Table I constants and every rainflow cycle counted as a
full cycle (`weight_by_cycle_count=False`), this reproduces the seven published
DST curves within 1.2 percentage points, and the deepest window (100-25) within
3.0 (see the end-to-end regression in the test suite). With the default
half-cycle weighting the curves end 4–8 points higher.
