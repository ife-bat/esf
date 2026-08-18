# esf — empirical stress factor degradation model

[![tests](https://github.com/ife-bat/esf/actions/workflows/tests.yml/badge.svg)](https://github.com/ife-bat/esf/actions/workflows/tests.yml)

`esf` models Li-ion battery degradation as a product of empirical stress
factors (temperature, SoC, DoD, time) scaling a nonlinear SEI-driven capacity
loss. It does two things:

1. **Fitting** — extract the model parameters from calendar-aging and
   cycle-life data.
2. **Simulation** — predict capacity loss for a drive cycle (rainflow cycle
   counting → stress factors → loss), given a set of parameters.

Based on Xu et al., *"Modeling of Lithium-Ion Battery Degradation for Cell
Life Assessment"*, IEEE Trans. Smart Grid 9(2), 2018
(<https://ieeexplore.ieee.org/document/7488267>), adapted at IFE. Provenance,
the rainflow/stress-factor background, and the original data sets are
described in [docs/background-notes.md](docs/background-notes.md). The
fitting workflow (user walkthrough + the architecture of the fitting code)
is documented in
[docs/fitting-architecture.md](docs/fitting-architecture.md), with a
runnable end-to-end example in
[scripts/examples/full_fitting_workflow.py](scripts/examples/full_fitting_workflow.py).

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended — this is what CI uses):

```bash
git clone https://github.com/ife-bat/esf.git
cd esf
uv sync --dev
uv run pytest        # everything should pass
```

Or with pip: `pip install -e .` (add `--group dev` for the test dependencies)
in your environment of choice.

## Quick start

### Simulate degradation for a drive cycle

```python
import numpy as np
import esf

prms = esf.get_example_params()          # parameters from the Xu et al. paper
drive_cycle = esf.drive_cycle_001(verbose=False)   # example: time/soc/c-rate/temperature frame

result = esf.drive_cycle_degradation_calculator(
    drive_cycle, prms, cycle_numbers=np.linspace(1, 1000, 20)
)
print(result[["cycle_number", "loss", "soh"]].tail())

cycles_to_eol = esf.calc_cycle_at_end_of_life(
    result["f_c"].iloc[0], result["f_t"].iloc[0], prms, eol_soh=0.8
)
```

Any drive cycle works as long as the frame has `time` (s), `soc` (0–1),
`c-rate`, and `temperature` (K) columns; `esf.load_drive_cycle` reads CSV
power profiles.

### Fit parameters from aging data

The two-stage procedure: fit the nonlinear SEI model at reference conditions,
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

prms.save_json("my_parameters.json")   # -> esf.ESFParams.load_json(...)
```

Every fit accepts `verbose=True` (report + plots) and a `parameter_overrides`
dict for adjusting the lmfit parameters, e.g.
`parameter_overrides={"x_ref": {"value": 1.0, "vary": False}}` or
`{"k__max": 0.9}`. Unknown parameter names raise with the list of valid ones.

## Units convention

Internal units, everywhere, after data ingestion:

| Quantity | Unit |
|---|---|
| time | seconds |
| temperature | K |
| SoC / DoD / SoH / loss | fraction (0–1) |
| rate | C-rate |

Conversion happens only in `esf.io` (on the way in — `add_data` takes
`time_unit=` / `temperature_unit=` and the selectors re-unit to the internal
convention) and in plotting/reporting (on the way out). Reference values such
as the calendar-time reference (`x_ref = 86 400 s = 1 day`) are model
parameters, not hidden unit changes. All pint quantities must come from the
single shared registry (`from esf.settings.units import ureg, Q_`).

Note in particular that the exponential temperature stress model is **not**
invariant under a unit change (its `x_ref/x` factor differs between degC and
K parameterizations) — temperatures are kelvin, always.

## Repository layout

- `esf/` — the package
  - `models/` — model equations (`base_models`), fitting (`fitting`),
    plot rendering (`fit_plotting`), the model register, rainflow counting
  - `external/` — the vendored peak detector and the adapter onto the
    `rainflow` package (see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md))
  - `io/` — data containers (`SampleData`) and selectors
  - `simulations/` — drive-cycle and DST degradation simulation
  - `settings/` — `ESFParams`, enums, constants, the shared unit registry
  - `utils/` — converters
- `data/` — aging data and example parameter files
- `development/` — the working documents: repo review, design decisions,
  session plan (start here if you want to change the code)
- `docs/` — the documentation site (built with Zensical): background notes,
  the [fitting architecture](docs/fitting-architecture.md), and the workflow /
  reference pages
- `scripts/` — the runnable example
  (`scripts/examples/full_fitting_workflow.py`, executed by the tests)
- `tests/` — pytest suite (`uv run pytest`); runs in CI on linux and windows

## Status

All fitting stages are implemented and **verified numerically** (synthetic-data
recovery tests): the SEI fit (calendar and cycling), the SoC / temperature /
time / **DoD** stress-factor fits, the multi-condition degradation-rates fit
and the chained pipeline, rainflow counting, and the drive-cycle degradation
simulation (pinned end-to-end regression). Beyond the core:

- **DoD stress-factor fit** — both the reference-conditions path and the
  non-reference stress-removal path (Xu et al. eqs. 20/31); the model form is
  chemistry-selected (empirical/exponential/quadratic for LMO/LFP/NMC).
- **Uncertainty propagation** — opt-in Monte-Carlo bands over the fitted
  covariance (`ParameterUncertainty` / `ParameterEnsemble` /
  `simulate_with_uncertainty`); the simulators stay float-only.
- **Publication reproduction** — the seven DST degradation curves of the paper
  are reproduced within a stated tolerance (end-to-end regression test).
- **Prediction from field data** — `OperationalData` turns a measured operating
  trace into a prediction.

Still open: the interactive apps, uncertainty Tier 2 (bootstrap) + per-point
measurement noise, and LFP-specific validation. See the round plans under
`development/`.

Full documentation (install, workflows, units, API, architecture) is built with
[Zensical](https://zensical.org) from `docs/`
(`uv run --group docs zensical serve`).

The roadmap and the reasoning behind the current design live in
[development/session-plan.md](development/session-plan.md) and
[development/design-decisions.md](development/design-decisions.md).

## Citing and licensing

`esf` is MIT licensed ([LICENSE](LICENSE)). Third-party components and their
licenses are listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

If you use this model in published work, cite the paper it implements:

> B. Xu, A. Oudalov, A. Ulbig, G. Andersson and D. S. Kirschen, "Modeling of
> Lithium-Ion Battery Degradation for Cell Life Assessment," *IEEE
> Transactions on Smart Grid*, vol. 9, no. 2, pp. 1131–1140, March 2018.
> doi: [10.1109/TSG.2016.2578950](https://doi.org/10.1109/TSG.2016.2578950)
