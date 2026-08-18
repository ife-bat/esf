# Repository review (2026-07-10)

Full read-through of the `esf` package, tests, scripts, and supporting folders.
This document records the current state and the problems found. The decisions on
what to do about them live in [design-decisions.md](design-decisions.md), and the
work breakdown lives in [session-plan.md](session-plan.md).

## 1. What the repository does

Empirical stress factor (ESF) degradation model for Li-ion batteries, based on
Xu et al. (IEEE Trans. Smart Grid, 2018). Two main capabilities:

1. **Fitting** empirical stress-factor models (SoC, temperature, DoD, time, high-SoC)
   to calendar- and cycle-aging data (`esf/models/fitting.py`, `esf/io/data.py`).
2. **Simulation** of degradation for drive cycles / DST cycles using fitted
   parameters (`esf/simulations/`), driven by rainflow cycle counting
   (`esf/models/cycle_counting_algorithm.py` + `lib/`).

Core math lives in `esf/models/base_models.py` (stress-factor relations and the
nonlinear SEI degradation model) and is small and clean. The complexity sits in
the plumbing around it.

## 2. Layout inventory

| Path | Role | State |
|---|---|---|
| `esf/models/base_models.py` (257 loc) | Core math | Good; a few doc bugs |
| `esf/models/register_models.py` (297 loc) | Model registry | Works; stringly-typed keys, copy-paste bug |
| `esf/models/fitting.py` (3021 loc) | Fit classes | Works; ~700 loc of `check_*` script code inside |
| `esf/models/_backup_multi_fit.py` (2114 loc) | Old backup | Dead code |
| `esf/models/cycle_counting_algorithm.py` (200 loc) | Rainflow wrapper | Imports top-level `lib` (packaging bug); two near-identical methods |
| `esf/io/data.py` (1971 loc) | Data containers | Works; ~450 loc of `check_*`/hardcoded-data functions |
| `esf/settings/parameters.py` (1055 loc) | `ESFParams` + enums | Works; ~400 loc of repetitive properties |
| `esf/simulations/{cycles,degradation,dst_cycle}.py` | Simulation | Works; `check_*` script code embedded |
| `esf/model_IFE/` | Vendored legacy IFE model | Standalone; ships data/pickles/xlsx inside package; own copy of `lib` |
| `lib/` | peak_det + rainflow | Top-level package, required by `esf` at runtime |
| `apps/` | Streamlit apps | Out of scope (keep, will be rewritten) |
| `scripts/` | Check/demo scripts | `Degradation_estimation.py` does `sys.path` hacks into `esf/model_IFE` |
| `data/` | Aging data + params | Duplicated in `esf/model_IFE/degradation_model/data/` and partly in `tests/data/` |
| `tests/` | pytest suite (27 tests) | Mostly smoke tests; 1 failing |

## 3. Problems found

### 3.1 Packaging / structure

- **P1 — broken wheel**: `esf.models.cycle_counting_algorithm` does
  `import lib.peak_det...`, but `pyproject.toml` only packages `esf`. An installed
  (non-editable) `esf` cannot import its own cycle counter. `lib/` must move into
  the package (e.g. `esf/external/`).
- **P2 — vendored legacy model inside the package**: `esf/model_IFE/` contains a
  complete, self-contained old implementation including CSV data, `.pkl` parameter
  files, an `.xlsx`, `Backup/` folders, and a second copy of `lib/`. It is only used
  by `scripts/Degradation_estimation.py` (via `sys.path` manipulation). It bloats the
  wheel and confuses the package namespace.
- **P3 — dead code**: `esf/models/_backup_multi_fit.py` (2114 loc) is an explicit
  backup file kept in version control; git already remembers it.
- **P4 — script code inside library modules**: `fitting.py`, `io/data.py`,
  `dst_cycle.py`, `cycles.py`, `parameters.py` all end with large `check_00x()` /
  `check_*()` functions (~1400 loc total). These are manual experiments, not tests.
- **P5 — data duplication**: the same aging CSVs exist in `data/Ageing_Data_Org/`,
  `esf/model_IFE/degradation_model/data/`, and (as fixtures, fine) `tests/data/`.
- **P6 — test deps not declared**: `pytest`/`pytest-datadir` only exist in
  `dev_requirements.txt`; `uv run pytest` fails out of the box even though a
  `uv.lock` exists.

### 3.2 Over-done design (simplification targets)

- **D1 — `ESFParams` property explosion**: ~40 near-identical properties
  (`<sf>_<regime>_stress_model`, `..._function`, `..._parameters`,
  `..._parameters_values` for every stress factor × regime). One generic accessor
  `get_stress_model(regime, sf)` returning `(function, parameter_values)` would
  replace ~400 lines. `unpack_parameters()` in `degradation.py` then shrinks too.
- **D2 — stringly-typed model registry**: registry keys look like
  `"Cycling:Empirical||empirical_dod_k_relation(dod)"` and are re-assembled by
  string formatting in three different places (`register_models.py`,
  `parameters.py:_get_model_object`, `degradation.py:unpack_parameters`). Parameter
  names are inferred from `__code__.co_varnames`, which is fragile.
- **D3 — parallel naming conventions**: the same concept is called
  `model_regime` ("Calendar"/"Cycling"), `Regime` enum ("calend"/"cycling"),
  regime strings in fits ("calend_vs_temperature"), and `DataType`
  ("cal-vs-temp"). Mapping between them is done with `startswith`/`endswith`
  string surgery.
- **D4 — three separate pint `UnitRegistry` instances** (`esf/settings/parameters.py`,
  `esf/simulations/cycles.py`, `apps/settings.py`). Quantities from different
  registries cannot interoperate, and each registry costs import time. One shared
  registry in one module is enough.
- **D5 — import-time side effects**: `esf/__init__.py` builds the global model
  register `_mr` on import; `check_*` blocks and `rich.print` shadowing built-in
  `print` in library modules produce console output from library calls.
- **D6 — unit inconsistency** (README "MustDo"): calendar time parameters are per
  second in some places and per day in others (`reference_calendar_time = 86400 s`,
  `x_ref=86_400` in `nonlinear_cal_model`, but `reference_time = 1.0`). Needs one
  documented convention with explicit conversion at the boundaries.
- **D7 — `Data` iterator state on the object** (`_iterator_index` on the instance,
  `__iter__` returning `self`) — nested/concurrent iteration breaks; a generator
  would be simpler and correct.

### 3.3 Concrete bugs found while reading

- **B1** `esf/simulations/cycles.py:217` — `_convert_unit()` scalar branch uses the
  `time` *module* instead of `value`: `Q_(time, unit_from)` → `TypeError` for any
  scalar input.
- **B2** `esf/models/register_models.py:198-203` — Cycling/Linear SoC model registers
  `parameter_names` with `k_linear_1_soc_cycling` listed **twice**;
  `k_linear_2_soc_cycling` is never mapped.
- **B3** `esf/settings/parameters.py:447` (`to_short_frame`) — row
  `"k_temperature_calendar"` reads `self.k_temperature_cycling` (copy-paste).
- **B4** `tests/test_sei_fit.py` — calls `simulate_primary()`, which was renamed
  (commit f7248f2); public `BaseFit.simulate()` raises `NotImplementedError`, so
  there is currently no public simulate on single fits. Test fails.
- **B5** `esf/models/base_models.py:209` — `constant_time_k_relation` docstring says
  "k_1: not used / k_2: the offset", but the function returns `k_1` (the registry
  varies `k_1` and fixes `k_2=0`, so the *docstring* is wrong, not the code).
- **B6** `esf/models/base_models.py:71-87` — `_exponential_non_zero` and
  `_exponential` are duplicates (the former is unused); both silently return 1.0
  where `x == x_ref` element-wise, which is correct only because
  `exp(k*(x_ref-x_ref)) == 1`, but the masking breaks for scalar inputs
  (`np.ones_like(scalar)` is 0-d; indexing with a mask works but returns 0-d edge
  cases). Needs pinning tests.

Found later during Session 1 (all fixed, see session-plan.md):

- **B7** `_exponential` in `base_models.py` truncated results to integers when
  given integer input arrays (e.g. temperatures in whole kelvin) because
  `np.ones_like(x)` inherited the integer dtype.
- **B8** `calc_cycle_at_end_of_life` crashed on numpy >= 2: `float()` on the
  length-1 array returned by `fsolve`.
- **B9** `ESFParams.save_json`/`load_json` were not round-trip exact (pandas
  `to_json` default `double_precision=10`; `read_json` parser off by one ULP).
- **B10** `load_json` crashed on the shipped `data/DegModelPara/esf_params.json`
  because it contains the legacy key `deg_per_cyc`.
- (Design, for Round 2:) `ESFParams.update()` mutates `self.__class__` with
  `make_dataclass` to absorb unknown keys — surprising and hard to reason
  about; candidate for removal or replacement with an explicit
  `custom_parameters` dict (relates to D1).

### 3.4 Test suite state (baseline 2026-07-10)

`uv run --with pytest --with pytest-datadir pytest -q`:
**26 passed, 1 failed (B4), 1 xfailed.**

Existing tests are mostly smoke tests (run and assert "not empty"), several
produce matplotlib figures and PNG files (`tests/data/out/` is written into the
repo). There are **no numerical tests** of:

- the stress-factor relations and the nonlinear degradation model (core math),
- the rainflow cycle counter,
- the degradation pipeline (`count_cycles` → `calculate_stress_factors` →
  `calculate_linear_degradation_rates` → `calc_loss`),
- `ESFParams` round-trips (JSON, units, model resolution),
- the converters.

Those are the highest-value tests to add first: they pin today's behaviour so
that the simplification refactors (D1–D7) can be done safely.
