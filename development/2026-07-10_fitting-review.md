# Fitting module review (Session 4 findings, input to Round 2)

Collected while building the synthetic-data recovery tests
([test_fit_recovery.py](../tests/test_fit_recovery.py)). Labels continue the
numbering from [2026-07-10_repo-review.md](2026-07-10_repo-review.md).

## What now works (verified numerically)

- SoC and temperature stress-factor fits recover known constants exactly
  (calendar and cycling regimes).
- The SEI fit (`NonlinearFit`) recovers alpha/beta/rate exactly at reference
  conditions (calendar and cycling) — after B12 below.
- `degradation_rates_fit` (`NonlinearMultiFit`) recovers per-temperature
  degradation rates exactly, and chaining it into
  `temperature_stress_factor_fit` recovers the temperature constant.
- All fits are silent by default and report through `logging.debug`;
  `verbose=True` restores the console output.

## Bugs found and fixed in Session 4

- **B12 — unbounded SEI parameters**: `alpha_sei`/`beta_sei` varied without
  bounds. The optimizer wandered into exp-overflow territory, which either
  crashed lmfit ("The array returned by a function changed size between
  calls") or converged to unphysical, degenerate solutions
  (alpha ≈ 0.88 instead of 0.06). Fixed with physical bounds
  (alpha ∈ [0, 1], beta ∈ [0, 1e4]) in `NonlinearFit.default_parameter_dict`.
- **B13 — shared mutable registry defaults**: `ModelItem.get_parameter_dict`
  returned a *shallow* copy, so any fit that adjusted a parameter (e.g. the
  x_ref unit conversion in `StressFactorFit`) silently corrupted the registry
  defaults for every later fit in the same process. Found because the
  recovery tests failed only when run after the stress-factor tests. Fixed
  with a deep copy + regression test.

## Findings for the Round-2 redesign (F-labels)

- **F1 — unit brittleness at the io/fit boundary** (the big one; extends D6):
  - The stress-factor fits assume `prms.temperature_unit` describes the unit
    of the *data*, but the data selectors always re-unit temperatures to
    kelvin. A chained pipeline (`degradation_rates_fit` →
    `temperature_stress_factor_fit`) therefore requires the user to call
    `prms.convert_units(...)` in between — easy to forget, silently wrong
    numbers if forgotten (the fit normalizes against the wrong reference).
  - `NonlinearFit`'s `x_ref` comes from `prms.reference_calendar_time`
    (seconds). Selectors default to seconds so the default path is
    consistent, but selecting with `time_unit="days"` silently
    mis-parameterizes the model unless `x_ref` is overridden per fit.
  - The exponential temperature model is **not unit-covariant** (the
    `x_ref/x` factor differs between degC and K parameterizations): a `k`
    fitted against degC data is *not* the same constant as one fitted
    against kelvin data. Standardize on kelvin internally (DD-9) and treat
    degC only as an io/display unit.
- **F2 — fits mutate `prms` as a side effect** — *read-side done
  (2026-07-11)*: `fitted_parameters()` now returns the fitted values as a
  plain dict (keyed by ESFParams attribute name) on both `StressFactorFit`
  and `NonlinearFit`, without touching `prms`; `update_prms()` consumes it.
  Callers can read results without the side effect. *Remaining:* making the
  mutation opt-in at the top-level fit functions (an `apply=` flag) is a
  contract change deferred to the hierarchy-slimming session.
- **F3 — verbosity triad** — *done (2026-07-11)*: `silent_mode` and the
  `echo` `level` argument removed (both were dead after Session 4); `echo`
  is now a thin logging-`debug` + verbose-`print` helper. Fits emit zero
  stdout unless `verbose=True`.
- **F4 — stringly regime handling** — *done (2026-07-11)*: the ~23 scattered
  `self.regime.startswith("cycling")` / `startswith("calend")` checks are
  replaced by `is_cycling` / `is_calendar` fit properties and the module-level
  `is_cycling_regime` / `is_calendar_regime` helpers (used by the standalone
  fit functions and `NonlinearMultiFit`, which is not a `BaseProcessor`).
  Classification now lives in one place and reads as intent. `regime` still
  arrives as either a `Regime` enum or a decorated string
  (`"cycling_vs_temperature"`); the helpers accept both — only the
  cycling/calendar prefix is significant. Pinned by `tests/test_fit_regime.py`.
  (Full coercion of `regime` to a `Regime` enum at the boundary remains a
  further step, but the scattered string-matching — the actual smell — is
  gone.)
- **F5 — parameter overrides via kwargs** — *done (2026-07-12)*: explicit
  `parameter_overrides` argument on `fit()` and the top-level single-fit
  functions, applied through `apply_parameter_overrides(strict=True)`:
  unknown parameter names and malformed values raise `ValueError` listing
  the valid names. Full-spec (`{"x_ref": {...}}`) and single-attribute
  (`{"k__max": 0.9}`) forms. The legacy `fit(**kwargs)` path delegates to
  the same core with `strict=False` (unknown keys logged, since arbitrary
  kwargs flow through it). Pinned in `tests/test_public_api.py`.
- **F6 — plotting lives inside the fit classes** — *done (2026-07-11)*:
  the plotting bodies (`_create_fit_subplots`, `_plot_fit`, and the ~190-line
  `NonlinearMultiFit.plot_results` + `_legend_builder`) moved to
  `esf/models/fit_plotting.py`; the fit classes keep thin `plot_results`
  delegators. `fitting.py` shrank by ~360 lines. Guarded by new plot smoke
  tests for the stress-factor and multi-fit paths.
- **F7 — DoD / time stress-factor paths** — *studied & pinned (2026-07-11)*:
  - **Update (2026-07-13): the reference-conditions DoD fit is now
    implemented** (`DoDSFfit`), validated against the published LMO constants
    (~1.5% recovery). Decisions in
    `development/DoD-fitting-decisions-updated.md`. The non-reference path
    still raises. Below is the state that motivated the study.
  - **DoD fit is not implemented, and was silently broken (B15)**:
    `DoDSFfit.preprocess_degradation_rates` returned `None`, so the parent
    pipeline died with a cryptic `TypeError: cannot unpack non-iterable
    NoneType`. A trivial "fix" would fit the wrong orientation (the class set
    `x_col="N"`, `y_col="DoD"` against a model expecting DoD as the
    independent variable) and produce garbage. Since the correct procedure
    needs modeling decisions (independent-variable orientation, `deg_at_eol`
    rescaling at reference, removing other stress factors at non-reference),
    the fit now raises a clear `NotImplementedError` and the four dead `pass`
    helper stubs were removed. The empirical DoD *model* works and drives
    simulation via the `k_*_dod` parameters (pinned by a test).
  - **Time stress factor works and is exact**: the real path
    (`time_stress_factor_calc` → `TimeSFCalc`, not the deprecated
    `calculate_time_stress_factor`) recovers `k_t = deg_rate_per_time /
    reference_time` exactly on synthetic data (temperature stress cancels).
    Pinned by a recovery test. `calculate_time_stress_factor` remains
    `DeprecationWarning`-flagged as unreliable.
  - Tests in `tests/test_dod_time.py`.
- **F8 — dead/placeholder classes** — *done (2026-07-11)*: removed
  `HighSoCSFfit` and `RateSFfit` (both had a `pass` `__init__` that never
  called super, so they were unusable), removed the `rate_stress_factor_fit`
  stub, and folded the `MultiFit` base into `NonlinearMultiFit` (the base only
  supplied a trivial `__init__` and two `pass` methods). Nothing referenced
  them outside their own definitions.

## Bugs found later

- **B14 — cycling degradation rate written to a stray attribute** (found
  2026-07-11 while extracting `fitted_parameters`): `NonlinearFit.update_prms`
  wrote `prms.deg_per_cyc = …`, but the `ESFParams` field is `deg_per_cycle`.
  Setting an unknown attribute on the dataclass instance silently created a
  stray `deg_per_cyc`, so the canonical field never received the fitted
  cycling rate (and serialization/`to_short_frame` reported the stale
  default). A Session-4 test had passed only because it read the same stray
  name. Fixed by separating the lmfit parameter name (`deg_per_cyc`) from the
  ESFParams field name (`deg_per_cycle`) and writing via `prms.set(...)`.

## Suggested Round-2 session shape

1. Units hardening (F1) — kelvin/seconds internally, selectors and fits agree
   by construction; recovery tests extended to prove the chained pipeline
   works without manual unit flips.
2. Fit API cleanup (F2, F3, F5) — explicit results, logging, discoverable
   parameter overrides.
3. DoD/time stress-factor study + tests (F7), then hierarchy slimming
   (F4, F6, F8).
