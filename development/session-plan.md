# Session plan

Working cycle: **(1)** review → plan → choose designs → break into sessions;
**(2)** pick a session, split into parts, implement + test each part; repeat (2)
until the session is done; then back to (1) to revise designs and the plan.

Labels refer to [2026-07-10_repo-review.md](2026-07-10_repo-review.md) and
[design-decisions.md](design-decisions.md).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Round 1 (planned 2026-07-10)

### Session 1 — Test foundation `[x]` (done 2026-07-10)
Goal: reproducible test runs and numerical tests that pin the core behaviour,
plus fixes for the small unambiguous bugs (each pinned by a test).

- [x] 1.1 Declare dev dependencies (`pytest`, `pytest-datadir`) in
      `pyproject.toml` dependency group; `uv run pytest` works (P6).
      Configure matplotlib Agg + stop tests writing PNGs into the repo
      (removed committed `tests/data/out/*.png`).
- [x] 1.2 `tests/test_base_models.py` (31 tests) — all stress-factor relations
      and the nonlinear degradation models: analytic values, x == x_ref edge
      cases, vector/scalar behaviour, monotonicity.
- [x] 1.3 `tests/test_cycle_counting.py` (9 tests) — triangle wave, nested
      ripple, constant SoC fallback; temperature/time/c-rate aggregation.
- [x] 1.4 `tests/test_degradation_pipeline.py` (17 tests) — per-stage unit
      tests, default vs model-set consistency, EoL round-trip, end-to-end
      regression with pinned numbers.
- [x] 1.5 `tests/test_parameters.py` (27 tests) + `tests/test_converters.py`
      (9 tests) — JSON round-trips, model resolution for every model set,
      `unpack_parameters` both paths, unit conversion.
- [x] 1.6 Bug fixes, each with a pinning test: B1 (`_convert_unit` scalar),
      B2 (duplicate parameter name in register), B3 (`to_short_frame`),
      B4 (public `simulate()`, un-broke `test_sei_fit.py`), B5 (docstring).
      New bugs found and fixed during the session:
      - **B7** integer input arrays truncated stress factors to whole numbers
        (`np.ones_like` inherited the int dtype in `_exponential`).
      - **B8** `calc_cycle_at_end_of_life` crashed on numpy >= 2
        (`float()` on a length-1 fsolve result).
      - **B9** `ESFParams.save_json`/`load_json` were not round-trip exact:
        pandas `to_json` truncates to 10 digits (4.14e-10 became 4e-10, a 3.4 %
        error in the calendar-time constant) and `read_json`'s parser is
        off-by-one-ULP; both replaced with the `json` module.
      - **B10** `load_json` crashed on the shipped
        `data/DegModelPara/esf_params.json` (legacy key `deg_per_cyc`);
        unknown keys are now skipped with a warning.

Result: full suite went from 26 passed / 1 failed to **119 passed, 1 xfailed**.

### Session 2 — Structure cleanup `[x]` (done 2026-07-10)
Goal: the wheel is installable and self-contained; no dead code in the package.

- [x] 2.1 Moved `lib/` → `esf/external/` (DD-2); updated the import in
      `cycle_counting_algorithm.py`. Verified: built the wheel, installed it in
      an isolated environment, ran the cycle counter (this was impossible
      before — P1 fixed).
- [x] 2.2 Moved `esf/model_IFE/` → `legacy/model_IFE/` (DD-3) with a
      `legacy/README.md`; fixed internal imports and
      `scripts/Degradation_estimation.py`; excluded `legacy` from builds.
      Also fixed a pre-existing numpy>=2 breakage in the legacy `peak_det`
      (`from numpy import NaN, Inf`). Verified imports resolve.
- [x] 2.3 Deleted `esf/models/_backup_multi_fit.py` (2114 loc, nothing
      imported it).
- [x] 2.4 Moved all `check_*` functions to `scripts/checks/`
      (`check_parameters.py`, `check_cycles.py`, `check_io_data.py`,
      `check_fitting_module.py`, `check_dst_cycle.py`); each verified to
      import. Library modules kept only library code
      (fitting.py 3021→2618 loc, io/data.py 1971→1654, parameters.py
      1055→989, dst_cycle.py 1081→827, cycles.py 700→566).
      Note: `rich.print` shadowing in library modules remains — handled with
      DD-5's "library code is silent" pass in Session 3/4.
- [x] 2.5 Data duplication: the copy inside the package left with
      `model_IFE` (2.2). Decision: `data/Ageing_Data_Org/` is the single
      source; `tests/data/` stays as deliberate pytest-datadir fixtures.

Result: full suite still green (119 passed, 1 xfailed); wheel verified
self-contained.

### Session 3 — Parameters & registry simplification `[x]` (done 2026-07-10)
Goal: one way to resolve a stress model; ~400 fewer lines in `parameters.py`.

- [x] 3.1 Single shared `UnitRegistry` in `esf/settings/units.py` (DD-8);
      `parameters.py` re-exports it, `cycles.py` uses it (the third copy is in
      `apps/`, which is out of scope).
- [x] 3.2 Structured registry keys (DD-7): `ModelRegister.models` keyed by
      `(regime, type, label)` tuples; `register()` requires unique
      `parameter_names` and rejects duplicate keys; `get()` is a direct
      lookup; `_get_model_object` collapsed from 20 lines of filtering to a
      dict lookup.
- [x] 3.3 Generic accessors on `ESFParams` (DD-6): `stress_model_label`,
      `stress_model`, `stress_model_function`, `stress_model_parameter_names`,
      `stress_model_parameter_values`, `stress_model_fit_parameters`, and
      `stress_models(regime)` for the active model set. The ~36 hand-written
      properties are deleted; the old names still resolve through
      `__getattr__` with a `DeprecationWarning`. `unpack_parameters()`,
      `pprint_model_set()`, and the fitting `set_mode`s /
      `_get_models_info` rewritten on top of the new API. This also fixed a
      latent copy-paste (the old `high_soc_calendar_stress_model` read the
      *cycling* label — benign with default labels, B11). 7 new tests.
- [x] 3.4 Units convention documented in the README (DD-9 target table);
      enforcement inside the fitting module deferred to Session 4/Round 2
      where the io/fit boundary is reworked.

### Session 4 — Fitting module round 1 `[x]` (done 2026-07-10)
Goal: fitting is verifiable, not yet redesigned.

- [x] 4.1 Synthetic-data recovery tests (`tests/test_fit_recovery.py`,
      9 tests): SoC + temperature stress factors (both regimes), SEI fit at
      reference conditions (calendar + cycling), `degradation_rates_fit`
      multi-temperature recovery, and the chained rates→stress-factor
      pipeline. Two bugs found and fixed with pinning tests:
      - **B12** unbounded `alpha_sei`/`beta_sei` crashed lmfit (exp overflow)
        or converged to degenerate unphysical solutions; physical bounds
        added.
      - **B13** `ModelItem.get_parameter_dict` returned a shallow copy, so
        fits corrupted the global registry defaults for later fits; deep
        copy + regression test.
- [x] 4.2 Fitting entry points are silent unless `verbose=True`; `echo`
      messages go to `logging.debug` (console only in verbose mode).
- [x] 4.3 Findings collected in
      [2026-07-10_fitting-review.md](2026-07-10_fitting-review.md)
      (F1–F8): unit brittleness at the io/fit boundary is the top item for
      Round 2, followed by fits-mutate-prms, the verbosity triad, and the
      untested DoD/time stress-factor path.

### Round 2 (planned; see 2026-07-10_fitting-review.md for details)
Re-review sessions, in suggested order:
1. **Units hardening** (F1, DD-9) `[x]` (done 2026-07-11): kelvin is the
   internal temperature unit everywhere.
   - `get_example_params` / `get_example_params_from_original_repo` now use
     kelvin (`reference_temperature=298.15`). The paper's k_temperature
     constants are kelvin-parameterized, so the previous degC references gave
     **wrong stress factors** whenever data/drive cycles were in kelvin
     (which the selectors and `drive_cycle_001` are). The pinned end-to-end
     regression was re-pinned: at reference temperature the temperature
     stress factor is now exactly 1.0 (it silently was ~4.9 before).
   - DST simulations and check scripts take kelvin; hardcoded temperature
     rate tables converted; the chained rates→stress-factor recovery test no
     longer needs a manual `convert_units` flip (that was the F1 symptom).
   - Also in this session: CI added (`.github/workflows/tests.yml`, PR #10).
   - Remaining F1 items for the fit-API session: selectors should *assert*
     unit consistency instead of trusting `prms.temperature_unit`, and
     `NonlinearFit.x_ref` should be derived from the selector's time unit.
2. **Fit API cleanup** (F2, F3, F5) `[~]` (partly done 2026-07-11):
   - F3 `[x]`: removed the dead `silent_mode` knob and `echo` `level`
     argument; `echo` is now logging-`debug` + verbose-`print`; fits emit no
     stdout unless `verbose=True`.
   - F2 `[~]`: added a non-mutating `fitted_parameters()` read-side to
     `StressFactorFit` and `NonlinearFit` (keyed by ESFParams attribute
     name); `update_prms` consumes it. Making the mutation opt-in at the
     top-level fit functions is deferred to the hierarchy-slimming session.
     Found & fixed **B14** (cycling rate written to a stray `deg_per_cyc`
     instead of the `deg_per_cycle` field).
   - F5 `[ ]`: parameter-override kwargs (`_update_parameter_dict`) not yet
     addressed — moves to the hierarchy-slimming session.
3. **Hierarchy slimming** (F6, F8) `[x]` (done 2026-07-11):
   - F8: removed dead placeholder classes (`HighSoCSFfit`, `RateSFfit`), the
     `rate_stress_factor_fit` stub, and the redundant `MultiFit` base
     (folded into `NonlinearMultiFit`).
   - F6: extracted all plotting into `esf/models/fit_plotting.py`; fit classes
     keep thin `plot_results` delegators. `fitting.py` 2621 → 2255 loc;
     plotting isolated in a 368-loc module. New plot smoke tests
     (`tests/test_fit_plotting.py`) pin the stress-factor and multi-fit paths.
   - **Remaining for a follow-up "fitting internals" session:** F4 (replace
     `startswith` regime parsing with the `Regime`/`DataType` enums), F5
     (discoverable parameter-override API instead of the stringly
     `_update_parameter_dict` kwargs), and F7 (DoD/time stress-factor study
     + tests — still the least-covered path).
4. **DoD/time stress-factor study + tests** (F7) `[x]` (done 2026-07-11):
   - DoD fit was silently broken (**B15**, cryptic unpack `TypeError`); now
     raises a clear `NotImplementedError`, dead `pass` scaffolding removed.
     The DoD *model* is verified to drive simulation.
   - Time stress factor (`time_stress_factor_calc`/`TimeSFCalc`) pinned with
     an exact recovery test. `tests/test_dod_time.py` (5 tests).
   - **F4 `[x]`** (done 2026-07-11): centralized the ~23 scattered
     `regime.startswith(...)` checks into `is_cycling`/`is_calendar` fit
     properties + `is_cycling_regime`/`is_calendar_regime` module helpers;
     `tests/test_fit_regime.py`.
   - **F5 `[x]`** (done 2026-07-12, in the finisher session): explicit
     `parameter_overrides` argument on `fit()` and the top-level single-fit
     functions (`sei_fit_at_reference_conditions`, `soc_stress_factor_fit`,
     `temperature_stress_factor_fit`). Strict validation: unknown parameter
     names and malformed values raise `ValueError` listing the valid names.
     Supports both the full-spec form (`{"x_ref": {"value": 1.0, "vary":
     False}}`) and the single-attribute form (`{"k__max": 0.9}`). The legacy
     `fit(**kwargs)` path still works and stays lenient (unknown keys are
     logged, since they may be unrelated kwargs).
   - **Future feature work (not cleanup):** actually implement the DoD fit —
     needs the modeling decisions listed under F7 in the fitting-review and
     reference validation.
5. **API finisher** `[x]` (done 2026-07-12):
   - Public API surface: `esf/__init__.py` now exports the two core
     workflows (data → fit → parameters; parameters + drive cycle →
     degradation) with `__all__` and a usage docstring; pinned by
     `tests/test_public_api.py`.
   - F5 parameter overrides (see above).
   - D7 (worst offender): `Data.__iter__` is a generator — nested/concurrent
     iterations are independent (the old implementation kept the position on
     the instance and silently broke nested loops); `_iterator_index`/`_uids`
     state removed.
6. README rewrite `[x]` (done 2026-07-12; closes Round 2):
   - README rewritten around the two workflows with verified quick-start
     examples, install (uv-first), units convention, layout, and an honest
     Status section replacing the stale MustDo/TODO lists.
   - Background material (provenance, rainflow notes, DST/aging data
     descriptions) moved to `docs/background-notes.md`.
   - Found & fixed in passing: the session-4 gating of the
     `sei_fit_at_reference_conditions` banner had been lost (a patch script
     aborted before writing; the banner printed regardless of `verbose`) —
     re-applied; and `_data_picker`'s data-quality prints are now
     `warnings.warn` (a missing strict column also raised a bare `KeyError`
     before reaching the intended strict-mode `ValueError`; fixed with
     `elif`).

### Round 3 (planned 2026-07-12; see 2026-07-12_round3-review.md)
Review done (findings G1–G9). Sessions:

1. **R3-1 — io.data cleanup** `[x]` (done 2026-07-12; io/data.py
   1642 → 1222 loc; suite 169 passed, 1 xfailed):
   - G1: deleted `_plot_data_deprecated` (~165 loc, unreferenced).
   - G2: `SampleData.get` is table-driven and raises `ValueError` for
     unsupported combinations (previously: silent `None`, plus one branch
     calling the never-existing `cycle_life_vs_soc`).
   - G4: `plot_data` (~240 loc) extracted to `esf/io/data_plotting.py`
     (`plot_sample_data`), thin delegator left on `SampleData` — same
     pattern as `fit_plotting`. `io/data.py` no longer imports matplotlib.
   - G5: deleted `average_data`, `_glob_data`, and the no-op base
     `Data.plot_data` stubs.
   - G6: pickle save/load round-trip pinned by a test (data + metadata
     survive) — first-ever coverage of the persistence path.
   - G7: `normalize_cols` failure goes to `logging` instead of print.
   - New tests in `tests/test_io.py`: `get` dispatch + raise, round trip,
     plot smoke.
2. **R3-2 — ESFParams custom-parameter round trip** `[x]` (done 2026-07-12,
   G9): custom parameters added with `update()` are recorded in a new
   `custom_parameter_names` field and restored by `load_json`/`from_json`
   (both now share `_from_parameter_mapping`). Stale/legacy keys that are
   *not* declared custom keep being dropped with a warning (B10 behaviour
   preserved). Decision on the `make_dataclass` mechanism: **kept** — custom
   parameters must be real dataclass fields for `asdict`/serialization, and
   the register's custom stress models need custom k-parameters. Fixed a
   latent bug while in there: repeated `update()` calls re-based the
   extended class on `ESFParams`, silently dropping earlier custom fields
   from serialization; the extension now bases on the current class.
   Suite: 173 passed, 1 xfailed.
3. **R3-3+ — feature work, needs owner input** `[ ]`: DoD fit (modeling
   decisions + reference data), uncertainty propagation, original-results
   reproduction (reference numbers + tolerances), new streamlit apps
   (drives `OperationalData` and final API shape).
4. **Fitting architecture document** `[x]` (owner-requested, 2026-07-12):
   [docs/fitting-architecture.md](../docs/fitting-architecture.md) — user
   walkthrough of the three-stage fitting flow + the design patterns and
   structure of the fitting code (registry/hub triangle, template-method
   hierarchy, contracts, extension recipe, gaps/direction). Backed by the
   runnable `scripts/examples/full_fitting_workflow.py`, executed by
   `tests/test_example_workflow.py` so the document cannot drift.
4. `io.data` simplification (D7), public API definition (`esf/__init__.py`
   exports), README rewrite, docs for the new apps to build on.

## Log

- 2026-07-10: Round-1 review done; baseline test run: 26 passed, 1 failed (B4),
  1 xfailed. Plan created. Session 1 started.
- 2026-07-10: Session 1 done (test foundation; suite 26→119 tests, 10 bugs
  found/fixed: B1–B5, B7–B10). Session 2 done (structure cleanup; wheel
  self-contained, legacy code out of the package, dead code removed, check
  scripts moved to `scripts/checks/`). ~2900 lines removed from the package.
  Everything left uncommitted for review — suggest committing Session 1 and
  Session 2 as separate commits.
- Plan revision after Round 1, Sessions 1–2: Sessions 3 and 4 stand as
  planned. Additions for Session 3 based on what Session 2 exposed:
  - 3.5 `[x]` "Library code is silent": removed `from rich import print`
    shadowing from `io/data.py`, `simulations/cycles.py`,
    `simulations/degradation.py`, and `models/fitting.py` (verbose output now
    uses the builtin print; full logging conversion goes with the fitting
    rework).
  - 3.6 `[x]` Decision on `ESFParams.update()`'s dynamic `make_dataclass`
    mutation: **kept for now** — custom parameters must be real dataclass
    fields for `asdict`/JSON round-trips to include them, and Session 4 will
    show how custom parameters are actually consumed by the fitting code.
    Revisit in Round 2.
  - 3.7 `[x]` README updated: structure section reflects the Session-2 moves
    (`esf/external/`, `legacy/`, `scripts/checks/`, `development/`), and the
    units convention (DD-9) is documented (full README rewrite still Round 2).
- 2026-07-10: Session 3 done. `parameters.py` 989→~840 loc with a single
  generic accessor path; registry keys structured; one pint registry; suite
  at 126 passed, 1 xfailed (internal code emits no deprecation warnings).
- 2026-07-10: Session 4 done. Recovery tests prove the fitting pipeline
  numerically (9 new tests); B12 + B13 fixed; fits silent by default;
  fitting-review findings (F1–F8) recorded and Round 2 planned. Suite at
  135 passed, 1 xfailed.
