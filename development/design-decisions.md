# Design decisions

Living document. Revisited at the start of every planning round
(see [session-plan.md](session-plan.md)). Problem labels (P*, D*, B*) refer to
[2026-07-10_repo-review.md](2026-07-10_repo-review.md).

## Guiding principles

1. **Tests before refactors.** Every simplification is preceded by tests that pin
   the current numerical behaviour, so refactors are provably behaviour-preserving.
2. **The package is `esf` and nothing else.** Everything `esf` needs at runtime
   lives under `esf/`; everything it does not need (legacy code, raw data, scripts,
   apps) lives outside it.
3. **One name per concept.** One regime vocabulary, one unit registry, one place
   where a stress model is resolved from parameters.
4. **Library code is silent.** No `print` at import or call time; use `logging`.
   Script-like exploration goes in `scripts/` or notebooks, not module tails.
5. **Small public API.** Users should be able to do the two core jobs with a
   handful of imports: load data → fit → get `ESFParams`; and
   drive cycle + `ESFParams` → degradation frame.

## Decisions

### DD-1: Keep the current architecture layers, simplify within them
`io` (data in/out) → `models` (math + fitting) → `simulations` (apply the model),
with `settings` holding parameters/constants. This layering is sound; the
over-engineering is *inside* the layers, not between them. No big-bang rewrite.

### DD-2: Move `lib/` into the package as `esf/external/` (fixes P1)
`peak_det` and `rainflow` are third-party-ish vendored algorithms that `esf`
imports at runtime, so they must ship with the wheel. Keep a thin wrapper
(`CycleCounter`) as the only consumer. Delete the duplicate under
`esf/model_IFE/lib_IFE` together with DD-3.

### DD-3: Move `esf/model_IFE/` out of the package to `legacy/model_IFE/` (fixes P2)
It is a frozen reference implementation used only by
`scripts/Degradation_estimation.py`. Keep it runnable from its new location
(adjust the script's imports), but out of the `esf` namespace and out of the wheel.
Not deleted — it is the reference for "reproduce the original results".

### DD-4: Delete `_backup_multi_fit.py` (fixes P3)
Git history keeps it. (Verified: nothing imports it.)

### DD-5: Move `check_*` functions out of library modules (fixes P4, D5)
Destination: `scripts/checks/` (one file per source module), imported nowhere.
Library modules keep zero executable tails beyond `if __name__ == "__main__"`
demos of a few lines, or none at all.

### DD-6: One generic stress-model accessor on `ESFParams` (fixes D1)
Replace the ~40 generated-by-hand properties with:

```python
prms.stress_model(regime, factor)         # -> StressModel(function, parameter_values, parameter_names)
prms.stress_models(regime)                # -> list[StressModel] from the active model set
```

The old properties are kept as thin deprecated aliases for one transition period
(the streamlit apps and notebooks still use some of them), then removed.
`unpack_parameters()` in `simulations/degradation.py` collapses to a loop over
`stress_models(...)`.

### DD-7: Registry keys become structured, not stringly (fixes D2, D3)
Registry lookup key: `(regime, factor, label)` tuple (or a small frozen dataclass),
with `Regime` and `StressFactor` enums as the single vocabulary. `DataType` keeps
its CSV-friendly string values but maps to the same enums. Explicit
`parameter_names` become mandatory at registration (no `__code__.co_varnames`
introspection).

### DD-8: One `UnitRegistry` (fixes D4)
`esf.settings.units` owns `ureg`/`Q_`; everything else imports from there.
(`pint.UnitRegistry` construction is also a measurable chunk of import time.)

### DD-9: Units convention (fixes D6)
Internal units, everywhere, after data ingestion:
**time = seconds, temperature = K, SoC/DoD/SoH/loss = fraction [0-1], rate = C**.
Conversion happens only in `io` (on the way in) and in plotting/reporting (on the
way out). The calendar-time reference (`x_ref = 86 400 s = 1 day`) is a model
parameter, not a hidden unit change. Documented in the README once implemented.

### DD-10: Public `simulate()` on fits (fixes B4)
`BaseFit.simulate()` delegates to the existing `_simulate()`; the old
`simulate_primary` name stays gone.

### DD-11: Testing strategy
- pytest with `pytest-datadir` (already in use), declared in
  `[dependency-groups] dev` in `pyproject.toml` so `uv run pytest` just works (P6).
- Priority order:
  1. **Core math** (`base_models`) — analytic expectations, edge cases.
  2. **Cycle counting** — synthetic SoC profiles with known rainflow answers.
  3. **Degradation pipeline** — unit tests per stage + one end-to-end regression
     with pinned numbers.
  4. **Parameters** — round-trips, model resolution, both model-set paths.
  5. **Fitting** — keep existing smoke tests, add convergence checks on synthetic
     data generated from known parameters (fit must recover them).
- Tests must not write into the repo (fix `tests/data/out/` usage) and must not
  need a display (matplotlib `Agg`).
- Bug fixes land together with a test that fails before / passes after.

### DD-12: What we do NOT do now
- No rewrite of `fitting.py` class hierarchy this round — pin behaviour first;
  simplification of the fit classes is a later planning round.
- No touching `apps/` (new apps come later; old ones stay as reference).
- No new features (uncertainty propagation, new stress factors) until the
  cleanup rounds are done.
