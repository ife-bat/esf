# Uncertainty propagation — design

Status: **Tier 0 + Tier 1 implemented** (2026-07-13) in `esf/uncertainty.py`,
tested in `tests/test_uncertainty.py`, exported from the package
(`ParameterUncertainty`, `ParameterEnsemble`, `simulate_with_uncertainty`,
and `BaseFit.parameter_uncertainty()`). Tier 2 (bootstrap) and per-point
measurement-noise handling remain future work. Companion to the Round-4 plan
([2026-07-13_round4-plan.md](2026-07-13_round4-plan.md)).

Implementation notes / deviations from the sketch below:

- Uncertainty is pulled from a completed fit via `fit.parameter_uncertainty()`
  (returning a `ParameterUncertainty`) rather than a `return_uncertainty=True`
  flag on the facades — this keeps the facade return types unchanged and lets
  several fits' blocks be merged with `+`.
- `simulate_with_uncertainty(sim_fn, ensemble, sim_kwargs=..., prms_kw="prms")`
  injects the sampled parameters as a keyword (the simulators take `prms` as a
  named, not first, argument), and supports `align_on=<x column>` for
  variable-length outputs (DST) and `index_columns`/`value_columns` to shape
  the banded output.

Goal: turn "the fit found `k_temperature_calendar = 0.0693`" into
"`0.0693 ± 0.0009`", and turn a simulated `SoH(cycle)` curve into a curve with
a credible band — end to end, from the fits through the simulators.

## 1. Where uncertainty comes from, and where it must go

**Sources**

1. **Fitted parameters.** Every lmfit fit produces a covariance matrix
   (`ModelResult.covar`) and per-parameter `stderr`. This is the dominant,
   readily-available source and the focus of this design.
2. **Measurement noise in the aging data.** Feeds the fits; currently ignored
   (all `y_err = 1`). Second priority.
3. **Model-form error** (is the exponential SoC model even right?). Out of
   scope — not quantifiable without alternative models.

**Sinks** (what a user wants a band on)

- fitted parameters, reported as `value ± σ`;
- `drive_cycle_degradation_calculator` → `loss`/`soh` vs cycle;
- `DSTCycleDeg` → SoH vs cycle;
- `calc_cycle_at_end_of_life` → N_eol ± σ.

## 2. Why the current half-built approach cannot work as-is

`ESFParams` already has `mode="uncertainties"` and can hold
`uncertainties.ufloat` values, and `get_model_function(wrap_for_uncertainties=True)`
wraps a model with `uncertainties.wrap`. The intent was to push `ufloat`
values through the model functions and let the
[`uncertainties`](https://pythonhosted.org/uncertainties/) package propagate
them by first-order (linear) error propagation.

This is architecturally stuck for three concrete reasons:

1. **The pipeline has non-arithmetic steps.** Rainflow cycle counting
   (`CycleCounter`) and the `fsolve` in `calc_cycle_at_end_of_life` /
   `_calculate_n_eol` cannot accept `ufloat`s — `uncertainties` only traverses
   differentiable arithmetic. So a linear-propagation path can never cover the
   whole simulator; today it is short-circuited by `to_numpy_float64()` at
   every boundary (`unpack_parameters`, `calc_loss`, …), which silently erases
   the uncertainty. That is why `drive_cycle_degradation_calculator` warns and
   converts to floats.
2. **First order is not enough here.** The SEI envelope
   `1 − α·e^{−xβf} − (1−α)e^{−xf}` is strongly nonlinear; near end of life a
   first-order Taylor band is unreliable.
3. **Correlations are dropped.** Independent `ufloat`s assume the fitted `k`s
   are uncorrelated. They are not — lmfit returns a full covariance matrix, and
   using only the diagonal `stderr` can over- or under-state the output band by
   a large factor.

Conclusion: keep `ufloat` for **representation and reporting**, but do
**propagation by Monte Carlo**, not by pushing `ufloat`s through the pipeline.

## 3. Proposed approach: three tiers

### Tier 0 — capture and report (cheap, do first)

Make the fits *store* their uncertainty and report it; no simulation
propagation yet.

- After a fit, read `ModelResult.covar` and `params[name].stderr`.
- Extend `fitted_parameters()` with a sibling `fitted_parameters_with_std()` /
  `fit_covariance()` returning `(names, nominal, covariance)` keyed by the
  ESFParams attribute names.
- Store it next to `ESFParams`. Two options:
  - **(preferred)** a companion object `ParameterUncertainty` holding, per fit
    stage, the parameter names + covariance block. Keeps `ESFParams` a clean
    value object.
  - or reuse the existing `mode="uncertainties"` + `ufloat` fields for the
    *nominal ± std* (diagonal only), plus a separate covariance store.
- `ESFParams.pprint`/`to_short_frame` gain a `± std` column when present.

This alone is immediately useful (parameter tables with error bars) and lays
the groundwork for Tier 1. It uses the `uncertainties` package only for
formatting/round-trip of scalar `value ± std`.

### Tier 1 — parametric Monte Carlo (the workhorse)

Propagate the fitted covariance through the **unchanged float pipeline** by
sampling.

```
ParameterEnsemble(prms, uncertainty, n=1000, seed=...)
    -> draws n ESFParams, each a copy of prms with the fitted parameters
       replaced by a joint sample ~ MultivariateNormal(nominal, covariance),
       clipped/rejected to respect physical bounds (alpha in [0,1], beta>=0,
       k_2_dod<0, ...)

simulate_with_uncertainty(sim_fn, prms, ensemble, quantiles=(.05,.5,.95))
    -> runs sim_fn(sampled_prms) for each sample, aligns the output frames,
       returns median + lower/upper band per output column
```

Properties:

- **Correct to all orders** and through the non-arithmetic steps, because each
  sample is an ordinary float run of the existing simulator — no changes to
  `drive_cycle_degradation_calculator`, `CycleCounter`, `fsolve`, etc.
- **Respects within-stage correlations** by sampling the joint covariance
  (`numpy.random.multivariate_normal`, or `scipy.stats` / `uncertainties.
  correlated_values` to build the sampler).
- Non-invasive: a new module `esf/uncertainty.py` (sampler + `simulate_with_
  uncertainty` + a band-plot helper). The simulators stay float-only.

Cost: `n` simulations. `drive_cycle_degradation_calculator` on
`drive_cycle_001` is milliseconds, so `n = 1000` is trivial; DST simulations
are heavier — expose `n` and allow a smaller default there.

Known limitation (state it in the API docs): the stored covariance is
**block-diagonal across fit stages** — the SEI fit, SoC fit, T fit, and DoD
fit are separate lmfit runs, so Tier 1 treats them as independent. But stage 2
(rates) is fit with the stage-1 SEI parameters *frozen*, so there is a real
cross-stage correlation Tier 1 omits. It captures within-stage correlation
(e.g. the three `k_*_dod` together) correctly.

### Tier 2 — pipeline Monte Carlo / bootstrap (reference)

The fully-correct method, for validating Tier 1 and for cases where cross-stage
correlation or data noise matters.

- Either **bootstrap** the input aging data (resample rows / cells), or
  **perturb** it by its measurement noise, then **re-run the entire staged
  fit** for each replicate, then simulate.
- Captures cross-stage correlation *and* measurement-noise uncertainty
  automatically, because the whole fit→simulate chain is repeated.
- Expensive (each replicate is a full fit). Intended as an offline/validation
  tool and a research entry point, not the default.

## 4. Recommended sequencing

1. **Tier 0** — capture covariance in the fits, store in a
   `ParameterUncertainty` companion, report `± std`. Small, high value.
2. **Tier 1** — `esf/uncertainty.py`: ensemble sampler + `simulate_with_
   uncertainty` + band plotting. This is the deliverable most users want.
3. **Tier 2** — a bootstrap driver reusing the staged fit, plus a test that
   Tier 1's band agrees with Tier 2's on a case where the model is mild.

## 5. Testing strategy

- **Analytic check**: for a single closed-form stress factor evaluated at one
  point, Tier 1's output std must match first-order propagation
  (`uncertainties` package) within Monte Carlo error — a cross-check between
  the two methods where both are valid.
- **Coverage check**: generate synthetic data with known noise, fit, and
  confirm the Tier-1 band covers the true curve at ~the nominal rate
  (e.g. ~90% inside the 90% band over many trials).
- **Determinism**: seeded ensembles reproduce.
- **Tier1 vs Tier2**: agreement on a near-linear case; documented divergence
  on a strongly nonlinear one (near EOL).

## 6. API sketch (subject to Tier-0 implementation)

```python
prms, uncertainty = esf.sei_fit_at_reference_conditions(..., return_uncertainty=True)
uncertainty.report()                       # parameter table with +/- std

ensemble = esf.ParameterEnsemble(prms, uncertainty, n=1000, seed=0)
band = esf.simulate_with_uncertainty(
    esf.drive_cycle_degradation_calculator,
    prms, ensemble,
    sim_kwargs=dict(cycle_numbers=cycle_numbers),
    quantiles=(0.05, 0.5, 0.95),
)
band[["cycle_number", "soh_p05", "soh_p50", "soh_p95"]]
```

The `return_uncertainty` flag keeps the existing (float) API unchanged by
default — uncertainty is strictly opt-in, mirroring the `parameter_overrides`
and `apply` conventions elsewhere.

## 7. Decisions needed from the owner before building, with answers

1. **Cross-stage correlation**: is Tier-1 (block-diagonal, within-stage
   correlations only) acceptable as the default, with Tier-2 as the reference?
   (Recommendation: yes.) -> YES
3. **Measurement noise**: do we have per-point / per-cell error estimates for
   the aging data, or is fit-covariance-only acceptable initially? (Affects
   whether Tier 2 is bootstrap or noise-perturbation.) -> FIT-COVARIANCE-ONLY ACCEPTABLE,
   BUT DOCUMENT THAT WE PLAN TO IMPLEMENT ERROR ESTIMATE HANDLING PER-POINT
5. **Bounds handling** when sampling: clip to bounds, or reject-and-resample?
   (Recommendation: reject-and-resample; warn if the rejection rate is high,
   which signals a parameter rail against its bound.) -> REJECT-AND-RESAMPLE
6. **Default `n`** and whether DST simulations get a smaller default for cost. -> WE SET
   n = 1000. WE ALSO REDUCE THE DEFAULT FOR DST TO 100.
   
