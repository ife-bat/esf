# Uncertainty

Fitting produces not just parameter values but a **covariance**. `esf` can
capture it and propagate it through a simulation as quantile bands. Uncertainty
is opt-in — the simulators stay float-only and are never slowed down by it.

Design and scope are in
[`development/uncertainty-propagation-design.md`](https://github.com/ife-bat/esf/blob/main/development/uncertainty-propagation-design.md).

## Capture (Tier 0)

Every fit exposes the fitted covariance over its varying parameters, keyed by
`ESFParams` attribute name:

```python
fit = esf.temperature_stress_factor_fit(prms, rates, apply=False)
unc = fit.parameter_uncertainty()      # a ParameterUncertainty
unc.report()                           # value, std, relative std per parameter
```

## Propagate (Tier 1)

`ParameterEnsemble` samples the covariance (jointly, respecting within-fit
correlations; out-of-bounds samples are rejected and redrawn) to produce
`ESFParams` copies. `simulate_with_uncertainty` runs any simulation function
over the ensemble and returns quantile bands.

```python
import esf

band = esf.simulate_with_uncertainty(
    simulate_fn,          # e.g. a drive-cycle simulation returning a frame
    uncertainty,          # a ParameterUncertainty (possibly merged across fits)
    prms,
    n=1000,               # ensemble size (use ~100 for expensive DST runs)
)
```

The result carries the median and the requested quantile bands over the
simulated quantity, which you can plot as a shaded region.

## Scope

- The covariance is **block-diagonal across fit stages** — separate fits are
  treated as independent (captures within-stage correlations, e.g. the three
  `k_*_dod` together, but not cross-stage correlation from freezing earlier
  parameters).
- Only fit covariance is modelled so far; per-point measurement noise and a
  full pipeline bootstrap (Tier 2) are planned.

See the [public API](../reference/api.md) for
`ParameterUncertainty` / `ParameterEnsemble` / `simulate_with_uncertainty`.
