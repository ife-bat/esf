"""Uncertainty propagation for the esf model.

Design and rationale: ``development/uncertainty-propagation-design.md``.

Two tiers are implemented here:

- **Tier 0 — capture/report.** :func:`extract_uncertainty` reads an lmfit
  fit's covariance and per-parameter standard errors and returns a
  :class:`ParameterUncertainty` keyed by ``ESFParams`` attribute names.
  ``BaseFit.parameter_uncertainty()`` is the convenience entry point.
- **Tier 1 — parametric Monte Carlo.** :class:`ParameterEnsemble` samples the
  fitted covariance (joint, respecting within-fit correlations; samples that
  violate parameter bounds are rejected and redrawn), producing ``ESFParams``
  copies. :func:`simulate_with_uncertainty` runs any simulation function over
  the ensemble and returns quantile bands — the simulators stay float-only and
  are not modified.

Scope note (per the design decisions): the covariance is **block-diagonal
across fit stages** — separate fits are treated as independent, which captures
within-stage correlations (e.g. the three ``k_*_dod`` together) but not the
cross-stage correlation from freezing earlier parameters. Measurement noise in
the aging data is not yet modelled (fit-covariance only); per-point error
handling is planned. Tier 2 (bootstrap of the staged fit) is the reference for
both and is not implemented here.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# lmfit -> ESFParams attribute names for the nonlinear (SEI) fit, whose model
# parameters are not carried on an esf ModelItem
_NONLINEAR_NAME_MAP = {
    "alpha_sei": "sei_alpha",
    "beta_sei": "sei_beta",
    "deg_per_cyc": "deg_per_cycle",
    "deg_per_time_unit": "deg_per_time_unit",
}

_DEFAULT_N = 1000  # DST simulations should use a smaller n (e.g. 100) for cost


@dataclass
class ParameterUncertainty:
    """Fitted parameters with their joint covariance, keyed by ESFParams names.

    Attributes:
        names: ESFParams attribute names of the (varying) parameters.
        nominal: the fitted values, one per name.
        covariance: the (k, k) covariance matrix over ``names``.
        bounds: (min, max) per name, used when sampling (reject-and-resample).
    """

    names: list[str]
    nominal: np.ndarray
    covariance: np.ndarray
    bounds: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.nominal = np.asarray(self.nominal, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        if not self.bounds:
            self.bounds = [(-np.inf, np.inf)] * len(self.names)
        k = len(self.names)
        if self.nominal.shape != (k,) or self.covariance.shape != (k, k):
            raise ValueError(
                f"shape mismatch: {k} names, nominal {self.nominal.shape}, "
                f"covariance {self.covariance.shape}"
            )

    @property
    def std(self) -> dict[str, float]:
        """Standard deviation (sqrt of the covariance diagonal) per name."""
        diag = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        return dict(zip(self.names, diag, strict=False))

    def report(self) -> pd.DataFrame:
        """A table of value, std, and relative std per parameter."""
        std = self.std
        rows = []
        for name, value in zip(self.names, self.nominal, strict=False):
            s = std[name]
            rows.append(
                {
                    "parameter": name,
                    "value": value,
                    "std": s,
                    "rel_std": (s / abs(value)) if value else np.nan,
                }
            )
        return pd.DataFrame(rows).set_index("parameter")

    def __add__(self, other: ParameterUncertainty) -> ParameterUncertainty:
        """Block-diagonal merge (the two fits are treated as independent)."""
        if not isinstance(other, ParameterUncertainty):
            return NotImplemented
        overlap = set(self.names) & set(other.names)
        if overlap:
            raise ValueError(f"cannot merge: parameters appear in both: {overlap}")
        k1, k2 = len(self.names), len(other.names)
        covariance = np.zeros((k1 + k2, k1 + k2))
        covariance[:k1, :k1] = self.covariance
        covariance[k1:, k1:] = other.covariance
        return ParameterUncertainty(
            names=list(self.names) + list(other.names),
            nominal=np.concatenate([self.nominal, other.nominal]),
            covariance=covariance,
            bounds=list(self.bounds) + list(other.bounds),
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` joint samples, redrawing any that violate the bounds.

        Returns an ``(n, k)`` array. Warns if bounds rejection is severe (a
        sign that a parameter is railed against a bound).
        """
        lower = np.array([b[0] for b in self.bounds])
        upper = np.array([b[1] for b in self.bounds])
        accepted = np.empty((0, len(self.names)))
        draws = 0
        max_draws = 100 * max(n, 1)
        while len(accepted) < n and draws < max_draws:
            batch = rng.multivariate_normal(self.nominal, self.covariance, size=n)
            draws += n
            ok = np.all((batch >= lower) & (batch <= upper), axis=1)
            accepted = np.vstack([accepted, batch[ok]])
        if len(accepted) < n:
            raise RuntimeError(
                f"could only draw {len(accepted)}/{n} in-bounds samples after "
                f"{draws} draws; a parameter is likely railed against its bound"
            )
        rejection = 1.0 - n / draws
        if rejection > 0.5:
            warnings.warn(
                f"high bounds-rejection rate ({rejection:.0%}) when sampling; "
                "a fitted parameter is close to a bound"
            )
        return accepted[:n]


def extract_uncertainty(fit) -> ParameterUncertainty:
    """Build a :class:`ParameterUncertainty` from a completed fit.

    Reads the lmfit covariance over the *varying* parameters and maps the
    lmfit parameter names to ESFParams attribute names. Parameters that do not
    map to an ESFParams field (e.g. a fixed ``x_ref``) are dropped.
    """
    result = fit.fit_result.get_fit()
    if getattr(result, "covar", None) is None:
        raise ValueError(
            "the fit did not produce a covariance estimate (result.covar is "
            "None); uncertainty cannot be extracted"
        )

    name_map = _lmfit_to_esf_name_map(fit)
    var_names = list(result.var_names)
    # keep only varying params that map to an ESFParams field, preserving order
    keep = [(i, n) for i, n in enumerate(var_names) if name_map.get(n)]
    if not keep:
        raise ValueError("no varying parameters map to ESFParams fields")
    idx = [i for i, _ in keep]
    esf_names = [name_map[n] for _, n in keep]

    covariance = np.asarray(result.covar, dtype=float)[np.ix_(idx, idx)]
    nominal = np.array([result.params[var_names[i]].value for i in idx])
    bounds = [
        (result.params[var_names[i]].min, result.params[var_names[i]].max)
        for i in idx
    ]
    return ParameterUncertainty(esf_names, nominal, covariance, bounds)


def _lmfit_to_esf_name_map(fit) -> dict[str, str]:
    """Map the fit's lmfit parameter names to ESFParams attribute names."""
    esf_model = getattr(fit, "_esf_model_object", None)
    if getattr(fit, "is_stress_factor_fit", False) and esf_model is not None:
        # model.param_names and esf_parameters[1:] are both in signature order
        return dict(
            zip(fit.model.param_names, esf_model.esf_parameters[1:], strict=False)
        )
    # nonlinear (SEI) fit
    return dict(_NONLINEAR_NAME_MAP)


class ParameterEnsemble:
    """A Monte Carlo ensemble of ``ESFParams`` sampled from fit uncertainty.

    Args:
        prms: the nominal parameter set; sampled parameters overwrite their
            entries, everything else is kept.
        uncertainty: the (possibly merged) :class:`ParameterUncertainty`.
        n: ensemble size (default 1000; use ~100 for DST simulations).
        seed: seed for the RNG (for reproducibility).
    """

    def __init__(
        self,
        prms,
        uncertainty: ParameterUncertainty,
        n: int = _DEFAULT_N,
        seed: int | None = None,
    ):
        self.prms = prms
        self.uncertainty = uncertainty
        self.n = int(n)
        rng = np.random.default_rng(seed)
        self._samples = uncertainty.sample(self.n, rng)

    def __len__(self) -> int:
        return self.n

    @property
    def samples(self) -> np.ndarray:
        """The raw ``(n, k)`` sample array."""
        return self._samples

    def __iter__(self):
        for row in self._samples:
            sampled = self.prms.copy()
            for name, value in zip(self.uncertainty.names, row, strict=False):
                sampled.set(name, float(value))
            yield sampled


def simulate_with_uncertainty(
    sim_fn: Callable,
    ensemble: ParameterEnsemble,
    *,
    sim_kwargs: dict | None = None,
    prms_kw: str = "prms",
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    value_columns: Sequence[str] | None = None,
    index_columns: Sequence[str] | None = None,
    align_on: str | None = None,
) -> pd.DataFrame:
    """Run ``sim_fn`` over the ensemble and return quantile bands.

    ``sim_fn`` is called once per ensemble member with the sampled parameters
    injected as the ``prms_kw`` keyword and everything else from
    ``sim_kwargs``; the float simulator is not modified. For each numeric
    output column a set of quantile columns ``<col>_pNN`` is produced.

    Args:
        sim_fn: a simulation function returning a DataFrame, taking the
            parameters as the ``prms_kw`` keyword (e.g.
            :func:`esf.drive_cycle_degradation_calculator`, called as
            ``sim_fn(df=..., prms=sampled, cycle_numbers=...)``). For anything
            that is not a plain function of ``prms`` (e.g. ``DSTCycleDeg``),
            wrap it in a small ``lambda``.
        ensemble: the :class:`ParameterEnsemble`; its ``prms`` is the nominal
            reference.
        sim_kwargs: the other keyword arguments forwarded to ``sim_fn`` (e.g.
            the drive-cycle frame and ``cycle_numbers``).
        prms_kw: the keyword under which to pass the sampled parameters.
        quantiles: the quantiles to report (default 5/50/95%).
        value_columns: columns to band; default is every float column. The
            other columns are taken from the nominal run as the index.
        index_columns: columns to carry through from the nominal run as the
            index (default: all columns that are not banded). Pass e.g.
            ``["cycle_number"]`` to keep the output narrow.
        align_on: if the per-sample outputs differ in length (e.g. DST runs to
            a variable cycle count), the name of the x-column to interpolate
            each output onto the nominal run's grid before taking quantiles.

    Returns:
        A DataFrame with the index columns plus ``<col>_pNN`` bands.
    """
    sim_kwargs = dict(sim_kwargs or {})

    def run(parameters):
        return sim_fn(**{**sim_kwargs, prms_kw: parameters})

    reference = run(ensemble.prms)
    if value_columns is None:
        value_columns = [
            c for c in reference.columns if pd.api.types.is_float_dtype(reference[c])
        ]
        if align_on in value_columns:
            value_columns.remove(align_on)
    if index_columns is None:
        index_columns = [c for c in reference.columns if c not in value_columns]

    frames = [run(sampled) for sampled in ensemble]

    if align_on is not None:
        grid = reference[align_on].to_numpy()
        stacks = {
            col: np.vstack(
                [np.interp(grid, f[align_on].to_numpy(), f[col].to_numpy()) for f in frames]
            )
            for col in value_columns
        }
    else:
        n_rows = len(reference)
        if any(len(f) != n_rows for f in frames):
            raise ValueError(
                "per-sample outputs differ in length; pass align_on=<x column> "
                "to interpolate them onto a common grid"
            )
        stacks = {
            col: np.vstack([f[col].to_numpy() for f in frames]) for col in value_columns
        }

    out = reference[index_columns].reset_index(drop=True).copy()
    for col in value_columns:
        for q in quantiles:
            out[f"{col}_p{int(round(q * 100)):02d}"] = np.quantile(
                stacks[col], q, axis=0
            )
    return out
