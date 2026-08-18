# Changelog

All notable changes to `esf` are recorded here.

**Versioning policy.** The project follows [Semantic Versioning](https://semver.org/)
under a `0.y.z` pre-1.0 series: while the public API is still settling, the
**minor** number (`y`) is bumped for features and any breaking change, and the
**patch** number (`z`) for backward-compatible fixes. The public API is what
`import esf` exposes (see `esf.__all__`); anything imported from a submodule is
internal and may change without a minor bump. A `1.0.0` release will be cut once
the API is declared stable.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- **Initial public release.** This repository is a cleaned snapshot of the
  internal development repository, published under the MIT license. It carries
  fresh history: the internal repository keeps the full development history and
  remains the source of truth. Snapshot taken from internal commit
  `d9fa6a691b6a231f679526f5fa79a304b653a30f`.
- Rainflow counting now uses the MIT-licensed
  [`rainflow`](https://pypi.org/project/rainflow/) package instead of a vendored
  GPL-3.0 implementation, so the whole project can ship under MIT.
  `esf/external/rainflow_adapter.py` wraps it and returns a `(3, n_cycle)`
  array of `(range, mean, count)`. The counting is unchanged — both are ASTM
  E1049-85 and agree exactly; `tests/test_cycle_counting.py` pins the ASTM
  reference result, and the DST reproduction regression is unaffected.

### Removed
- The Streamlit reference apps, the frozen `model_IFE` reference implementation
  and its operating data, the exploratory notebooks, and the third-party
  reference PDFs are not part of the public repository.

## [0.1.0]

First versioned release after the Round 1–3 cleanup and the Round 4 capability
work. Highlights relative to the unversioned `0.0.4`:

### Added
- Public API (`import esf`): the two core workflows (data → fit → parameters;
  parameters + drive cycle → degradation) plus fit facades, `SampleData`,
  `DSTCycleDeg`, and uncertainty propagation, all pinned by tests.
- Uncertainty propagation (`ParameterUncertainty`, `ParameterEnsemble`,
  `simulate_with_uncertainty`): opt-in Monte-Carlo bands over the fitted
  covariance; the simulators stay float-only.
- DoD stress-factor fit: reference-conditions path implemented and validated
  against the published LMO constants.
- Architecture document (`docs/fitting-architecture.md`) backed by a runnable,
  test-executed example.
- Property-based tests for the stress-factor relations.

### Changed
- Temperatures are kelvin internally everywhere (units convention DD-9); this
  fixed wrong stress factors when data arrived in kelvin.
- `streamlit`/`plotly` moved out of the runtime dependencies: they are only
  needed by the reference apps, not by the library.
- Conservative lower bounds added to the runtime dependencies.
- Fits can run without mutating `ESFParams` (`apply=False`); results are read
  via `fitted_parameters()` / `parameter_uncertainty()`.
- Package structure: vendored algorithms live in `esf/external/`.

### Fixed
- Numerous correctness bugs pinned by regression tests during Rounds 1–3
  (JSON round-trips, integer-input stress factors, numpy≥2 breakages, registry
  corruption across fits, and more — see `development/session-plan.md`).
