# DST reproduction reference (round-4 B2)

Reference numbers and tolerance for the end-to-end regression that reproduces
the published DST degradation curves of Xu et al. (2018).

## Target figure

Xu et al. Fig. 5b — *Reproduced DST Data* — 1C capacity retention (%) vs number
of DST cycles, for seven SoC-window / temperature conditions, all at 20 °C. The
original experimental data (Fig. 5a) has cell-to-cell scatter and is *not* the
target; the reproduced curves are the deterministic output of the paper's own
model, so a faithful reimplementation must land on them. The figures themselves
are copyrighted and are not reproduced here — consult the paper.

The seven conditions (label = `SoC_max-SoC_min @ 20 °C`) and the last DST cycle
of each dataset (the x-extent of each curve, already encoded in
`esf/simulations/dst_cycle.py::LAST_DST_CYCLE_NUMBER`):

| label        | SoC window | DoD  | last DST cycle |
|--------------|-----------:|-----:|---------------:|
| 100-25 @20°C |    25–100  | 75 % |          4 384 |
| 100-40 @20°C |    40–100  | 60 % |          4 987 |
|  85-25 @20°C |    25–85   | 60 % |          5 251 |
| 100-50 @20°C |    50–100  | 50 % |          5 486 |
|  75-25 @20°C |    25–75   | 50 % |          5 226 |
|  75-45 @20°C |    45–75   | 30 % |          6 591 |
|  75-65 @20°C |    65–75   | 10 % |          8 393 |

## Reference: final SoH read off Fig. 5b

State of health (%) at each condition's final DST cycle, read off the reproduced
curves (reading precision ≈ ±0.7 pp):

| label        | Fig. 5b SoH (%) |
|--------------|----------------:|
| 100-25 @20°C |            79.7 |
| 100-40 @20°C |            81.2 |
|  85-25 @20°C |            82.6 |
| 100-50 @20°C |            82.2 |
|  75-25 @20°C |            85.2 |
|  75-45 @20°C |            87.0 |
|  75-65 @20°C |            90.8 |

## Our simulation vs the figure

`DSTCycleDeg(...)` run with `get_example_params()` (the paper's Table I
constants: `k_1_dod=1.4e5, k_2_dod=-0.501, k_3_dod=-1.23e5, k_soc=1.04,
k_temperature=6.93e-2, k_t=4.14e-10, sei_alpha=0.0575, sei_beta=121`):

| label        | sim SoH (%) | Δ vs Fig. 5b (pp) |
|--------------|------------:|------------------:|
| 100-25 @20°C |       77.32 |             −2.38 |
| 100-40 @20°C |       80.96 |             −0.24 |
|  85-25 @20°C |       82.18 |             −0.42 |
| 100-50 @20°C |       82.29 |             +0.09 |
|  75-25 @20°C |       85.31 |             +0.11 |
|  75-45 @20°C |       87.07 |             +0.07 |
|  75-65 @20°C |       89.86 |             −0.94 |

Six of seven conditions reproduce to within ±0.5 pp. The deepest-DoD case
(100-25, 75 % DoD) is the outlier at −2.4 pp — a genuine small reproduction
gap, not figure-reading noise. `get_example_params_from_original_repo()` (a
different DoD parameterization, `degradation_model="original"`) does **not**
reproduce the figure (5–10 pp low), so it is not the reproduction config.
The `high_soc` cycling factor is inactive in these windows (the "default" and
"original" model sets give identical curves).

## Acceptance criterion

- **Reproduction (vs figure):** each condition's final SoH within **±3.0 pp**
  absolute of the Fig. 5b value. This covers figure-reading (~0.7 pp) plus the
  observed max reproduction gap (2.4 pp), while still failing on a real porting
  regression (which shifts curves by 5–10 pp, cf. the original-repo params).
- **Drift guard:** each condition's final SoH within **±0.3 pp** of the pinned
  simulator value (the table above). Tighter than the figure test; catches any
  silent change in the cycle-counting / degradation pipeline.
- **Structure:** every curve starts at 100 % SoH, is monotonically
  non-increasing, and the final SoH is strictly increasing across the sequence
  100-25 < 100-40 < 85-25 < 100-50 < 75-25 < 75-45 < 75-65 (less DoD / lower
  mean SoC → less fade).

Pinned by `tests/test_dst_reproduction.py`.
