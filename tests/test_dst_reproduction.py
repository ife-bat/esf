"""B2: reproduce the published DST degradation curves (Xu et al. 2018, Fig. 5b).

Running ``DSTCycleDeg`` with the paper's Table I parameters
(``get_example_params``) reproduces the "Reproduced DST Data" figure. This test
pins that reproduction against the figure (reference values and tolerance in
``development/dst-reproduction-reference.md``) and, more tightly, guards the
simulator against silent drift.
"""

import numpy as np
import pytest

from esf.settings.parameters import get_example_params
from esf.simulations.dst_cycle import DSTCycleDeg

TEMPERATURE_K = 293.15  # 20 degC, the condition of all seven curves in Fig. 5

# Final SoH (%) read off Xu et al. Fig. 5b (reproduced curves), per condition.
FIG5B_FINAL_SOH = {
    (100, 25): 79.7,
    (100, 40): 81.2,
    (85, 25): 82.6,
    (100, 50): 82.2,
    (75, 25): 85.2,
    (75, 45): 87.0,
    (75, 65): 90.8,
}
# absolute SoH tolerance (percentage points) for the figure reproduction:
# figure-reading (~0.7) + observed max reproduction gap (2.4).
FIG5B_ABS_TOL = 3.0

# Current simulator output; a tight regression pin against silent drift.
SIM_FINAL_SOH = {
    (100, 25): 77.32,
    (100, 40): 80.96,
    (85, 25): 82.18,
    (100, 50): 82.29,
    (75, 25): 85.31,
    (75, 45): 87.07,
    (75, 65): 89.86,
}
SIM_ABS_TOL = 0.3

# ascending final-SoH order (deepest DoD / highest mean SoC fades most)
ORDER = [(100, 25), (100, 40), (85, 25), (100, 50), (75, 25), (75, 45), (75, 65)]


@pytest.fixture(scope="module")
def dst_curves():
    """Simulate all seven conditions once (fast, ~0.3 s total)."""
    prms = get_example_params()
    curves = {}
    for soc_max, soc_min in ORDER:
        c = DSTCycleDeg(
            soc_min,
            soc_max,
            prms=prms,
            temperature=TEMPERATURE_K,
            number_of_points_pr_test=22,
        )
        curves[(soc_max, soc_min)] = {
            "soh": np.asarray(c.soh, dtype=float),
            "cycles": np.asarray(c.cycle_v, dtype=float),
        }
    return curves


class TestReproducePublishedDSTCurves:
    """The headline B2 acceptance: match Fig. 5b within the stated tolerance."""

    @pytest.mark.parametrize("key", ORDER)
    def test_final_soh_matches_figure(self, dst_curves, key):
        final = dst_curves[key]["soh"][-1]
        assert final == pytest.approx(FIG5B_FINAL_SOH[key], abs=FIG5B_ABS_TOL)

    def test_ordering_matches_figure(self, dst_curves):
        finals = [dst_curves[key]["soh"][-1] for key in ORDER]
        assert finals == sorted(finals)  # strictly increasing along ORDER
        assert all(a < b for a, b in zip(finals, finals[1:], strict=False))


class TestDSTRegressionPin:
    """Tight drift guard on the current simulator output."""

    @pytest.mark.parametrize("key", ORDER)
    def test_final_soh_pinned(self, dst_curves, key):
        final = dst_curves[key]["soh"][-1]
        assert final == pytest.approx(SIM_FINAL_SOH[key], abs=SIM_ABS_TOL)


class TestDSTCurveShape:
    """Structural properties every reproduced curve must have."""

    @pytest.mark.parametrize("key", ORDER)
    def test_starts_at_full_health(self, dst_curves, key):
        assert dst_curves[key]["soh"][0] == pytest.approx(100.0, abs=0.05)

    @pytest.mark.parametrize("key", ORDER)
    def test_monotonically_non_increasing(self, dst_curves, key):
        soh = dst_curves[key]["soh"]
        assert np.all(np.diff(soh) <= 1e-9)

    @pytest.mark.parametrize("key", ORDER)
    def test_last_cycle_matches_dataset_extent(self, dst_curves, key):
        # the x-extent of each curve is the last DST cycle of that dataset
        from esf.simulations.dst_cycle import LAST_DST_CYCLE_NUMBER

        soc_max, soc_min = key
        expected = LAST_DST_CYCLE_NUMBER[f"{soc_min}_{soc_max}"]
        assert dst_curves[key]["cycles"][-1] == pytest.approx(expected, rel=1e-3)
