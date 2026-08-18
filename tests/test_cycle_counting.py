"""Tests for the rainflow cycle counting wrapper (esf.models.cycle_counting_algorithm).

The profiles are synthetic, so the expected rainflow results are known exactly:
- a triangle wave decomposes into half cycles of the full amplitude,
- a small ripple nested in a large swing becomes one full small cycle,
- a constant profile takes the no-cycles fallback branch.
"""

import numpy as np
import pytest

from esf.external import rainflow_adapter
from esf.models.cycle_counting_algorithm import CycleCounter


def make_counter(soc, dt=60.0, temperature=25.0, c_rate=-1.0, delta=0.01):
    soc = np.asarray(soc, dtype=np.float64)
    t = np.arange(len(soc)) * dt
    counter = CycleCounter(
        time_v=t,
        soc_v=soc,
        temp_v=np.full_like(soc, temperature),
        crate_v=np.full_like(soc, c_rate),
        delta=delta,
    )
    counter.rainflow_process()
    return counter


@pytest.fixture
def triangle_counter():
    # 0.9 -> 0.3 -> 0.9 -> 0.3 -> 0.9 (two full oscillations, 60 s steps)
    n = 60
    seg = np.linspace(0.9, 0.3, n)
    soc = np.concatenate([seg, seg[::-1][1:], seg[1:], seg[::-1][1:]])
    return make_counter(soc)


class TestTriangleWave:
    def test_four_half_cycles(self, triangle_counter):
        np.testing.assert_allclose(triangle_counter.arr_n, [0.5, 0.5, 0.5, 0.5])
        assert np.sum(triangle_counter.arr_n) == pytest.approx(2.0)

    def test_dod_is_full_amplitude(self, triangle_counter):
        np.testing.assert_allclose(triangle_counter.arr_dod, 0.6, atol=1e-12)

    def test_cycle_mean_soc(self, triangle_counter):
        np.testing.assert_allclose(triangle_counter.arr_soc_mean, 0.6, atol=1e-12)

    def test_cycle_time_spans_one_leg(self, triangle_counter):
        # each half cycle spans one leg of the triangle: 59 steps of 60 s
        np.testing.assert_allclose(triangle_counter.arr_time, 59 * 60.0)

    def test_temperature_and_crate_are_averaged(self, triangle_counter):
        np.testing.assert_allclose(triangle_counter.arr_temp, 25.0)
        np.testing.assert_allclose(triangle_counter.arr_crate, -1.0)

    def test_mean_soc_close_to_profile_mean(self, triangle_counter):
        assert triangle_counter.mean_soc == pytest.approx(0.6, abs=0.01)


class TestNestedRipple:
    def test_ripple_extracted_as_full_cycle(self):
        # large swing 0.95 -> 0.25 -> 0.95 with a 0.1 ripple on the way down
        soc = np.concatenate(
            [
                np.linspace(0.95, 0.60, 20),
                np.linspace(0.60, 0.70, 8),
                np.linspace(0.70, 0.25, 25),
                np.linspace(0.25, 0.95, 30),
            ]
        )
        counter = make_counter(soc, dt=30.0)

        order = np.argsort(counter.arr_dod)
        dod = counter.arr_dod[order]
        n = counter.arr_n[order]
        soc_mean = counter.arr_soc_mean[order]

        # one full small cycle (the ripple) ...
        np.testing.assert_allclose(dod[0], 0.1, atol=1e-12)
        assert n[0] == pytest.approx(1.0)
        assert soc_mean[0] == pytest.approx(0.65, abs=1e-12)
        # ... and the outer swing as two half cycles
        np.testing.assert_allclose(dod[1:], 0.7, atol=1e-12)
        np.testing.assert_allclose(n[1:], 0.5)
        np.testing.assert_allclose(soc_mean[1:], 0.6, atol=1e-12)


class TestConstantProfile:
    def test_fallback_when_no_turning_points(self):
        n_points = 50
        counter = make_counter(
            np.full(n_points, 0.42), dt=10.0, temperature=30.0, c_rate=0.0
        )

        assert list(counter.arr_dod) == [0.0]
        assert list(counter.arr_n) == [1]
        assert counter.arr_soc_mean[0] == pytest.approx(0.42)
        assert counter.arr_temp[0] == pytest.approx(30.0)
        # spans the full profile duration
        assert counter.arr_time[0] == pytest.approx((n_points - 1) * 10.0)


class TestInputValidation:
    def test_requires_time_and_soc(self):
        with pytest.raises(AssertionError):
            CycleCounter(time_v=None, soc_v=None)


class TestRainflowAdapter:
    """Pins the counting itself, independent of the CycleCounter plumbing.

    The counter delegates to the MIT-licensed ``rainflow`` package through
    ``esf.external.rainflow_adapter``. These cases lock the contract that
    adapter promises: the ASTM E1049-85 result, the column layout, and the
    short-sequence edge cases.
    """

    def test_rainflow_matches_astm_reference(self):
        # The worked example from ASTM E1049-85 (sec. 5.4.5, fig. 6): the
        # turning-point sequence and the cycles it decomposes into.
        turning_points = [-2, 1, -3, 5, -1, 3, -4, 4, -2]

        out = rainflow_adapter.rainflow(turning_points)

        expected = [
            # (range, mean, count)
            (3.0, -0.5, 0.5),
            (4.0, -1.0, 0.5),
            (4.0, 1.0, 1.0),
            (8.0, 1.0, 0.5),
            (9.0, 0.5, 0.5),
            (8.0, 0.0, 0.5),
            (6.0, 1.0, 0.5),
        ]
        assert out.shape == (3, len(expected))
        np.testing.assert_allclose(out.T, expected)
        # half and full cycles together account for the whole sequence
        assert out[rainflow_adapter.ROW_COUNT].sum() == pytest.approx(4.0)

    def test_two_turning_points_are_one_half_cycle(self):
        out = rainflow_adapter.rainflow([0.9, 0.3])

        np.testing.assert_allclose(out.T, [(0.6, 0.6, 0.5)])

    @pytest.mark.parametrize(
        "turning_points", [[], [0.5], [0.5, 0.5]], ids=["empty", "single", "flat"]
    )
    def test_degenerate_sequences_yield_no_cycles(self, turning_points):
        out = rainflow_adapter.rainflow(turning_points)

        assert out.shape == (3, 0)
