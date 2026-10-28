"""Power-versus-time traces become the drive-cycle frame.

The numbers below are trapezoidal integrals worked out by hand, so a change
in the sign, the time base, or the energy rule fails the test rather than
shifting a snapshot.
"""

import numpy as np
import pandas as pd
import pytest

from esf.simulations.degradation import unpack_drive_cycle_frame
from esf.simulations.power_profile import (
    PowerProfileError,
    drive_cycle_from_power,
    drive_cycle_from_power_csv,
)

# 0, 1800, 5400 s; discharge-positive power 0, 20, 0 W.
# Energy into the pack: 0, -5, -15 Wh. On a 30 Wh pack from SoC 0.8:
# SoC 0.8, 0.6333..., 0.3 and C-rate 0, -2/3, 0.
_TIME_S = [0.0, 1800.0, 5400.0]
_POWER_W = [0.0, 20.0, 0.0]


def _cycle(**overrides):
    kwargs = dict(
        capacity_wh=30.0,
        initial_soc=0.8,
        temperature=25.0,
        positive="discharge",
    )
    kwargs.update(overrides)
    return drive_cycle_from_power(_TIME_S, _POWER_W, **kwargs)


class TestHandComputedTrace:
    def test_frame_matches_the_trapezoid(self):
        frame = _cycle()

        assert list(frame.columns) == ["time", "soc", "c-rate", "temperature"]
        assert frame["time"].tolist() == pytest.approx(_TIME_S)
        assert frame["soc"].tolist() == pytest.approx([0.8, 0.8 - 5.0 / 30.0, 0.3])
        assert frame["c-rate"].tolist() == pytest.approx([0.0, -20.0 / 30.0, 0.0])
        assert frame["temperature"].tolist() == pytest.approx([298.15, 298.15, 298.15])

    def test_frame_unpacks_for_the_degradation_calculator(self):
        time, soc, temperature, c_rate = unpack_drive_cycle_frame(_cycle())

        assert time.tolist() == pytest.approx(_TIME_S)
        assert soc[0] == pytest.approx(0.8)
        assert temperature[0] == pytest.approx(298.15)
        assert c_rate[1] == pytest.approx(-20.0 / 30.0)

    def test_charge_positive_is_the_opposite_sign(self):
        discharged = _cycle()
        charged = drive_cycle_from_power(
            _TIME_S,
            [0.0, -20.0, 0.0],
            capacity_wh=30.0,
            initial_soc=0.8,
            temperature=25.0,
            positive="charge",
        )

        assert charged["soc"].tolist() == pytest.approx(discharged["soc"].tolist())
        assert charged["c-rate"].tolist() == pytest.approx(discharged["c-rate"].tolist())

    def test_hours_and_kilowatts_match_seconds_and_watts(self):
        from_hours = drive_cycle_from_power(
            [0.0, 0.5, 1.5],
            [0.0, 0.02, 0.0],
            capacity_wh=30.0,
            initial_soc=0.8,
            temperature=298.15,
            temperature_unit="K",
            time_unit="hours",
            power_unit="kW",
            positive="discharge",
        )

        assert from_hours["time"].tolist() == pytest.approx(_TIME_S)
        assert from_hours["soc"].tolist() == pytest.approx(_cycle()["soc"].tolist())
        assert from_hours["temperature"].iloc[0] == pytest.approx(298.15)

    def test_exact_window_edge_is_kept(self):
        # 10 W discharge for 1 h from SoC 0.5 on a 20 Wh pack ends at 0.
        frame = drive_cycle_from_power(
            [0.0, 3600.0],
            [10.0, 10.0],
            capacity_wh=20.0,
            initial_soc=0.5,
            temperature=25.0,
            positive="discharge",
        )

        assert frame["soc"].tolist() == pytest.approx([0.5, 0.0])
        assert frame["c-rate"].tolist() == pytest.approx([-0.5, -0.5])


class TestSocWindow:
    def test_undersized_pack_names_the_minimum_capacity(self):
        # 10 W discharge for 1 h is 10 Wh out. From SoC 0.5 that needs 20 Wh.
        with pytest.raises(PowerProfileError) as exc:
            drive_cycle_from_power(
                [0.0, 3600.0],
                [10.0, 10.0],
                capacity_wh=10.0,
                initial_soc=0.5,
                temperature=25.0,
                positive="discharge",
            )

        assert exc.value.minimum_capacity_wh == pytest.approx(20.0)
        assert "20 Wh" in str(exc.value)

    def test_charging_a_full_pack_has_no_rescuing_capacity(self):
        with pytest.raises(PowerProfileError) as exc:
            drive_cycle_from_power(
                [0.0, 3600.0],
                [10.0, 10.0],
                capacity_wh=100.0,
                initial_soc=1.0,
                temperature=25.0,
                positive="charge",
            )

        assert exc.value.minimum_capacity_wh is None
        assert "Lower the starting SoC" in str(exc.value)

    def test_discharging_an_empty_pack_has_no_rescuing_capacity(self):
        with pytest.raises(PowerProfileError) as exc:
            drive_cycle_from_power(
                [0.0, 3600.0],
                [10.0, 10.0],
                capacity_wh=100.0,
                initial_soc=0.0,
                temperature=25.0,
                positive="discharge",
            )

        assert exc.value.minimum_capacity_wh is None
        assert "Raise the starting SoC" in str(exc.value)


class TestTractionEfficiency:
    def test_regen_is_scaled_and_discharge_is_not(self):
        # 1 h of 10 W regen (negative when positive means discharge).
        # Half of it reaches the pack: 5 Wh in, SoC 0.5 -> 1.0 on 10 Wh.
        regen = drive_cycle_from_power(
            [0.0, 3600.0],
            [-10.0, -10.0],
            capacity_wh=10.0,
            initial_soc=0.5,
            temperature=25.0,
            positive="discharge",
            source="traction",
            regen_efficiency=0.5,
        )
        discharge = drive_cycle_from_power(
            [0.0, 3600.0],
            [10.0, 10.0],
            capacity_wh=20.0,
            initial_soc=1.0,
            temperature=25.0,
            positive="discharge",
            source="traction",
            regen_efficiency=0.5,
        )

        assert regen["soc"].tolist() == pytest.approx([0.5, 1.0])
        assert regen["c-rate"].tolist() == pytest.approx([0.5, 0.5])
        assert discharge["soc"].tolist() == pytest.approx([1.0, 0.5])
        assert discharge["c-rate"].tolist() == pytest.approx([-0.5, -0.5])

    def test_terminal_power_rejects_an_efficiency(self):
        with pytest.raises(PowerProfileError, match="terminal"):
            _cycle(regen_efficiency=0.5)


class TestCsv:
    def test_battery_csv_headers(self, tmp_path):
        path = tmp_path / "Battery.csv"
        path.write_text(
            "Time (s),Battery Power (W)\n0,0\n1800,20\n5400,0\n",
            encoding="utf-8",
        )

        frame = drive_cycle_from_power_csv(
            path,
            capacity_wh=30.0,
            initial_soc=0.8,
            temperature=25.0,
            positive="discharge",
        )

        assert frame["soc"].tolist() == pytest.approx(_cycle()["soc"].tolist())

    def test_missing_column_names_the_header(self, tmp_path):
        path = tmp_path / "power.csv"
        path.write_text("t,p\n0,0\n1,1\n", encoding="utf-8")

        with pytest.raises(PowerProfileError, match="Battery Power"):
            drive_cycle_from_power_csv(
                path,
                capacity_wh=10.0,
                initial_soc=0.5,
                temperature=25.0,
            )


class TestRejectedInput:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"capacity_wh": 0.0}, "capacity_wh"),
            ({"initial_soc": 1.2}, "initial_soc"),
            ({"temperature_unit": "watt"}, "temperature"),
            ({"time_unit": "meter"}, "time"),
            ({"positive": "sink"}, "positive"),
            ({"source": "bus"}, "source"),
        ],
    )
    def test_bad_arguments(self, kwargs, match):
        with pytest.raises(PowerProfileError, match=match):
            _cycle(**kwargs)

    def test_time_must_increase(self):
        with pytest.raises(PowerProfileError, match="Time must increase"):
            drive_cycle_from_power(
                [0.0, 10.0, 10.0],
                [0.0, 1.0, 0.0],
                capacity_wh=10.0,
                initial_soc=0.5,
                temperature=25.0,
            )

    def test_one_sample_is_rejected(self):
        with pytest.raises(PowerProfileError, match="at least two"):
            drive_cycle_from_power(
                [0.0],
                [0.0],
                capacity_wh=10.0,
                initial_soc=0.5,
                temperature=25.0,
            )

    def test_non_numeric_power(self):
        with pytest.raises(PowerProfileError, match="power"):
            drive_cycle_from_power(
                [0.0, 1.0],
                [0.0, "abc"],
                capacity_wh=10.0,
                initial_soc=0.5,
                temperature=25.0,
            )


def test_public_names():
    import esf

    assert esf.drive_cycle_from_power is drive_cycle_from_power
    assert esf.drive_cycle_from_power_csv is drive_cycle_from_power_csv
    assert "drive_cycle_from_power" in esf.__all__
    assert "drive_cycle_from_power_csv" in esf.__all__


def test_rest_holds_soc():
    frame = drive_cycle_from_power(
        [0.0, 10.0, 20.0],
        [0.0, 0.0, 0.0],
        capacity_wh=50.0,
        initial_soc=0.4,
        temperature=20.0,
    )
    assert frame["soc"].tolist() == pytest.approx([0.4, 0.4, 0.4])
    assert frame["c-rate"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert isinstance(frame, pd.DataFrame)
    assert np.all(np.isfinite(frame.to_numpy(dtype=float)))
