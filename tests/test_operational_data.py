"""Tests for OperationalData (round-4 A5).

OperationalData ingests a raw field-data trace (SoC / C-rate / temperature /
SoH over time), converts it to the package's internal units, and hands off the
drive-cycle frame the degradation predictor consumes.
"""

import numpy as np
import pandas as pd
import pytest

import esf
from esf.io.data import Data, OperationalData
from esf.simulations.cycles import drive_cycle_001


def _field_frame(datetime=True):
    """A raw field-data frame built from the internal drive cycle 001.

    drive_cycle_001 is already in internal units, so converting it back to raw
    field units (percent SoC, degC, wall-clock timestamps) lets the tests check
    that ingestion recovers the original internal-unit values exactly.
    """
    dc = drive_cycle_001(verbose=False)
    frame = pd.DataFrame(
        {
            "SOC (%)": dc["soc"].to_numpy() * 100.0,
            "Crate (C)": dc["c-rate"].to_numpy(),
            "Temp (DegC)": dc["temperature"].to_numpy() - 273.15,
            "SOH (%)": 100.0,
        }
    )
    if datetime:
        base = pd.Timestamp("2020-10-08 15:29:47")
        frame["DateTime"] = base + pd.to_timedelta(dc["time"].to_numpy(), unit="s")
    else:
        frame["t"] = dc["time"].to_numpy()
    return dc, frame


COLUMN_MAP = {
    "datetime": "DateTime",
    "soc": "SOC (%)",
    "c_rate": "Crate (C)",
    "temperature": "Temp (DegC)",
    "soh": "SOH (%)",
}


class TestIngestionUnits:
    def test_datetime_trace_round_trips_to_internal_units(self):
        dc, frame = _field_frame(datetime=True)
        op = OperationalData.from_field_dataframe(frame, column_map=COLUMN_MAP)
        out = op.to_drive_cycle()
        assert list(out.columns) == ["time", "soc", "temperature", "c-rate"]
        assert np.allclose(out["time"], dc["time"])  # elapsed seconds
        assert np.allclose(out["soc"], dc["soc"])  # fraction
        assert np.allclose(out["temperature"], dc["temperature"])  # kelvin
        assert np.allclose(out["c-rate"], dc["c-rate"])

    def test_numeric_time_column_with_unit(self):
        dc, frame = _field_frame(datetime=False)
        frame["t_hours"] = frame["t"] / 3600.0
        cmap = {**COLUMN_MAP, "time": "t_hours"}
        cmap.pop("datetime")
        op = OperationalData.from_field_dataframe(
            frame, column_map=cmap, time_unit="hours"
        )
        assert np.allclose(op.to_drive_cycle()["time"], dc["time"])

    def test_kelvin_and_fraction_inputs_are_passed_through(self):
        dc, frame = _field_frame(datetime=True)
        frame["SoC_frac"] = dc["soc"].to_numpy()
        frame["T_K"] = dc["temperature"].to_numpy()
        cmap = {**COLUMN_MAP, "soc": "SoC_frac", "temperature": "T_K"}
        op = OperationalData.from_field_dataframe(
            frame, column_map=cmap, soc_in_percent=False, temperature_unit="K"
        )
        out = op.to_drive_cycle()
        assert np.allclose(out["soc"], dc["soc"])
        assert np.allclose(out["temperature"], dc["temperature"])


class TestDriveCycleAndValidation:
    def test_missing_required_column_raises(self):
        _, frame = _field_frame(datetime=True)
        cmap = {k: v for k, v in COLUMN_MAP.items() if k != "c_rate"}
        with pytest.raises(ValueError, match="required column"):
            OperationalData.from_field_dataframe(
                frame.drop(columns=["Crate (C)"]), column_map=cmap
            )

    def test_no_time_axis_raises(self):
        _, frame = _field_frame(datetime=True)
        cmap = {k: v for k, v in COLUMN_MAP.items() if k != "datetime"}
        # no datetime mapping and no default 't' column present
        with pytest.raises(ValueError, match="time axis"):
            OperationalData.from_field_dataframe(
                frame.drop(columns=["DateTime"]), column_map=cmap
            )

    def test_to_drive_cycle_drops_incomplete_rows(self):
        _, frame = _field_frame(datetime=True)
        op = OperationalData.from_field_dataframe(frame, column_map=COLUMN_MAP)
        op.data.loc[5, "temperature"] = np.nan
        out = op.to_drive_cycle()
        assert len(out) == len(op.data) - 1
        assert not out.isna().any().any()


class TestSoHAndPrediction:
    def test_measured_soh_present(self):
        _, frame = _field_frame(datetime=True)
        op = OperationalData.from_field_dataframe(frame, column_map=COLUMN_MAP)
        soh = op.measured_soh()
        assert list(soh.columns) == ["time", "SoH"]
        assert np.allclose(soh["SoH"], 1.0)  # 100 % -> fraction

    def test_measured_soh_absent_returns_none(self):
        _, frame = _field_frame(datetime=True)
        cmap = {k: v for k, v in COLUMN_MAP.items() if k != "soh"}
        op = OperationalData.from_field_dataframe(
            frame.drop(columns=["SOH (%)"]), column_map=cmap
        )
        assert op.measured_soh() is None

    def test_predict_returns_loss(self):
        _, frame = _field_frame(datetime=True)
        op = OperationalData.from_field_dataframe(frame, column_map=COLUMN_MAP)
        prms = esf.get_example_params()
        result = op.predict(prms)
        assert "loss" in result.columns
        assert float(result["loss"].iloc[-1]) > 0

    def test_predict_more_passes_more_loss(self):
        _, frame = _field_frame(datetime=True)
        op = OperationalData.from_field_dataframe(frame, column_map=COLUMN_MAP)
        prms = esf.get_example_params()
        one = op.predict(prms, cycle_numbers=np.array([1.0]))["loss"].iloc[-1]
        many = op.predict(prms, cycle_numbers=np.array([1.0, 500.0]))["loss"].iloc[-1]
        assert many > one


class TestPlumbing:
    def test_exported_from_package(self):
        assert esf.OperationalData is OperationalData

    def test_data_base_accepts_a_dataframe(self):
        # regression: Data.__init__ used `data or ...`, which raised on a
        # non-empty DataFrame (truth-value ambiguity)
        frame = pd.DataFrame({"a": [1, 2]})
        assert Data(data=frame).data.equals(frame)
