from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from esf.simulations.cycles import (
    _calculate_soc,
    _convert_from_power_to_c_rate,
    _convert_time,
    _convert_unit,
    _time_integrated_power,
    _time_power_cumsum,
    add_step,
    convert_power_cycle_to_rate_cycle,
    drive_cycle_001,
    drive_cycle_002,
    load_drive_cycle,
    plot_drive_cycle,
    resample_drive_cycle,
    sim_cycles,
)


def test_calculate_soc():
    """Test the _calculate_soc function."""
    step = {"t": 1.0, "r": 0.5}
    soc_0 = 0.8
    delta_time = 0.1

    c_rate, delta_soc, t, soc = _calculate_soc(step, soc_0, delta_time)

    assert c_rate == 0.5
    assert delta_soc == pytest.approx(0.5 * 0.1)
    assert len(t) == 11  # 0.0 to 1.0 with step 0.1 = 11 points
    assert t[0] == 0.0
    assert t[-1] == 1.0
    assert soc[0] == 0.8
    assert soc[-1] == pytest.approx(0.8 + 1.0 * 0.5)


def test_add_step():
    """Test the add_step function."""
    step = {"t": 1.0, "r": 0.5, "l": "charge"}
    step_number = 1
    cycle_number = 1
    soc_0 = 0.8
    t_0 = 0.0
    temperature = 298.15
    delta_time = 0.1

    df = add_step(step, step_number, cycle_number, soc_0, t_0, temperature, delta_time)

    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns
    assert "cycle" in df.columns
    assert "step" in df.columns
    assert "soc" in df.columns
    assert "c-rate" in df.columns
    assert "temperature" in df.columns

    assert df["cycle"].iloc[0] == 1
    assert df["step"].iloc[0] == 1
    assert df["soc"].iloc[0] == 0.8
    assert df["c-rate"].iloc[0] == 0.5
    assert df["temperature"].iloc[0] == 298.15


def test_sim_cycles():
    """Test the sim_cycles function."""
    drive_cycle = [
        {"t": 0.5, "r": -0.5, "l": "discharge"},
        {"t": 0.5, "r": 0.5, "l": "charge"},
    ]
    number_of_cycles = 2
    delta_time = 0.1

    cycles, df = sim_cycles(
        drive_cycle,
        number_of_cycles,
        delta_time,
        soc_cut_off=0.2,
        t_cut_off=10.0,
        temperature=298.15,
        t_start=0.0,
        soc_start=1.0,
    )

    assert len(cycles) == 2
    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns
    assert "cycle" in df.columns
    assert "step" in df.columns
    assert "soc" in df.columns
    assert "c-rate" in df.columns

    # Check that we have data for both cycles
    assert df["cycle"].nunique() == 2

    # Check that SOC stays within bounds
    assert df["soc"].min() >= 0.2
    assert df["soc"].max() <= 1.0


def test_drive_cycle_001():
    """Test the drive_cycle_001 function."""
    df = drive_cycle_001(verbose=False, temperature=298.15)

    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns
    assert "cycle" in df.columns
    assert "step" in df.columns
    assert "soc" in df.columns
    assert "c-rate" in df.columns
    assert "temperature" in df.columns

    # Check that the drive cycle has the expected structure
    assert df["step"].nunique() > 1  # Multiple steps
    assert df["cycle"].nunique() == 1  # One cycle


def test_drive_cycle_002():
    """Test the drive_cycle_002 function."""
    df = drive_cycle_002(verbose=False, temperature=298.15)

    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns
    assert "cycle" in df.columns
    assert "step" in df.columns
    assert "soc" in df.columns
    assert "c-rate" in df.columns
    assert "temperature" in df.columns

    # Check that the drive cycle has the expected structure
    assert df["step"].nunique() > 1  # Multiple steps
    assert df["cycle"].nunique() == 1  # One cycle


def test_time_integrated_power():
    """Test the _time_integrated_power function."""
    # Create a test dataframe
    df = pd.DataFrame({"time": [0, 1, 2, 3, 4], "power": [0, 10, 20, -10, -20]})

    # Test positive power integration
    positive_power = _time_integrated_power(df, direction="positive")
    assert positive_power == pytest.approx(15.0)  # Area under the positive power curve

    # Test negative power integration
    negative_power = _time_integrated_power(df, direction="negative")
    assert negative_power == pytest.approx(
        15.0
    )  # Area under the negative power curve (absolute value)

    # Test with invalid direction
    with pytest.raises(ValueError):
        _time_integrated_power(df, direction="invalid")


def test_time_power_cumsum():
    """Test the _time_power_cumsum function."""
    # Create a test dataframe
    df = pd.DataFrame({"time": [0, 1, 2, 3, 4], "power": [0, 10, 20, -10, -20]})

    # Test with regeneration
    cumsum_with_regen = _time_power_cumsum(df, regenerative=True)
    assert len(cumsum_with_regen) == 5
    assert cumsum_with_regen.iloc[-1] == pytest.approx(0)  # Net zero with regeneration

    # Test without regeneration
    cumsum_without_regen = _time_power_cumsum(df, regenerative=False)
    assert len(cumsum_without_regen) == 5
    assert cumsum_without_regen.iloc[-1] == pytest.approx(
        -30.0
    )  # Based on actual implementation


def test_convert_from_power_to_c_rate():
    """Test the _convert_from_power_to_c_rate function."""
    # Test with scalar
    power = 100  # watts
    battery_capacity = 50  # watt-hours
    c_rate = _convert_from_power_to_c_rate(power, battery_capacity)
    assert c_rate == pytest.approx(2.0)  # 100W / 50Wh = 2C

    # Test with Series
    power_series = pd.Series([50, 100, 150])
    c_rate_series = _convert_from_power_to_c_rate(power_series, battery_capacity)
    assert isinstance(c_rate_series, pd.Series)
    assert c_rate_series.iloc[0] == pytest.approx(1.0)
    assert c_rate_series.iloc[1] == pytest.approx(2.0)
    assert c_rate_series.iloc[2] == pytest.approx(3.0)


def test_convert_time():
    """Test the _convert_time function."""
    # Test with scalar
    time = 1.0  # hours
    converted_time = _convert_time(time, "hours", "minutes")
    assert converted_time == pytest.approx(60.0)

    # Test with Series
    time_series = pd.Series([1.0, 2.0, 3.0])  # hours
    converted_time_series = _convert_time(time_series, "hours", "minutes")
    assert isinstance(converted_time_series, pd.Series)
    assert converted_time_series.iloc[0] == pytest.approx(60.0)
    assert converted_time_series.iloc[1] == pytest.approx(120.0)
    assert converted_time_series.iloc[2] == pytest.approx(180.0)


def test_convert_unit():
    """Test the _convert_unit function."""
    # Test with Series
    value_series = pd.Series([1.0, 2.0, 3.0])  # hours
    converted_value_series = _convert_unit(value_series, "hours", "minutes")
    assert isinstance(converted_value_series, pd.Series)
    assert converted_value_series.iloc[0] == pytest.approx(60.0)
    assert converted_value_series.iloc[1] == pytest.approx(120.0)
    assert converted_value_series.iloc[2] == pytest.approx(180.0)

    # Test with scalar (regression: the scalar branch used to reference the
    # time module instead of the value argument and raised a TypeError)
    assert _convert_unit(2.0, "hours", "minutes") == pytest.approx(120.0)


def test_convert_power_cycle_to_rate_cycle():
    """Test the convert_power_cycle_to_rate_cycle function."""
    # Create a test dataframe with delta_time column
    df = pd.DataFrame({"time": [0, 1, 2, 3, 4], "power": [0, 100, 200, -100, -200]})

    # Add delta_time column to avoid NaN in the first row
    df["delta_time"] = df["time"].diff().fillna(0)

    result_df = convert_power_cycle_to_rate_cycle(
        df,
        initial_soc=1.0,
        temperature=25,
        battery_oversize_factor=1.0,
        use_regeneration=True,
        regeneration_factor=1.0,
        verbose=False,
    )

    assert isinstance(result_df, pd.DataFrame)
    assert "time" in result_df.columns
    assert "soc" in result_df.columns
    assert "c-rate" in result_df.columns
    assert "temperature" in result_df.columns

    # Check temperature
    assert result_df["temperature"].iloc[0] == 25


def test_resample_drive_cycle():
    """Test the resample_drive_cycle function."""
    # Create a test dataframe with 1-second intervals
    df = pd.DataFrame(
        {
            "time": np.arange(0, 3600, 1) / 3600,  # Convert to hours
            "soc": np.linspace(1.0, 0.5, 3600),
            "c-rate": np.ones(3600) * -0.5,
            "temperature": np.ones(3600) * 25,
        }
    )

    # Resample to 1-minute intervals
    resampled_df = resample_drive_cycle(df, time_unit="1min")

    assert isinstance(resampled_df, pd.DataFrame)
    assert "time" in resampled_df.columns
    assert "soc" in resampled_df.columns
    assert "c-rate" in resampled_df.columns
    assert "temperature" in resampled_df.columns

    # Check that we have approximately 60 rows (1 hour at 1-minute intervals)
    assert 55 <= len(resampled_df) <= 65


@pytest.mark.parametrize(
    "path_type",
    [
        "single_path",
        "list_of_paths",
    ],
)
def test_load_drive_cycle(path_type, tmp_path):
    """Test the load_drive_cycle function."""
    # Create a test CSV file
    csv_content = "Sec;Watts\n0;0\n1;100\n2;200\n3;-100\n4;-200"

    file_path1 = tmp_path / "test_drive_cycle1.csv"
    file_path1.write_text(csv_content)

    file_path2 = tmp_path / "test_drive_cycle2.csv"
    file_path2.write_text(csv_content)

    if path_type == "single_path":
        path = file_path1
    else:
        path = [file_path1, file_path2]

    df = load_drive_cycle(
        path,
        seperator=";",
        index_col=None,
        power_col="Watts",
        time_col="Sec",
        time_unit="seconds",
        power_unit="W",
    )

    assert isinstance(df, pd.DataFrame)
    assert "time" in df.columns
    assert "power" in df.columns

    if path_type == "single_path":
        assert len(df) == 5
    else:
        assert len(df) == 10  # Two files with 5 rows each


@patch("matplotlib.pyplot.subplots")
def test_plot_drive_cycle(mock_subplots):
    """Test the plot_drive_cycle function."""
    # Create a mock figure and axes
    mock_fig = MagicMock()
    mock_axes = [MagicMock(), MagicMock()]
    mock_subplots.return_value = (mock_fig, mock_axes)

    # Create a test dataframe
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "soc": [1.0, 0.9, 0.8, 0.7, 0.6],
            "c-rate": [-0.1, -0.1, -0.1, -0.1, -0.1],
        }
    )

    # Call the function
    plot_drive_cycle(df, soc_cut_off=0.2)

    # Check that the function called the expected matplotlib functions
    mock_subplots.assert_called_once()

    # We can't easily check the plot content without complex mocking,
    # but we can verify the function runs without errors
