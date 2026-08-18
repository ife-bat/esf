import pytest


def test_creating_fit_data_object():
    from esf.io.data import SampleData

    fit_data = SampleData()
    assert fit_data.data.empty


def test_creating_fit_data_object_with_data(shared_datadir):
    import pandas as pd

    from esf.io.data import SampleData
    from esf.settings.parameters import DataType

    # loading and preprocessing data (cols == {t, SoH, SoC}):
    d = pd.read_csv(shared_datadir / "cyc_test_data.csv")
    d.columns = ["t", "SoH"]
    d["SoC"] = 0.5

    n_rows = d.shape[0]
    assert not d.empty
    assert n_rows > 2

    fit_data = SampleData()
    fit_data.add_data(
        d,
        comment="This is a test data set",
        time_unit="days",
        data_type=DataType.CALENDAR_VS_SOC,
    )
    assert not fit_data.data.empty
    assert fit_data.data.shape[0] == n_rows

    fit_data.calculate_life_fraction()
    d_processed = fit_data.calendar_life_vs_soc(strict_mode=False)
    assert not d_processed.empty
    assert d_processed.shape[0] == n_rows
    assert "SoH" in d_processed.columns
    assert "SoC" in d_processed.columns
    assert "t" in d_processed.columns
    assert "L" in d_processed.columns


@pytest.mark.xfail
def test_creating_fit_data_object_with_missing_data(shared_datadir):
    import pandas as pd

    from esf.io.data import SampleData
    from esf.settings.parameters import DataType

    # loading and preprocessing data (cols == {t, SoH, SoC}):
    d = pd.read_csv(shared_datadir / "cyc_test_data.csv")
    d.columns = ["t", "SoH"]
    d["SoC"] = 0.5

    n_rows = d.shape[0]
    assert not d.empty
    assert n_rows > 2

    fit_data = SampleData()
    fit_data.add_data(
        d,
        comment="This is a test data set",
        time_unit="days",
        data_type=DataType.CALENDAR_VS_SOC,
    )
    assert not fit_data.data.empty
    assert fit_data.data.shape[0] == n_rows

    fit_data.calculate_life_fraction()
    fit_data.calendar_life_vs_soc(strict_mode=True)


def test_get_dispatches_to_selectors():
    from esf.io.data import example_sample_data
    from esf.settings.parameters import DataType

    data = example_sample_data()
    data.calculate_life_fraction()
    via_get = data.get(DataType.CALENDAR_VS_TEMPERATURE, strict_mode=False)
    direct = data.calendar_life_vs_temperature(strict_mode=False)
    assert via_get.equals(direct)


def test_get_unsupported_data_type_raises():
    # regression test: unsupported combinations used to silently return None
    # (and one branch called a selector method that never existed)
    from esf.io.data import SampleData
    from esf.settings.parameters import DataType

    with pytest.raises(ValueError, match="Unsupported data type"):
        SampleData().get(DataType.OPERATIONAL_DATA)


def test_save_and_load_round_trip(tmp_path):
    # the persistence path had no callers and no tests; pin the pickle
    # round trip: data and metadata must survive
    import numpy as np
    import pandas as pd

    from esf.io.data import SampleData
    from esf.settings.parameters import DataType

    frame = pd.DataFrame(
        {"t": np.linspace(1, 100, 8), "SoH": np.linspace(1.0, 0.9, 8)}
    )
    frame["T"] = 298.15
    original = SampleData()
    original.add_data(
        frame,
        data_type=DataType.CALENDAR_VS_TEMPERATURE,
        comment="round-trip check",
        time_unit="days",
    )

    save_path = original.save(tmp_path / "sample.pkl")
    restored = SampleData.load_sample_data(save_path)

    assert restored.data.shape[0] == original.data.shape[0]
    pd.testing.assert_frame_equal(
        restored.data.reset_index(drop=True), original.data.reset_index(drop=True)
    )
    uid = original.data["uid"].iloc[0]
    assert restored.metadata.get_metadata(uid).time_unit == "days"
    assert restored.metadata.get_metadata(uid).comment == "round-trip check"


def test_plot_data_smoke():
    import matplotlib
    from matplotlib.figure import Figure

    from esf.io.data import example_sample_data

    fig = example_sample_data().plot_data(return_figure=True)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 6
    matplotlib.pyplot.close(fig)


def test_fixture(calendar_data_vs_temperature):
    import pandas as pd

    assert isinstance(calendar_data_vs_temperature, pd.DataFrame)
    assert not calendar_data_vs_temperature.empty


def test_hardcoded_soc_deg_rates():
    from esf.io.data import get_hardcoded_soc_deg_rates

    df = get_hardcoded_soc_deg_rates()

    # Check DataFrame structure
    assert not df.empty
    assert df.shape == (4, 2)
    assert list(df.columns) == ["SoC", "deg_rate"]

    # Check values
    expected_soc = [0.5, 0.6, 0.8, 1.0]
    expected_rates = [0.000036, 0.000039, 0.000049, 0.000060]

    assert df["SoC"].tolist() == expected_soc
    assert df["deg_rate"].tolist() == expected_rates


def test_hardcoded_temperature_deg_rates():
    from esf.io.data import get_hardcoded_temperature_deg_rates

    df = get_hardcoded_temperature_deg_rates()

    # Check DataFrame structure
    assert not df.empty
    assert df.shape == (5, 2)
    assert list(df.columns) == ["T", "deg_rate"]

    # Check values (temperatures in kelvin per the units convention)
    expected_temps = [288.15, 298.15, 308.15, 318.15, 328.15]
    expected_rates = [0.000017, 0.000035, 0.000069, 0.000130, 0.000236]

    assert df["T"].tolist() == expected_temps
    assert df["deg_rate"].tolist() == expected_rates
