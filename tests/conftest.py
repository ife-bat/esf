import matplotlib

# Use a non-interactive backend so tests can run headless (CI, servers).
matplotlib.use("Agg")

import pytest


# uses pytest-datadir (shared_datadir is the folder named "data" in the tests folder).
@pytest.fixture
def calendar_data_vs_temperature_df(shared_datadir):
    import pandas as pd

    data_path = shared_datadir / "calend_deg_at_50_SoC.csv"
    sep = ","
    assert data_path.is_file()

    return pd.read_csv(data_path, sep=sep)


@pytest.fixture
def calendar_data_vs_soh_df(shared_datadir):
    import pandas as pd

    data_path = shared_datadir / "calend_deg_at_25_deg.csv"
    sep = ","
    assert data_path.is_file()

    return pd.read_csv(data_path, sep=sep)


@pytest.fixture
def calendar_data_vs_temperature(calendar_data_vs_temperature_df):
    import pandas as pd

    from esf.io.data import extract_values_from_column_name

    column_names = list(calendar_data_vs_temperature_df.columns)

    n = [
        extract_values_from_column_name(
            name, "=", convert_to=int, verbose=False, add_norm=273.15
        )
        for name in column_names
    ]
    n[0] = "t"
    calendar_data_vs_temperature_df.columns = n

    data = pd.melt(
        calendar_data_vs_temperature_df, id_vars=n[0], value_name="SoH", var_name="T"
    )

    return data


@pytest.fixture
def calendar_data_vs_soh(calendar_data_vs_soh_df):
    import pandas as pd

    from esf.io.data import extract_values_from_column_name, percentage_to_float

    column_names = list(calendar_data_vs_soh_df.columns)

    n = [
        extract_values_from_column_name(
            name,
            "=",
            convert_to=percentage_to_float,
            verbose=True,
        )
        for name in column_names
    ]

    n[0] = "t"
    calendar_data_vs_soh_df.columns = n

    data = pd.melt(
        calendar_data_vs_soh_df, id_vars=n[0], value_name="T", var_name="SoH"
    )

    return data
