"""The bundled data must be reachable from an *installed* package, and a
temperature in the wrong unit must not fail silently.

Both of these were real defects. `data/` used to sit outside the package and be
resolved as ``esf/settings/../../data``, which happens to work in a source
checkout and raises ``FileNotFoundError`` from a wheel — so the test suite,
which always runs in a checkout, could never see it. These tests assert the
property that actually matters (the path is *inside* the package) rather than
merely that the files exist.
"""

import warnings

import pytest

from esf.io.data import example_sample_data
from esf.settings.parameters import (
    DATA_FOLDER,
    DST_DATA_FOLDER,
    PACKAGE_FOLDER,
    PRMS_FOLDER,
    check_temperature_unit,
    get_example_params,
)
from esf.simulations.dst_cycle import dst_cycles_from_experimental_data


class TestDataShipsWithThePackage:
    def test_data_folder_is_inside_the_package(self):
        assert DATA_FOLDER.is_relative_to(PACKAGE_FOLDER), (
            f"{DATA_FOLDER} is outside {PACKAGE_FOLDER}; it will not ship in "
            "the wheel and will not resolve for an installed consumer"
        )

    @pytest.mark.parametrize("folder", [DATA_FOLDER, DST_DATA_FOLDER, PRMS_FOLDER])
    def test_folders_exist(self, folder):
        assert folder.exists(), folder

    def test_dst_cycle_module_agrees_with_settings(self):
        """The two modules used to derive this path independently."""
        from esf.simulations import dst_cycle

        assert dst_cycle.DST_DATA_FOLDER == DST_DATA_FOLDER
        assert dst_cycle.PRMS_FOLDER == PRMS_FOLDER

    def test_experimental_dst_data_loads(self):
        frame = dst_cycles_from_experimental_data(DST_DATA_FOLDER)
        assert frame["label"].nunique() == 7
        assert not frame.empty

    def test_example_sample_data_loads(self):
        """Also reads DATA_FOLDER, and was broken for installed consumers."""
        data = example_sample_data()
        assert data is not None


class TestTemperatureUnitGuard:
    def test_warns_on_celsius_into_kelvin_parameters(self):
        with pytest.warns(UserWarning, match="likely degrees celsius"):
            check_temperature_unit(20.0, "K")

    def test_warns_on_kelvin_into_celsius_parameters(self):
        with pytest.warns(UserWarning, match="likely kelvin"):
            check_temperature_unit(293.15, "degC")

    @pytest.mark.parametrize(
        ("temperature", "unit"),
        [(293.15, "K"), (298.15, "kelvin"), (20.0, "degC"), (-10.0, "degC")],
    )
    def test_silent_when_the_units_are_consistent(self, temperature, unit):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_temperature_unit(temperature, unit)

    @pytest.mark.parametrize("value", [None, float("nan"), "not-a-number"])
    def test_tolerates_values_it_cannot_judge(self, value):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_temperature_unit(value, "K")

    def test_unknown_unit_is_not_second_guessed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_temperature_unit(20.0, "rankine")

    def test_dst_simulation_warns_and_returns_a_flat_curve(self):
        """The end-to-end symptom this guard exists to announce."""
        from esf.simulations.dst_cycle import DSTCycleDeg

        prms = get_example_params()
        assert prms.temperature_unit == "K"
        with pytest.warns(UserWarning, match="likely degrees celsius"):
            curve = DSTCycleDeg(25, 100, prms=prms, temperature=20.0)
        assert curve.soh[-1] == pytest.approx(100.0, abs=0.01)

    def test_correct_kelvin_is_silent_and_degrades(self):
        from esf.simulations.dst_cycle import DSTCycleDeg

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            curve = DSTCycleDeg(
                25, 100, prms=get_example_params(), temperature=293.15
            )
        assert curve.soh[-1] == pytest.approx(77.32, abs=0.05)
