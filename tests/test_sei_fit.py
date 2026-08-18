import pytest


@pytest.fixture
def prms():
    from esf.settings.parameters import ESFParams

    return ESFParams(
        degradation_model="default",
        battery_chemistry="LMO",
    )


@pytest.fixture
def sei_fit_object_calend(prms, calendar_data_vs_temperature):
    from esf.io.data import SampleData
    from esf.models.fitting import NonlinearFit
    from esf.settings.parameters import DataType

    fit_data = SampleData()
    fit_data.add_data(
        calendar_data_vs_temperature,
        comment="This is test data for Temperature Stress Factor fitting",
        data_type=DataType.CALENDAR_VS_TEMPERATURE,
        time_unit="years",
        temperature_unit="K",
    )
    fit_data.calculate_life_fraction()
    return NonlinearFit(
        fit_data.calendar_life_vs_temperature(
            filter_value=298.15, time_unit="days", strict_mode=False
        ),
        prms,
        regime="calend",
    )


def test_sei_fit_object_creation(sei_fit_object_calend):
    import numpy as np

    # sei_fit_object_calend.primary_fit_method = "nelder"
    sei_fit_object_calend.fit()
    df1 = sei_fit_object_calend.simulate(steps=20)
    assert not df1.empty
    assert len(df1) == 20
    df2 = sei_fit_object_calend.simulate(x=np.array([1, 2, 3]))
    assert not df2.empty
    assert len(df2) == 3
    sei_fit_object_calend.plot_results()
