def test_sei_calendar_fitting():
    from esf.io.data import example_sample_data
    from esf.models.fitting import NonlinearFit as FitClass
    from esf.settings.parameters import get_example_params

    data = example_sample_data()
    prms = get_example_params()

    data_selector = data.calendar_life_vs_temperature
    stress_factor = FitClass(
        data_selector(filter_value=298.15),
        prms,
        regime="calend_vs_temperature",
        is_reference=True,
    )
    stress_factor.fit()
    assert stress_factor.fit_result is not None


def test_sei_cycle_fitting():
    from esf.io.data import example_sample_data
    from esf.models.fitting import NonlinearFit as FitClass
    from esf.settings.parameters import get_example_params

    data = example_sample_data()
    prms = get_example_params()

    data_selector = data.cycle_life_vs_temperature
    stress_factor = FitClass(
        data_selector(),
        prms,
        regime="cycling_vs_temperature",
    )
    stress_factor.fit()
    assert stress_factor.fit_result is not None
