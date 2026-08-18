"""Smoke tests for the fit plotting paths.

These pin that each fit type can render its result figure (headless, Agg).
They guard the extraction of the plotting code into esf.models.fit_plotting.
"""

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from esf.io.data import SampleData
from esf.models.base_models import (
    exponential_soc_k_relation,
    exponential_temperature_k_relation,
    nonlinear_cal_model,
)
from esf.models.fitting import SoCSFfit, degradation_rates_fit
from esf.settings.parameters import DataType, get_example_params


def test_stress_factor_fit_plot_returns_figure():
    soc = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
    rates = 3.6e-5 * exponential_soc_k_relation(soc, 1.3)
    frame = pd.DataFrame({"SoC": soc, "deg_rate": rates})

    fit = SoCSFfit(frame, get_example_params(), regime="calend")
    fit.fit()
    fig = fit.plot_results()

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2  # main + residuals
    matplotlib.pyplot.close(fig)


def test_nonlinear_multi_fit_plot_returns_figure():
    t_days = np.concatenate([np.linspace(0.5, 30, 15), np.linspace(50, 3000, 25)])
    frames = []
    for temperature in (288.15, 298.15, 308.15):
        stress = float(
            exponential_temperature_k_relation(
                np.array([temperature]), 0.0693, x_ref=298.15
            )[0]
        )
        loss = nonlinear_cal_model(t_days * 86_400.0, 0.06, 110.0, 4.5e-4 * stress)
        frame = pd.DataFrame({"t": t_days, "SoH": 1 - loss})
        frame["T"] = temperature
        frames.append(frame)
    data = SampleData()
    data.add_data(
        pd.concat(frames),
        data_type=DataType.CALENDAR_VS_TEMPERATURE,
        time_unit="days",
        temperature_unit="K",
    )
    data.calculate_life_fraction()
    selection = data.calendar_life_vs_temperature(strict_mode=False)

    prms = get_example_params()
    prms.sei_alpha, prms.sei_beta = 0.06, 110.0
    fit = degradation_rates_fit(
        prms,
        selection,
        data_type=DataType.CALENDAR_VS_TEMPERATURE,
        return_fit_object=True,
    )
    fig = fit.plot_results()

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    matplotlib.pyplot.close(fig)
