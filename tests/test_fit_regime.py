"""Tests for the centralized regime classification (finding F4).

The scattered ``regime.startswith(...)`` checks were replaced by
``is_cycling_regime`` / ``is_calendar_regime`` and the ``is_cycling`` /
``is_calendar`` fit properties. These pin that classification for every form
the ``regime`` takes (enum, bare string, decorated string).
"""

import numpy as np
import pandas as pd
import pytest

from esf.models.fitting import (
    SoCSFfit,
    coerce_regime,
    is_calendar_regime,
    is_cycling_regime,
)
from esf.settings.parameters import Regime, get_example_params


@pytest.mark.parametrize(
    "regime",
    [
        Regime.CYCLING,
        "cycling",
        "cycling_vs_temperature",
        "cycling_vs_dod",
    ],
)
def test_cycling_regimes(regime):
    assert is_cycling_regime(regime)
    assert not is_calendar_regime(regime)


@pytest.mark.parametrize(
    "regime",
    [
        Regime.CALENDAR,
        "calend",
        "calend_vs_temperature",
        "calend_vs_soc",
    ],
)
def test_calendar_regimes(regime):
    assert is_calendar_regime(regime)
    assert not is_cycling_regime(regime)


def test_operational_is_neither():
    assert not is_cycling_regime(Regime.OPERATIONAL)
    assert not is_calendar_regime(Regime.OPERATIONAL)


def test_fit_properties_match_regime():
    frame = pd.DataFrame(
        {"SoC": np.array([0.3, 0.5, 0.7]), "deg_rate": np.array([1.0, 1.0, 1.1])}
    )
    prms = get_example_params()

    calendar_fit = SoCSFfit(frame, prms, regime="calend")
    assert calendar_fit.is_calendar
    assert not calendar_fit.is_cycling

    cycling_fit = SoCSFfit(frame, prms, regime="cycling")
    assert cycling_fit.is_cycling
    assert not cycling_fit.is_calendar


class TestCoerceRegime:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (Regime.CYCLING, Regime.CYCLING),
            ("cycling", Regime.CYCLING),
            ("cycling_vs_temperature", Regime.CYCLING),
            (Regime.CALENDAR, Regime.CALENDAR),
            ("calend", Regime.CALENDAR),
            ("calend_vs_soc", Regime.CALENDAR),
            ("operational", Regime.OPERATIONAL),
            (None, None),
        ],
    )
    def test_coerces_to_enum(self, value, expected):
        assert coerce_regime(value) is expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unrecognized regime"):
            coerce_regime("banana")


def test_fit_regime_is_coerced_to_enum():
    # A4: the fit stores a Regime enum, not the decorated string passed in
    frame = pd.DataFrame(
        {"SoC": np.array([0.3, 0.5, 0.7]), "deg_rate": np.array([1.0, 1.0, 1.1])}
    )
    fit = SoCSFfit(frame, get_example_params(), regime="cycling_vs_temperature")
    assert fit.regime is Regime.CYCLING
    assert type(fit.regime).__name__ == "Regime"
