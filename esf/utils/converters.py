import numpy as np
import uncertainties

from esf.settings.parameters import TIME_UNIT, ureg


def to_seconds(t, unit=TIME_UNIT):
    """Convert time to seconds."""
    t = ureg.Quantity(t, ureg[unit])
    return t.to(ureg["seconds"]).magnitude


def to_hours(t, unit=TIME_UNIT):
    """Convert time to hours."""
    t = ureg.Quantity(t, ureg[unit])
    return t.to(ureg["hours"]).magnitude


def to_numpy_float64(x):
    if isinstance(x, ureg.Quantity):
        x = x.magnitude
    elif isinstance(x, uncertainties.UFloat):
        x = x.nominal_value
    return np.float64(x)


def to_numpy_floats64(x_values):
    if isinstance(x_values, tuple):
        return tuple(to_numpy_float64(x) for x in x_values)
    if isinstance(x_values, list):
        return [to_numpy_float64(x) for x in x_values]
    if isinstance(x_values, dict):
        return {k: to_numpy_float64(v) for k, v in x_values.items()}
    return tuple(to_numpy_float64(x) for x in x_values)
