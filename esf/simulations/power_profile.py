"""Build a drive cycle from a battery power trace.

``drive_cycle_degradation_calculator`` consumes ``time`` (s), ``soc`` (0–1),
``c-rate`` and ``temperature`` (K). A power trace supplies none of the last
three. SoC and C-rate follow from the pack energy, the starting SoC and the
sign of the power column; temperature is stamped on as a constant.

C-rate is power into the pack divided by pack energy (W / Wh, so per hour).
Negative C-rate discharges and SoC falls, which is the sign the rest of the
package uses.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import pandas as pd
from pint.errors import DimensionalityError, UndefinedUnitError

from esf.settings.units import Q_

# A trace that lands on 0 or 1 within this tolerance is inside the window.
_SOC_ATOL = 1e-9

DRIVE_CYCLE_COLUMNS = ["time", "soc", "c-rate", "temperature"]

PositivePower = Literal["discharge", "charge"]
PowerSource = Literal["terminal", "traction"]


class PowerProfileError(ValueError):
    """A power trace could not be turned into a drive cycle.

    ``minimum_capacity_wh`` is set when the trace walks SoC outside 0–1 and a
    larger pack would have kept it inside from the same starting SoC. It is
    ``None`` when no pack size would, because the battery is already full and
    the trace charges it, or already empty and the trace discharges it.
    """

    def __init__(self, message: str, *, minimum_capacity_wh: float | None = None):
        super().__init__(message)
        self.minimum_capacity_wh = minimum_capacity_wh


def drive_cycle_from_power(
    time,
    power,
    *,
    capacity_wh: float,
    initial_soc: float,
    temperature: float,
    temperature_unit: str = "degC",
    time_unit: str = "seconds",
    power_unit: str = "W",
    positive: PositivePower = "discharge",
    source: PowerSource = "terminal",
    regen_efficiency: float = 1.0,
) -> pd.DataFrame:
    """Turn a power-versus-time trace into a drive-cycle frame.

    Args:
        time: elapsed time, one value per sample, in ``time_unit``.
        power: power at the same samples, in ``power_unit``.
        capacity_wh: pack energy capacity in watt-hours. C-rate is
            ``power_into_the_pack / capacity_wh``.
        initial_soc: state of charge of the first sample, as a fraction.
        temperature: constant temperature, in ``temperature_unit``. Stored
            as kelvin.
        temperature_unit: pint unit of ``temperature``. ``"degC"`` or ``"K"``.
        time_unit: pint unit of ``time``. The returned ``time`` column is
            seconds.
        power_unit: pint unit of ``power``.
        positive: what a positive power means. ``"discharge"`` is power
            leaving the battery (SoC falls). ``"charge"`` is power entering
            it (SoC rises).
        source: ``"terminal"`` uses the measured battery power as-is.
            ``"traction"`` treats it as a drivetrain request and returns
            regen to the pack scaled by ``regen_efficiency``. Discharge is
            not scaled.
        regen_efficiency: fraction of regen power that reaches the pack,
            in ``(0, 1]``. Only used when ``source`` is ``"traction"``.

    Returns:
        Frame with ``time`` (s), ``soc`` (0–1), ``c-rate`` and
        ``temperature`` (K). SoC is the trapezoidal integral of power into
        the pack. C-rate is the sample's own power into the pack.

    Raises:
        PowerProfileError: the trace is unusable, or SoC leaves 0–1. The
            latter names the smallest pack that would have stayed inside,
            when one exists.
    """
    time_s = _elapsed_seconds(_numeric_1d(time, "time"), time_unit)
    power_w = _to_unit(_numeric_1d(power, "power"), power_unit, "watt", "power")
    if len(time_s) != len(power_w):
        raise PowerProfileError(
            f"time has {len(time_s)} samples and power has {len(power_w)}."
        )
    _require_increasing(time_s)
    capacity = _positive_finite(capacity_wh, "capacity_wh")
    start = _fraction(initial_soc, "initial_soc")
    temperature_k = _kelvin(temperature, temperature_unit)
    into_w = _power_into_battery_w(
        power_w,
        positive=positive,
        source=source,
        regen_efficiency=regen_efficiency,
    )

    energy_wh = _cumulative_energy_wh(time_s, into_w)
    soc = start + energy_wh / capacity
    if np.any(soc < -_SOC_ATOL) or np.any(soc > 1.0 + _SOC_ATOL):
        raise _soc_window_error(soc, capacity, start, energy_wh)

    return pd.DataFrame(
        {
            "time": time_s,
            "soc": soc,
            "c-rate": into_w / capacity,
            "temperature": np.full(len(time_s), temperature_k),
        }
    )


def drive_cycle_from_power_csv(
    path: str | pathlib.Path,
    *,
    capacity_wh: float,
    initial_soc: float,
    temperature: float,
    temperature_unit: str = "degC",
    positive: PositivePower = "discharge",
    time_col: str = "Time (s)",
    power_col: str = "Battery Power (W)",
    separator: str = ",",
    time_unit: str = "seconds",
    power_unit: str = "W",
    source: PowerSource = "terminal",
    regen_efficiency: float = 1.0,
) -> pd.DataFrame:
    """Read a power-profile CSV and build a drive-cycle frame.

    Defaults match a file with columns ``Time (s)`` and ``Battery Power (W)``
    separated by commas. See :func:`drive_cycle_from_power` for the physics
    and the remaining arguments.

    Raises:
        PowerProfileError: the file is missing a column, or the trace cannot
            be converted.
    """
    frame = pd.read_csv(path, sep=separator)
    missing = [name for name in (time_col, power_col) if name not in frame.columns]
    if missing:
        present = ", ".join(repr(column) for column in frame.columns)
        wanted = ", ".join(repr(name) for name in missing)
        raise PowerProfileError(
            f"Power-profile CSV is missing {wanted}. Columns present: {present}."
        )
    return drive_cycle_from_power(
        frame[time_col],
        frame[power_col],
        capacity_wh=capacity_wh,
        initial_soc=initial_soc,
        temperature=temperature,
        temperature_unit=temperature_unit,
        time_unit=time_unit,
        power_unit=power_unit,
        positive=positive,
        source=source,
        regen_efficiency=regen_efficiency,
    )


def _numeric_1d(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise PowerProfileError(f"{name} must be one-dimensional.")
    numeric = pd.to_numeric(pd.Series(array), errors="coerce")
    if numeric.isna().any():
        row = int(numeric.isna().to_numpy().argmax())
        raise PowerProfileError(
            f"{name} contains a missing or non-numeric value at sample {row + 1}."
        )
    out = numeric.to_numpy(dtype=float)
    if len(out) < 2:
        raise PowerProfileError(
            f"{name} needs at least two samples; got {len(out)}."
        )
    return out


def _elapsed_seconds(time: np.ndarray, unit: str) -> np.ndarray:
    return _to_unit(time, unit, "second", "time")


def _to_unit(values: np.ndarray, unit_from: str, unit_to: str, what: str) -> np.ndarray:
    try:
        converted = Q_(values, unit_from).to(unit_to).magnitude
    except (UndefinedUnitError, DimensionalityError) as exc:
        raise PowerProfileError(
            f"Cannot convert {what} from {unit_from!r} to {unit_to}."
        ) from exc
    return np.asarray(converted, dtype=float)


def _kelvin(temperature: float, unit: str) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise PowerProfileError("temperature must be a number.") from exc
    if not np.isfinite(value):
        raise PowerProfileError("temperature must be finite.")
    try:
        kelvin = float(Q_(value, unit).to("kelvin").magnitude)
    except (UndefinedUnitError, DimensionalityError) as exc:
        raise PowerProfileError(
            f"Cannot convert temperature from {unit!r} to kelvin."
        ) from exc
    return kelvin


def _positive_finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PowerProfileError(f"{name} must be a number.") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise PowerProfileError(f"{name} must be a positive finite number.")
    return number


def _fraction(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PowerProfileError(f"{name} must be a number.") from exc
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise PowerProfileError(f"{name} must lie between 0 and 1.")
    return number


def _require_increasing(time_s: np.ndarray) -> None:
    steps = np.diff(time_s)
    if np.any(steps <= 0.0):
        index = int(np.argmax(steps <= 0.0)) + 1
        raise PowerProfileError(
            "Time must increase down the trace; it does not at sample "
            f"{index + 1} ({time_s[index - 1]:g} -> {time_s[index]:g} s)."
        )


def _power_into_battery_w(
    power_w: np.ndarray,
    *,
    positive: str,
    source: str,
    regen_efficiency: float,
) -> np.ndarray:
    if positive not in ("discharge", "charge"):
        raise PowerProfileError(
            f"positive={positive!r} is not supported. Use 'discharge' or 'charge'."
        )
    if source not in ("terminal", "traction"):
        raise PowerProfileError(
            f"source={source!r} is not supported. Use 'terminal' or 'traction'."
        )
    try:
        efficiency = float(regen_efficiency)
    except (TypeError, ValueError) as exc:
        raise PowerProfileError("regen_efficiency must be a number.") from exc
    if not np.isfinite(efficiency) or efficiency <= 0.0 or efficiency > 1.0:
        raise PowerProfileError("regen_efficiency must lie in (0, 1].")
    if source == "terminal" and efficiency != 1.0:
        raise PowerProfileError(
            "regen_efficiency applies to traction power. This trace is "
            "battery-terminal power, so leave regen_efficiency at 1."
        )

    # Positive demand: the battery is asked to supply power.
    demand_w = power_w if positive == "discharge" else -power_w
    if source == "terminal":
        return -demand_w

    into = -demand_w
    regenerating = demand_w < 0.0
    into[regenerating] = into[regenerating] * efficiency
    return into


def _cumulative_energy_wh(time_s: np.ndarray, power_into_w: np.ndarray) -> np.ndarray:
    """Trapezoidal watt-hours into the pack. The first sample is zero."""
    dt_h = np.diff(time_s) / 3600.0
    segment = 0.5 * (power_into_w[:-1] + power_into_w[1:]) * dt_h
    energy = np.empty(len(time_s), dtype=float)
    energy[0] = 0.0
    energy[1:] = np.cumsum(segment)
    return energy


def _minimum_capacity_wh(energy_wh: np.ndarray, initial_soc: float) -> float | None:
    """Smallest Wh that keeps ``initial_soc + energy/capacity`` inside 0–1.

    ``None`` when the starting SoC leaves no room on a side the trace uses.
    """
    lowest = float(np.min(energy_wh))
    highest = float(np.max(energy_wh))
    bounds: list[float] = []
    if lowest < 0.0:
        if initial_soc <= 0.0:
            return None
        bounds.append(-lowest / initial_soc)
    if highest > 0.0:
        headroom = 1.0 - initial_soc
        if headroom <= 0.0:
            return None
        bounds.append(highest / headroom)
    if not bounds:
        return 0.0
    return max(bounds)


def _soc_window_error(
    soc: np.ndarray,
    capacity_wh: float,
    initial_soc: float,
    energy_wh: np.ndarray,
) -> PowerProfileError:
    minimum = _minimum_capacity_wh(energy_wh, initial_soc)
    head = (
        f"State of charge leaves 0–1 ({float(soc.min()):.4g} to "
        f"{float(soc.max()):.4g}) for a {capacity_wh:.6g} Wh pack starting "
        f"at {initial_soc:.4g}."
    )
    if minimum is None and initial_soc >= 1.0 and float(np.max(energy_wh)) > 0.0:
        detail = (
            "No pack size keeps this trace inside 0–1 from a full battery, "
            "because the trace charges it further. Lower the starting SoC."
        )
    elif minimum is None and initial_soc <= 0.0 and float(np.min(energy_wh)) < 0.0:
        detail = (
            "No pack size keeps this trace inside 0–1 from an empty battery, "
            "because the trace discharges it further. Raise the starting SoC."
        )
    elif minimum is None:
        detail = "No pack size keeps this trace inside 0–1 from this starting SoC."
    else:
        detail = (
            "The smallest pack that stays inside 0–1 from that starting SoC "
            f"is {minimum:.6g} Wh."
        )
    return PowerProfileError(f"{head} {detail}", minimum_capacity_wh=minimum)
