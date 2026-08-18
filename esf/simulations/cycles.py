import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esf.settings.parameters import CATEGORICAL_COLORS
from esf.settings.units import Q_

INTERNAL_TIME_UNIT = "hours"
INTERNAL_POWER_UNIT = "watts"
_capacity_unit = Q_(1, INTERNAL_POWER_UNIT) * Q_(1, INTERNAL_TIME_UNIT)
INTERNAL_CAPACITY_UNIT = _capacity_unit.units

def sim_cycles(
    drive_cycle,
    number_of_cycles,
    delta_time,
    soc_cut_off=0.2,
    t_cut_off=100.0,
    temperature=None,
    t_start=0.0,
    soc_start=1.0,
):
    cycles = range(1, number_of_cycles + 1)
    t_0 = t_start
    soc_0 = soc_start
    frames = []
    for cycle_number in cycles:
        for step_number, step in enumerate(drive_cycle):
            step_df = add_step(
                step,
                step_number + 1,
                cycle_number,
                soc_0,
                t_0,
                temperature=temperature,
                delta_time=delta_time,
                soc_cut_off=soc_cut_off,
                t_cut_off=t_cut_off,
            )
            t_0 = step_df["time"].iloc[-1] + delta_time
            soc_0 = step_df["soc"].iloc[-1]

            frames.append(step_df)
    df = pd.concat(frames)
    df = df.reset_index(drop=True)
    return cycles, df


def add_step(
    step: dict,
    step_number: int,
    cycle_number: int = 1,
    soc_0: float = 1.0,
    t_0: float = 0.0,
    temperature: float | None = None,
    delta_time: float = 0.01,
    soc_cut_off: float = 0.2,
    t_cut_off: float = 10.0,
) -> pd.DataFrame:
    """Add a step to the simulation."""

    # t_1 = step.get("t") + delta_time  # hours
    # c_rate = step.get("r")  # C rate (cap/h)
    # delta_soc = c_rate * delta_time
    # t = np.arange(0.0, t_1, delta_time)
    # soc = soc_0 + t * c_rate

    c_rate, delta_soc, t, soc = _calculate_soc(step, soc_0, delta_time)

    # checking cut-offs (allowing for some margin):
    t_mask = t <= (t_cut_off + delta_time / 10.0)
    soc_mask_min = soc >= (soc_cut_off + np.abs(delta_soc / 10.0))
    soc_mask_max = soc <= (1.0 + np.abs(delta_soc / 10.0))
    mask = t_mask & soc_mask_min & soc_mask_max
    t = t[mask] + t_0
    soc = soc[mask]

    step_number = np.full_like(t, step_number)
    cycle_number = np.full_like(t, cycle_number)
    c_rate = np.full_like(t, c_rate)

    d = {
        "time": t,
        "cycle": cycle_number,
        "step": step_number,
        "soc": soc,
        "c-rate": c_rate,
    }

    # TODO: consider making it possible to add temperature profile (and not used one value) inside here
    #  (though it is always possible to overwrite the temperature column afterwards)
    if temperature is not None:
        d["temperature"] = np.full_like(t, temperature)
    df = pd.DataFrame(d)
    return df


def _calculate_soc(step, soc_0, delta_time):
    t_1 = step.get("t") + delta_time  # hours
    c_rate = step.get("r")  # C rate (cap/h)
    delta_soc = c_rate * delta_time
    t = np.arange(0.0, t_1, delta_time)
    soc = soc_0 + t * c_rate
    return c_rate, delta_soc, t, soc


def drive_cycle_001(
    verbose: bool = True,
    temperature: float = 298.15,
    time_unit="hours",
) -> pd.DataFrame:
    """Simulate a drive cycle.

    This is a simple drive simulation, where the battery is discharged and charged
    according to a simple drive cycle. The simulation is performed for a number of cycles.

    This is meant as an example of how to use the simulation functions in the ESF package.

    Arguments:
        verbose (bool, optional): If True, print information about the simulation.
        temperature (float, optional): The temperature to simulate.
        time_unit (str, optional): The unit of time to return the data frame in.

    Returns:
        pandas.DataFrame: The data frame with the simulation results.
    """

    step_discharge = dict(
        t=0.5, r=-1.0, l="discharge"
    )  # Note! time unit is in hours, rate unit is in C rate!
    step_charge = dict(t=5.0, r=0.2, l="charge")
    step_rest_1h = dict(t=1.0, r=0.0, l="rest")
    step_discharge_fast = dict(t=0.5, r=-3.0, l="discharge")
    step_charge_fast = dict(t=0.01, r=3.0, l="charge")
    step_rest_short = dict(t=0.1, r=0.0, l="rest")

    drive_cycle = [
        step_discharge,
        step_rest_1h,
        step_discharge_fast,
        step_charge_fast,
        step_rest_short,
        step_charge,
        step_rest_1h,
    ]
    if verbose:
        print(f"temperature: {temperature}")
        print(f"drive cycle: {drive_cycle}")

    # just for measuring how much time it takes for performing the simulation:
    start_time = time.time()

    # Setting up the simulation:
    soc_start = 1.0
    delta_time = 0.05  # hours
    t_cut_off = 10.0  # hours
    soc_cut_off = 0.2
    number_of_cycles = 1

    if verbose:
        print(f"Simulating {number_of_cycles} cycles...")

    cycles, df = sim_cycles(
        drive_cycle,
        number_of_cycles,
        delta_time,
        soc_cut_off,
        t_cut_off,
        temperature,
        soc_start=soc_start,
    )

    # just for measuring how much time it takes for performing the simulation:
    end_time = time.time()
    tot_time = end_time - start_time
    time_pr_cycle = tot_time / len(cycles)

    # convert time to the requested unit (is currently given in hours):

    if time_unit != "hours":
        try:
            df["time"] = _convert_time(df["time"], "hours", time_unit)
        except ValueError as err:
            raise ValueError(f"Invalid time unit: {time_unit}") from err

    if verbose:
        print(
            f"Elapsed time: {tot_time:.4f} hours ({time_pr_cycle:.4f} hours per cycle)"
        )
        print(f"Generated data frame (time unit: {time_unit}):")
        print(df.head(20))

    return df


def _convert_time(time, time_unit_from, time_unit_to):
    # Check if time is a pandas Series and handle accordingly
    if isinstance(time, pd.Series):
        # Apply the conversion to each element in the Series
        return time.apply(lambda x: Q_(x, time_unit_from).to(time_unit_to).magnitude)
    else:
        # Handle scalar
        return Q_(time, time_unit_from).to(time_unit_to).magnitude


def _convert_unit(value, unit_from, unit_to):
    if isinstance(value, pd.Series):
        # Apply the conversion to each element in the Series
        return value.apply(lambda x: Q_(x, unit_from).to(unit_to).magnitude)
    return Q_(value, unit_from).to(unit_to).magnitude


def drive_cycle_002(
    verbose: bool = True,
    temperature: float = 298.15,
    time_unit="hours",
) -> pd.DataFrame:
    """Simulate a drive cycle.

    This is a simple drive simulation, where the battery is discharged and charged
    according to a simple drive cycle. The simulation is performed for a number of cycles.

    This is meant as an example of how to use the simulation functions in the ESF package.

    Arguments:
        verbose (bool, optional): If True, print information about the simulation.
        temperature (float, optional): The temperature to simulate.
        time_unit (str, optional): The unit of time to return the data frame in.

    Returns:
        pandas.DataFrame: The data frame with the simulation results.
    """
    step_discharge = dict(
        t=0.6, r=-1.0, l="discharge"
    )  # Note! time unit is in hours, rate unit is in C rate!
    step_charge = dict(t=5.0, r=0.2, l="charge")
    step_rest_long = dict(t=12.0, r=0.0, l="rest")
    step_discharge_fast = dict(t=0.5, r=-4.0, l="discharge")
    step_charge_fast = dict(t=0.01, r=3.0, l="charge")
    step_rest_short = dict(t=1.0, r=0.0, l="rest")

    drive_cycle = [
        step_discharge,
        step_rest_long,
        step_discharge_fast,
        step_charge_fast,
        step_rest_short,
        step_charge,
        step_rest_long,
    ]
    if verbose:
        print(f"temperature: {temperature}")
        print(f"drive cycle: {drive_cycle}")

    # just for measuring how much time it takes for performing the simulation:
    start_time = time.time()

    # Setting up the simulation:
    soc_start = 1.0
    delta_time = 0.05  # hours
    t_cut_off = 10.0  # hours
    soc_cut_off = 0.2
    number_of_cycles = 1

    if verbose:
        print(f"Simulating {number_of_cycles} cycles...")

    cycles, df = sim_cycles(
        drive_cycle,
        number_of_cycles,
        delta_time,
        soc_cut_off,
        t_cut_off,
        temperature,
        soc_start=soc_start,
    )

    # just for measuring how much time it takes for performing the simulation:
    end_time = time.time()
    tot_time = end_time - start_time
    time_pr_cycle = tot_time / len(cycles)

    # convert time to the requested unit (is currently given in hours):
    if time_unit != "hours":
        try:
            df["time"] = _convert_time(df["time"], "hours", time_unit)
        except ValueError as err:
            raise ValueError(f"Invalid time unit: {time_unit}") from err

    if verbose:
        print(
            f"Elapsed time: {tot_time:.4f} hours ({time_pr_cycle:.4f} hours per cycle)"
        )
        print(f"Generated data frame (time unit: {time_unit}):")
        print(df.head(20))

    return df


def _time_integrated_power(
    df: pd.DataFrame, direction: str = "positive", power_col: str = "power"
) -> float:
    """Calculate the time integrated positive power of the drive cycle."""
    from scipy import integrate

    # Units: [W] * [h] = [Wh]
    # Columns: "power" and "time"

    if direction == "positive":
        df = df[df[power_col] > 0]
    elif direction == "negative":
        df = df[df[power_col] < 0]
        df[power_col] = df[power_col] * -1.0
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return integrate.trapezoid(df[power_col], df["time"])


def _time_power_cumsum(
    df: pd.DataFrame,
    power_col: str = "power",
    time_col: str = "time",
    regenerative: bool = True,
) -> pd.Series:
    """Calculate the time of the power peaks of the drive cycle."""
    df["delta_time"] = df[time_col].diff()

    df["time_times_power"] = df["delta_time"] * df[power_col]
    if not regenerative:
        mask = df[power_col] > 0
        df.loc[mask, "time_times_power"] = 0.0
    df["cap_usage"] = df["time_times_power"].cumsum()
    return df["cap_usage"]


def _convert_from_power_to_c_rate(
    power: float | pd.Series | np.ndarray,
    battery_capacity: float,
    power_unit: str = "watts",
    battery_capacity_unit: str = "watts*hours",
) -> float:
    """Convert a power in watts to a c-rate."""

    battery_capacity = (
        Q_(battery_capacity, battery_capacity_unit).to(f"{power_unit}*hours").magnitude
    )
    if isinstance(power, pd.Series):
        return power.apply(lambda x: x / battery_capacity)
    return power / battery_capacity


def plot_drive_cycle(
    df: pd.DataFrame,
    figsize: tuple | None = None,
    soc_cut_off: float = 0.2,
    time_unit="hours",
    rate_unit="C-rate",
) -> None:
    """Plot the drive cycle simulation.

    Args:
        df (pandas.DataFrame): The data frame with the simulation results. It must contain the columns
            "time", "soc", and "c-rate".
        figsize (tuple, optional): The size of the figure.
        soc_cut_off (float, optional): The state of charge cut-off.
        time_unit (str, optional): The unit of time.
        rate_unit (str, optional): The unit of the rate.

    """
    figsize = figsize or (8, 10)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle("Battery usage simulation", fontweight="bold")
    ax[0].plot(
        df["time"], df["soc"], "-", label="SOC", lw=0.5, color=CATEGORICAL_COLORS[0]
    )
    ax[0].hlines(
        1.0,
        df["time"].min(),
        df["time"].max(),
        color=CATEGORICAL_COLORS[2],
        lw=2.0,
        alpha=0.5,
        linestyle="--",
    )
    ax[0].hlines(
        soc_cut_off,
        df["time"].min(),
        df["time"].max(),
        color=CATEGORICAL_COLORS[2],
        lw=2.0,
        alpha=0.5,
        linestyle="--",
    )
    ax[1].plot(
        df["time"],
        df["c-rate"],
        "-",
        label="C-rate",
        lw=0.5,
        color=CATEGORICAL_COLORS[1],
    )
    ax[1].set_xlabel(f"Time [{time_unit}]")
    ax[0].set_ylabel("SoC [frac.]")
    ax[1].set_ylabel(f"Rate [{rate_unit}]")
    ax[0].set_ylim([-0.02, 1.02])
    ax[1].set_ylim([-8.5, 8.5])
    ax[1].set_xlim([df["time"].min(), df["time"].max()])
    fig.tight_layout()
    fig.align_ylabels()


def load_drive_cycle(
    path: pathlib.Path | str | list[pathlib.Path | str],
    seperator: str = ";",
    index_col: str | None = None,
    power_col: str = "Watts",
    time_col: str = "Sec",
    time_unit: str = "seconds",
    power_unit: str = "W",
):
    """Load a drive cycle from a file or a list of files.

    Args:
        path: The path to the file or a list of paths to files.
        seperator: The seperator of the file.
        index_col: The column to use as the index.
        power_col: The column to use as the power.
        time_col: The column to use as the time.
        time_unit: The unit of the time used in the file.
        power_unit: The unit of the power used in the file.
    """
    if isinstance(path, (pathlib.Path, str)):
        df = pd.read_csv(path, sep=seperator, index_col=index_col)

    elif isinstance(path, list):
        df_list = []
        initial_time = 0.0
        for p in path:
            df = pd.read_csv(p, sep=seperator, index_col=index_col)
            df[time_col] = df[time_col] + initial_time
            df_list.append(df)
            initial_time = df[time_col].max()

        df = pd.concat(df_list)

    df = df.rename(columns={power_col: "power", time_col: "time"})
    df["power"] = df["power"].astype(float)
    df["time"] = df["time"].astype(float)
    df["time"] = _convert_unit(df["time"], time_unit, INTERNAL_TIME_UNIT)
    df["power"] = _convert_unit(df["power"], power_unit, INTERNAL_POWER_UNIT)
    return df


def convert_power_cycle_to_rate_cycle(
    df: pd.DataFrame,
    initial_soc: float = 1.0,
    temperature: float = 25,
    battery_oversize_factor: float = 1.0,
    use_regeneration: bool = True,
    regeneration_factor: float = 1.0,
    verbose: bool = True,
):
    """Convert a power cycle to a rate cycle.

    Args:
        df: The drive cycle to convert (power cycle, i.e. power vs time).
        initial_soc: The initial state of charge.
        temperature: The temperature.
        battery_oversize_factor: The battery oversize factor.
        use_regeneration: Whether to use regeneration.
        regeneration_factor: The regeneration factor.
        verbose: Whether to print verbose output.

    Returns:
        df: The drive cycle with the converted rate (c-rate) and SoC.
    """

    # TODO: handle additional columns in the drive cycle (e.g. temperature)

    main_columns = ["time", "soc", "c-rate", "temperature"]

    # Calculate the minimum power needed
    minimum_power_needed = df["power"].max()
    minimum_capacity_needed_without_regeneration = _time_integrated_power(
        df, direction="positive"
    )

    df["cumulated_power_times_time"] = _time_power_cumsum(
        df, regenerative=use_regeneration
    )

    usage_need = df["cumulated_power_times_time"].abs().max()

    if use_regeneration:
        battery_capacity = usage_need * regeneration_factor * battery_oversize_factor
    else:
        battery_capacity = (
            minimum_capacity_needed_without_regeneration * battery_oversize_factor
        )
    if verbose:
        print(80 * "-")
        print(
            f"Minimum power needed: {minimum_power_needed:.2f} {INTERNAL_POWER_UNIT}."
        )
        details = f"(oversized by factor {battery_oversize_factor:.2f})"
        if not use_regeneration:
            print(
                f"Minimum capacity needed without regeneration: {minimum_capacity_needed_without_regeneration:.2f} {INTERNAL_CAPACITY_UNIT}."
            )
        else:
            print(
                f"Minimum capacity needed with regeneration: {usage_need:.2f} {INTERNAL_CAPACITY_UNIT}."
            )
            details = f"{details}(regeneration efficiency {regeneration_factor:.2f})"
        print(
            f"Battery capacity: {battery_capacity:.2f} {INTERNAL_CAPACITY_UNIT} {details}."
        )
        print(80 * "-")

    df["c-rate"] = _convert_from_power_to_c_rate(
        df["power"], battery_capacity, INTERNAL_POWER_UNIT, INTERNAL_CAPACITY_UNIT
    )

    df["soc"] = initial_soc + df["cumulated_power_times_time"] / battery_capacity

    if df["soc"].min() < 0.0:
        print("Warning: SOC is less than 0.0 at some point in time.")
        if initial_soc < 1.0:
            print("Warning: Initial SOC is less than 1.0.")

    df["temperature"] = temperature

    df = df[main_columns]
    return df


def resample_drive_cycle(df: pd.DataFrame, time_unit: str = "1min"):
    """Resample the drive cycle to a time step.

    Args:
        df: The drive cycle to resample.
        time_unit: The new time step (e.g. "1min", "1h").
    """
    df = df.copy()
    df["time"] = _convert_unit(df["time"], INTERNAL_TIME_UNIT, "seconds")
    df["t"] = pd.to_datetime(df["time"], unit="s")
    df["dt"] = df["t"] - df["t"].iloc[0]
    df = df.set_index("t")
    df = df.drop(columns=["time"])
    df = df.resample(time_unit).mean()
    df = df.reset_index()
    df["time"] = df["dt"].dt.total_seconds()
    df = df.drop(columns=["t", "dt"])

    if INTERNAL_TIME_UNIT != "seconds":
        df["time"] = _convert_unit(df["time"], "seconds", INTERNAL_TIME_UNIT)

    return df
