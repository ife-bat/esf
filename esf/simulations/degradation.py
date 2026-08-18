import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from esf.models.base_models import (
    exponential_dod_k_relation,
    exponential_soc_k_relation,
    exponential_temperature_k_relation,
    proportional_time_k_relation,
)
from esf.models.cycle_counting_algorithm import CycleCounter
from esf.settings.parameters import ESFParams
from esf.utils.converters import to_numpy_float64, to_numpy_floats64


def drive_cycle_degradation_calculator(
    df: pd.DataFrame,
    prms: ESFParams,
    cycle_numbers: np.ndarray | list | None = None,
    cycle_number_method: str | int = "default",
    peak_finder_resolution=0.01,
    repetitions=1,
    use_model_sets=True,
    verbose=False,
    **kwargs,
) -> pd.DataFrame:
    """Calculate degradation based on (n repetitions of) a given drive cycle.

    The function first finds the linear cycle-induced degradation pr. cycle (as provided) and the linear calendar
    degradation for the full drive cycle. The total degradation is then calculated using the nonlinear_total_degradation_model function.

    It is recommended that you use one single drive cycle as input, and then provide how many cycles you would like to simulate through
    the cycle_numbers argument.

    This function calculates the linear degradation based on a drive cycle by unpacking the data frame and parameters objects.
    The unpacking is not "general" and only supports the following columns in the data frame: "time", "soc", "c-rate", and "temperature",
    and the following stress models: time, soc, temperature, dod, and high-soc.

    Arguments:
        df (pandas.DataFrame): The data frame with the drive cycle information. It must contain the columns
            "time", "soc", "c-rate", and "temperature".
        prms (ESFParams): The parameters for the degradation model.
        cycle_numbers (numpy.ndarray, optional): The cycles to simulate.
        cycle_number_method (str or int optional): The method to use for calculating the cycle numbers. If an integer n is provided,
            an array of n elements in the range 1 to the total number of cycles found by the cycle counter.
        peak_finder_resolution (float, optional): The resolution for the peak finder.
        repetitions (int, optional): The number of repetitions of drive cycle in the test to simulate (defaults to 1).
        use_model_sets (bool, optional): If True, use the model sets. Default is False.
        verbose (bool, optional): If True, print the progress of the simulation.

    Returns:
        cycle info frame (pandas.DataFrame).

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from esf.settings.parameters import get_example_params
        >>> from esf.simulations.degradation import drive_cycle_degradation_calculator
        >>> from esf.simulations.cycles import drive_cycle_001
        >>>
        >>> my_drive_cycle_df = drive_cycle_001()
        >>> my_prms = get_example_params()
        >>> my_selected_cycle_numbers= np.linspace(1, 100, 10)
        >>> cycle_df = drive_cycle_degradation_calculator(my_drive_cycle_df, my_prms, cycle_numbers=my_selected_cycle_numbers)

    Note:
        If you want to work without using the dataframe input, you can find the degradation loss for each cycle by performing the following steps:

        0. Unpack the parameters (unpack_parameters)
        1. Perform cycle counting on the drive cycle (count_cycles)
        2. Use the results from the cycle counting to calculate the stress factors for the drive cycle (calculate_stress_factors)
        3. Calculate the degradation rates for the drive cycle (calculate_linear_degradation_rates)
        4. Calculate the total degradation loss for the drive cycle (calc_loss)
    """

    if prms.mode == "uncertainties":
        warnings.warn(
            f"Warning: mode is set to {prms.mode}, but this function does not support "
            f"this mode yet (sorry). "
            "Will instead convert all uncertainty objects to regular numpy floats"
        )

    if verbose:
        print("\n-> [drive_cycle_degradation_calculator]")

    if repetitions > 1:
        if cycle_numbers is None:
            df = repeat_drive_cycle(df, repetitions=repetitions)
        else:
            print(
                "It is not possible to use both repetitions and cycle_numbers -> using cycle_numbers"
            )

    # unpack the data frame and parameters:
    t, soc, temperature, c_rate = unpack_drive_cycle_frame(df)
    keyword_parameters = unpack_parameters(prms, use_model_sets=use_model_sets)

    if verbose:
        print(" keyword_parameters ".center(80, "-"))
        print(keyword_parameters)

    # perform cycle counting:
    arr_crate, arr_dod, arr_n, arr_soc_mean, arr_temp, arr_time = count_cycles(
        t,
        soc,
        c_rate,
        temperature,
        peak_finder_resolution=peak_finder_resolution,
        verbose=verbose,
        **kwargs,
    )

    # calculate the stress factors
    f_c, f_t, *stress_factors = calculate_stress_factors(
        arr_time,
        arr_soc_mean,
        arr_temp,
        arr_crate,
        arr_n,
        arr_dod,
        use_model_sets=use_model_sets,
        **keyword_parameters,
        verbose=verbose,
        **kwargs,
    )

    # pick or generate cycle numbers (where one cycle is defined as the complete drive cycle provided in the data frame)
    if cycle_numbers is None:
        # It is not straight forward to use this tool without providing cycle numbers. It might be that you need to
        # tweak some of the parameters for the cycle counting algorithm to get the correct number of cycles.
        # If you need results vs. time, it should (I think) provide similar results independent on what automatic cycle numbering
        # method is used. But it has not been tested thoroughly.
        warnings.warn(
            "Not providing cycle numbers is not recommended - you are now living dangerously."
        )
        if cycle_number_method == "cumsum":
            cycle_numbers = arr_n.cumsum()
        elif cycle_number_method == "rainflow":
            cycle_numbers = np.arange(1, len(arr_n) + 1)
        elif isinstance(cycle_number_method, int):
            cycle_numbers = np.linspace(1, arr_n.sum(), cycle_number_method)
        else:
            print(
                "Using default cycle numbers (10 numbers starting from 1 to the total number of counted cycles)"
            )
            cycle_numbers = np.linspace(1, arr_n.sum(), 10)

    # calculate the linear degradation rates (f_d) (also returns f_c and f_t summed pr cycle)
    f_c, f_t, f_d = calculate_linear_degradation_rates(
        cycle_numbers, f_c, f_t, **keyword_parameters, verbose=verbose, **kwargs
    )

    # calculate the total degradation loss
    loss = calc_loss(f_d, prms)

    delta_loss = np.diff(loss, prepend=0)
    soh = 1.0 - loss

    cycle_info_frame = pack_cycle_info_frame(
        cycle_numbers,
        delta_loss,
        arr_n,
        arr_time,
        arr_crate,
        arr_dod,
        arr_soc_mean,
        arr_temp,
        f_c,
        f_t,
        f_d,
        loss,
        soh,
    )
    return cycle_info_frame


def calc_fd(f_c: np.ndarray, f_t: np.ndarray) -> np.ndarray:
    """Calculate the linear degradation rate for one cycle."""
    f_c = f_c.sum()
    f_t = f_t.sum()
    return f_c + f_t


def calculate_linear_degradation_rates(
    cycle_numbers: np.ndarray | list,
    f_c: np.ndarray,
    f_t: np.ndarray,
    verbose: bool = False,
    **kwargs,
):
    """Calculate the linear degradation rates."""
    # Consider not returning f_c and f_t
    if verbose:
        print("   [calculating the linear degradation rates] ")
    f_c = f_c.sum()
    f_t = f_t.sum()
    f_d = cycle_numbers * f_c + f_t

    return f_c, f_t, f_d


def pack_cycle_info_frame(
    cycle_numbers,
    delta_loss,
    arr_n,
    arr_time,
    arr_crate,
    arr_dod,
    arr_soc_mean,
    arr_temp,
    f_c,
    f_t,
    f_d,
    loss,
    soh,
):
    """Pack the cycle information into a data frame.

    Arguments:
        cycle_numbers (np.ndarray): The cycle numbers.
        delta_loss (np.ndarray): The delta loss.
        arr_n (np.ndarray): The number of cycles.
        arr_time (np.ndarray): The time.
        arr_crate (np.ndarray): The c-rate.
        arr_dod (np.ndarray): The dod.
        arr_soc_mean (np.ndarray): The mean soc.
        arr_temp (np.ndarray): The temperature.
        f_c (float): The linear cycling degradation factor.
        f_t (float): The linear calendaring degradation factor.
        f_d (np.ndarray): The combined linear degradation factor vs drive cycle number.
        loss (np.ndarray): The loss.
        soh (np.ndarray): The State of Health (1-loss).
    """
    # calculate the total time and number of cycles
    t_tot_pr_cycle = arr_time.sum()
    t_tot = t_tot_pr_cycle * cycle_numbers
    n_tot = cycle_numbers
    # pack into a dataframe
    #  create arrays with the same length as the cycle numbers
    only_ones = np.ones_like(cycle_numbers)
    arr_time = only_ones * t_tot_pr_cycle
    arr_temp = only_ones * arr_temp.mean()
    arr_dod = only_ones * arr_dod.mean()
    arr_soc_mean = only_ones * arr_soc_mean.mean()
    arr_crate = only_ones * arr_crate.mean()
    arr_n = only_ones * arr_n.sum()
    f_t = only_ones * f_t
    f_c = only_ones * f_c
    #  create the cycle info frame
    cycle_info_frame = pd.DataFrame(
        {
            "cycle_number": cycle_numbers,
            "t_tot": t_tot,
            "n_tot": n_tot,
            "t": arr_time,
            "soc": arr_soc_mean,
            "temperature": arr_temp,
            "c_rate": arr_crate,
            "dod": arr_dod,
            "n": arr_n,
            "f_c": f_c,
            "f_t": f_t,
            "f_d": f_d,
            "delta_loss": delta_loss,
            "loss": loss,
            "soh": soh,
        }
    )
    return cycle_info_frame


def unpack_parameters(prms, use_model_sets=False):
    """Unpack the parameters for the degradation model.

    This function can be used to unpack the parameters for the degradation model to a dictionary
    that can be used as keyword arguments in the degradation model functions (those functions do not
    support ESFParams objects as input).

    Arguments:
        prms (ESFParams): The parameters for the degradation model.
        use_model_sets (bool, optional): If True, use the model sets. Default is False.

    Returns:
        dict: The unpacked parameters.

    Note:
        The function converts the prms to numpy float64 (i.e. propagation of errors is not supported yet).

        Also, looking forward to the model set feature is properly implemented - then most of this
        function can be removed.

    """
    # Note! the functions currently do not support uncertainties, but converts ufloats to floats
    # keyword name (as expected by calculate_stress_factors) per (regime, factor):
    keyword_names = {
        ("Calendar", "time"): "time_calendar",
        ("Calendar", "soc"): "soc_calendar",
        ("Cycling", "soc"): "soc_cycling",
        ("Calendar", "temperature"): "temperature_calendar",
        ("Cycling", "temperature"): "temperature_cycling",
        ("Cycling", "dod"): "dod",
        ("Cycling", "high_soc"): "high_soc_cycling",
        ("Calendar", "high_soc"): "high_soc_calendar",
    }
    parameters = {"cycle_unit": prms.cycle_unit}
    for (regime, factor), keyword in keyword_names.items():
        parameters[f"{keyword}_stress_model"] = prms.stress_model_function(
            regime, factor
        )
        parameters[f"{keyword}_stress_model_parameters"] = to_numpy_floats64(
            prms.stress_model_parameter_values(regime, factor)
        )

    if use_model_sets:
        for key, regime in (("calendar_models", "Calendar"), ("cycling_models", "Cycling")):
            parameters[key] = [
                (factor, function, to_numpy_floats64(values))
                for factor, function, values in prms.stress_models(regime)
            ]
    return parameters


def repeat_drive_cycle(df, repetitions=2, delta_time_estimator="constant"):
    t = df["time"].values
    soc = df["soc"].values
    temperature = df["temperature"].values
    c_rate = df["c-rate"].values
    # Convert repetitions to integer
    repetitions = int(repetitions)
    if delta_time_estimator == "constant":
        step_size = t[1] - t[0]
        total_length = int(len(t) * repetitions)
        t_end = t[0] + (total_length - 1) * step_size
        t = np.linspace(t[0], t_end, total_length)
    else:
        delta_time = t.max() - t.min()
        step_size = delta_time / (len(t) - 1)

        t_list = []

        for i in range(repetitions):
            if i == 0:
                t_list.append(t)
            else:
                t_list.append(t + (i * delta_time + step_size))
        t = np.concatenate(t_list)
    soc = np.tile(soc, repetitions)
    temperature = np.tile(temperature, repetitions)
    c_rate = np.tile(c_rate, repetitions)
    return pd.DataFrame(
        {"time": t, "soc": soc, "temperature": temperature, "c-rate": c_rate}
    )


def unpack_drive_cycle_frame(df):
    # un-packing the data (update this if new columns are added):
    df = df.dropna()
    t = df["time"].values
    soc = df["soc"].values
    temperature = df["temperature"].values
    c_rate = df["c-rate"].values
    arrays = [t, soc, temperature, c_rate]
    return arrays


def count_cycles(
    t: np.ndarray,
    soc: np.ndarray,
    c_rate: np.ndarray,
    temperature: np.ndarray,
    peak_finder_resolution: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # cycle counting
    cycle_count = CycleCounter(
        time_v=t,
        soc_v=soc,
        temp_v=temperature,
        crate_v=c_rate,
        delta=peak_finder_resolution,
        **kwargs,
    )
    cycle_count.rainflow_process()
    arr_dod = cycle_count.arr_dod
    arr_n = cycle_count.arr_n
    arr_soc_mean = cycle_count.arr_soc_mean
    arr_temp = cycle_count.arr_temp
    arr_time = cycle_count.arr_time
    arr_crate = cycle_count.arr_crate
    soc_mean = cycle_count.mean_soc
    counted_cycles = len(arr_n)

    log_message = f"Counted cycles: {counted_cycles} "
    log_message += f"Mean SOC: {soc_mean} "
    logging.debug(log_message)
    return arr_crate, arr_dod, arr_n, arr_soc_mean, arr_temp, arr_time


def calc_f(*args: np.ndarray) -> np.ndarray:
    """
    Calculate the stress factor
    """
    return np.prod(np.vstack([*args]), axis=0)


def calc_cycle_at_end_of_life(
    f_c: float, f_t: float, prms: ESFParams, eol_soh: float, verbose: bool = False
) -> float:
    """Calculate the cycle number at end of life.

    Arguments:
        f_c (float): The linear cycling degradation rate.
        f_t (float): The linear calendaring degradation rate.
        prms (ESFParams): The parameters for the degradation model.
        eol_soh (float): The end of life state of health.
        verbose (bool, optional): If True, print debug information. Defaults to False.

    Returns:
        float: The cycle number at end of life.
    """
    # TODO: check if prms already contains the eol_soh, if so, make eol_soh optional
    from scipy.optimize import fsolve

    eol_loss = 1 - eol_soh
    if not isinstance(f_c, np.ndarray):
        f_c = np.array([f_c])
    if not isinstance(f_t, np.ndarray):
        f_t = np.array([f_t])

    def _loss_at_cycle(
        cycle: np.ndarray, f_c: np.ndarray, f_t: np.ndarray, eol_loss: float
    ) -> float:
        """Helper function to find cycle where loss crosses eol_soh"""
        f_c, f_t, f_d = calculate_linear_degradation_rates(cycle, f_c, f_t)
        loss = calc_loss(f_d, prms=prms)
        return loss - eol_loss

    # Find cycle number where loss crosses eol_soh
    initial_guess = 100.0
    if verbose:
        print("Solving for cycle at end of life")
    cycle_at_eol, *_rest = fsolve(
        _loss_at_cycle, initial_guess, args=(f_c, f_t, eol_loss), full_output=True
    )
    if verbose:
        print(f"initial_guess: {initial_guess}")
        print(f"f_c: {f_c}")
        print(f"f_t: {f_t}")
        print(f"eol_soh: {eol_soh}")
        print(f"eol_loss: {eol_loss}")
        print(f"cycle_at_eol: {cycle_at_eol}")
        print("additional output:")
        print(_rest)
        print("Checking loss at end of life cycle:")
        f_c, f_t, f_d = calculate_linear_degradation_rates(cycle_at_eol, f_c, f_t)
        print(calc_loss(f_d, prms=prms)[0])

    # fsolve returns a length-1 array; index before converting (float() on a
    # 1-element ndarray is an error on numpy >= 2)
    return float(cycle_at_eol[0])


def calc_loss(f_d: np.ndarray, prms: ESFParams, **kwargs) -> np.ndarray:
    """Calculate the total degradation loss.

    Arguments:
        f_d (np.array): The linear degradation.
        prms (ESFParams): The parameters for the degradation model.

    Returns:
        np.array: The total degradation loss.

    Note:
        The function converts the prms to numpy float64 (i.e. propagation of errors is not supported yet).
    """
    from esf.models.base_models import nonlinear_total_degradation_model

    alpha_sei = to_numpy_float64(prms.sei_alpha)
    beta_sei = to_numpy_float64(prms.sei_beta)

    loss = nonlinear_total_degradation_model(alpha_sei, beta_sei, f_d)
    return loss


def calculate_stress_factors(
    arr_time: np.ndarray,
    arr_soc_mean: np.ndarray,
    arr_temp: np.ndarray,
    arr_crate: np.ndarray,
    arr_n: np.ndarray,
    arr_dod: np.ndarray,
    k_time: float = 0.00000624,
    k_soc: float = 1.03255520,
    k_temperature: float = 0.07023903,
    reference_temperature: float = 20.0,
    k_dod_1: float = 4.2701e-06,
    k_dod_2: float = 1.45764279,
    k_dod_3: float = 0.0,
    cycle_unit: str | None = None,
    time_calendar_stress_model: Callable | None = None,
    soc_calendar_stress_model: Callable | None = None,
    soc_cycling_stress_model: Callable | None = None,
    temperature_calendar_stress_model: Callable | None = None,
    temperature_cycling_stress_model: Callable | None = None,
    dod_stress_model: Callable | None = None,
    high_soc_cycling_stress_model: Callable | None = None,
    high_soc_calendar_stress_model: Callable | None = None,
    additional_calendar_models: list | None = None,
    additional_cycling_models: list | None = None,
    use_model_sets: bool = False,
    calendar_models: list | None = None,
    cycling_models: list | None = None,
    verbose: bool = False,
    **kwargs,
) -> list[np.ndarray]:
    """Calculate the degradation of a battery cell based on the ESF model.

    Arguments:

        **kwargs: Additional keyword arguments.

    """
    # TODO: move support for this to the drive cycle degradation calculator
    if cycle_unit == "EFC":
        arr_n = np.array(arr_n) * np.array(arr_dod)

    if use_model_sets:
        # TODO: fix this so that it works more generally (only soc, time, dod, and temperature are supported so far for values):
        calendar_values = {
            "soc": arr_soc_mean,
            "time": arr_time,
            "temperature": arr_temp,
        }
        cycling_values = {"soc": arr_soc_mean, "temperature": arr_temp, "dod": arr_dod}

        calendar_stress_factors = _calc_degradation_from_model_set(
            calendar_models, calendar_values, verbose=verbose
        )
        cycling_stress_factors = _calc_degradation_from_model_set(
            cycling_models, cycling_values, verbose=verbose
        )

    else:
        # -----------------------------------------------------------------------------
        # Required models calendaring:
        # -----------------------------------------------------------------------------
        time_calendar_stress_model = (
            time_calendar_stress_model or proportional_time_k_relation
        )
        time_calendar_stress_model_parameters = kwargs.get(
            "time_calendar_stress_model_parameters", (k_time,)
        )
        soc_calendar_stress_model = (
            soc_calendar_stress_model or exponential_soc_k_relation
        )
        soc_calendar_stress_model_parameters = kwargs.get(
            "soc_calendar_stress_model_parameters", (k_soc,)
        )
        temperature_calendar_stress_model = (
            temperature_calendar_stress_model or exponential_temperature_k_relation
        )
        temperature_calendar_stress_model_parameters = kwargs.get(
            "temperature_calendar_stress_model_parameters", (k_temperature,)
        )
        # -----------------------------------------------------------------------------
        # Required models cycling:
        # -----------------------------------------------------------------------------

        dod_stress_model = dod_stress_model or exponential_dod_k_relation
        dod_stress_model_parameters = kwargs.get(
            "dod_stress_model_parameters",
            (
                k_dod_1,
                k_dod_2,
                k_dod_3,
            ),
        )

        soc_cycling_stress_model = (
            soc_cycling_stress_model or exponential_soc_k_relation
        )
        soc_cycling_stress_model_parameters = kwargs.get(
            "soc_cycling_stress_model_parameters", (k_soc,)
        )
        temperature_cycling_stress_model = (
            temperature_cycling_stress_model or exponential_temperature_k_relation
        )
        temperature_cycling_stress_model_parameters = kwargs.get(
            "temperature_cycling_stress_model_parameters", (k_temperature,)
        )

        calendar_stress_factors = _calc_default_calendar_stress_factors(
            arr_soc_mean,
            arr_temp,
            arr_time,
            soc_calendar_stress_model,
            soc_calendar_stress_model_parameters,
            temperature_calendar_stress_model,
            temperature_calendar_stress_model_parameters,
            time_calendar_stress_model,
            time_calendar_stress_model_parameters,
            additional_calendar_models,
        )

        cycling_stress_factors = _calc_default_cycling_stress_factors(
            arr_dod,
            arr_n,
            arr_soc_mean,
            arr_temp,
            dod_stress_model,
            dod_stress_model_parameters,
            soc_cycling_stress_model,
            soc_cycling_stress_model_parameters,
            temperature_cycling_stress_model,
            temperature_cycling_stress_model_parameters,
            additional_cycling_models,
        )

    f_calendar = calc_f(*calendar_stress_factors)
    f_cycling = calc_f(*cycling_stress_factors)

    return f_cycling, f_calendar, calendar_stress_factors, cycling_stress_factors


def _calc_default_cycling_stress_factors(
    arr_dod,
    arr_n,
    arr_soc_mean,
    arr_temp,
    dod_stress_model,
    dod_stress_model_parameters,
    soc_stress_model,
    soc_stress_model_parameters,
    temperature_stress_model,
    temperature_stress_model_parameters,
    additional_cycling_models,
):
    dod_stress_cycling = dod_stress_model(arr_dod, *dod_stress_model_parameters)
    soc_stress_cycling = soc_stress_model(arr_soc_mean, *soc_stress_model_parameters)
    temperature_stress_cycling = temperature_stress_model(
        arr_temp, *temperature_stress_model_parameters
    )
    stress_factors = [
        dod_stress_cycling,
        soc_stress_cycling,
        temperature_stress_cycling,
    ]
    # additional stress factors:
    if additional_cycling_models is not None:
        available_variables = {
            "n": arr_n,
            "soc_mean": arr_soc_mean,
            "temperature": arr_temp,
            "dod": arr_dod,
        }
        for model_info in additional_cycling_models:
            model = model_info["model"]
            model_variable = model_info["variable"]
            model_parameters = model_info["parameters"]
            partial_additional_stress = model(
                available_variables[model_variable], *model_parameters
            )
            stress_factors.append(partial_additional_stress)
    return stress_factors


def _calc_default_calendar_stress_factors(
    arr_soc_mean,
    arr_temp,
    arr_time,
    soc_stress_model,
    soc_stress_model_parameters,
    temperature_stress_model,
    temperature_stress_model_parameters,
    time_stress_model,
    time_stress_model_parameters,
    additional_calendar_models,
):
    time_stress_calendar = time_stress_model(arr_time, *time_stress_model_parameters)
    soc_stress_calendar = soc_stress_model(arr_soc_mean, *soc_stress_model_parameters)
    temperature_stress_calendar = temperature_stress_model(
        arr_temp, *temperature_stress_model_parameters
    )
    stress_factors = [
        time_stress_calendar,
        soc_stress_calendar,
        temperature_stress_calendar,
    ]
    if additional_calendar_models is not None:
        available_variables = {
            "time": arr_time,
            "soc_mean": arr_soc_mean,
            "temperature": arr_temp,
        }

        for model_info in additional_calendar_models:
            model = model_info["model"]
            model_variable = model_info["variable"]
            model_parameters = model_info["parameters"]
            partial_additional_stress = model(
                available_variables[model_variable], *model_parameters
            )

            stress_factors.append(partial_additional_stress)
    return stress_factors


def _calc_degradation_from_model_set(models, x_values, verbose=True):
    """Calculate the degradation from a set of models."""

    stress_factors = []
    if verbose:
        print("-> [_calc_degradation_from_model_set]")

    for model_info in models:
        x_name, model_function, model_parameters = model_info
        txt = "  -> model info:\n"
        txt += f"   {x_name=}\n   {model_function=}\n   {model_parameters=}\n"

        if "soc" in x_name:
            x = x_values["soc"]
        elif "time" in x_name:
            x = x_values["time"]
        elif "temperature" in x_name:
            x = x_values["temperature"]
        elif "dod" in x_name:
            x = x_values["dod"]
        else:
            x = x_values[x_name]
        _stress = model_function(x, *model_parameters)
        txt += f"   {x=}\n   partial stress={_stress}\n"
        stress_factors.append(_stress)

        if verbose:
            print(txt)
    if verbose:
        print("[_calc_degradation_from_model_set] [done]\n")
    return stress_factors
