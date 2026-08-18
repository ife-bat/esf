import numpy as np


# -------- Calendar degradation model ----------------------------------
def nonlinear_degradation_model(x, alpha_sei, beta_sei, degradation_rate, x_ref=1.0):
    """
    This function calculates the degradation (loss) of a battery cell over time, using a nonlinear model.

    The model is based on the following equation (ref [1]):

    .. math::
        1 - \\alpha_{SEI} * \\exp(-\\frac{x}{x_{ref}} * \\beta_{SEI} * \\text{degradation_rate}) -
        (1 - \\alpha_{SEI}) * \\exp(-\\frac{x}{x_{ref}} * \\text{degradation_rate})

    The authors assume that a certain portion of the battery’s active material is
    consumed in the transient stage to form the SEI film. The formation rate is inversely proportional
    to the SEI film already formed, and stops when a stable SEI has been formed (steady-state stage)

    The authors also assume that the linearized rate during SEI formation differ from the linearized
    rate during the reminding battery life by a constant factor.

    References:
        [1] B. Xu et al., “Modelling of lithium-ion battery degradation for cell life assessment”
        IEEE transactions on smart grid, vol. 9, no. 2, pp. 1131–1140, 2018.


    Args:
        x (float): time or cycle number
        alpha_sei (float): the portion of the charge capacity irreversibly consumed during the SEI film formation
        beta_sei (float): proportionality factor between the degradation rate during SEI formation and
           the degradation rate during the reminding battery life
        degradation_rate (float): the time- or cycle-dependent linear degradation rate
        x_ref (float): reference time or cycle number (default is 1.0)

    Returns:
        float: the degradation of the battery cell
    """

    return (
        1
        - alpha_sei * np.exp((-x / x_ref) * beta_sei * degradation_rate)
        - (1 - alpha_sei) * np.exp((-x / x_ref) * degradation_rate)
    )


# TODO: refactor the degradation methods to use the nonlinear_degradation_model function instead of this one:
def nonlinear_total_degradation_model(alpha_sei, beta_sei, linear_degradation):
    return (
        1
        - alpha_sei * np.exp(-beta_sei * linear_degradation)
        - (1 - alpha_sei) * np.exp(-linear_degradation)
    )


# TODO: refactor the fitting methods to use the nonlinear_degradation_model function instead of these two:
def nonlinear_cal_model(t, alpha_sei, beta_sei, deg_per_time_unit, x_ref=86_400):
    # used in the fitting procedures
    # (you must refactor fitting methods before replacing this with nonlinear_degradation_model)
    return nonlinear_degradation_model(t, alpha_sei, beta_sei, deg_per_time_unit, x_ref)


def nonlinear_cycle_model(N, alpha_sei, beta_sei, deg_per_cyc, x_ref=1.0):
    # used in the fitting procedures
    # (you must refactor fitting methods before replacing this with nonlinear_degradation_model)
    return nonlinear_degradation_model(N, alpha_sei, beta_sei, deg_per_cyc, x_ref)


# --------- Convenience functions -----------------------------------


def _exponential_non_zero(x, x_ref, factor):
    # convenience function for the exponential functions to prevent
    # needing to calculate the exponential of zero (might cause numpy problems)
    non_zero_mask = x != x_ref
    v = np.ones_like(x, dtype=np.float64)
    v[non_zero_mask] = np.exp(factor[non_zero_mask])
    return v


def _exponential(x, x_ref, factor):
    # convenience function for the exponential functions to prevent
    # exposing the exponential to non-floats and zero values
    non_zero_mask = x != x_ref
    # dtype must be forced to float: integer input (e.g. temperatures in whole
    # kelvin) would otherwise truncate the stress factors to whole numbers
    v = np.ones_like(x, dtype=np.float64)
    factor = np.float64(factor)
    v[non_zero_mask] = np.exp(factor[non_zero_mask])
    return v


# --------- Stress Factor Models -----------------------------------


# TODO: check with jinsung code if all models are implemented
# utility functions for extracting information:
# TODO: refactor this (used in fitting method _find_linear_loss to use the nonlinear_degradation_model function):
def linear_loss_residuals(x, cycle_number, sei_alpha, sei_beta, loss, reference_cycle):
    exp_arg_1 = -cycle_number / reference_cycle * sei_beta * x
    exp_1 = np.exp(exp_arg_1)
    exp_arg_2 = -cycle_number / reference_cycle * x
    exp_2 = np.exp(exp_arg_2)
    return loss - 1 + sei_alpha * exp_1 + (1 - sei_alpha) * exp_2


def residuals_solve_cycle_num(x, alpha_sei, beta_sei, deg_per_cyc, full_deg, x_ref=1.0):
    return full_deg - nonlinear_cycle_model(
        x, alpha_sei, beta_sei, deg_per_cyc, x_ref=x_ref
    )


# main stress factor models:
def linear_soc_k_relation(x, k_1, k_2, x_ref=0.5):
    """
    Linear SOC stress factor model.

    Args:
        x (float): The SOC value.
        k_1 (float): The rate constant.
        k_2 (float): The offset.
        x_ref (float): The reference SOC value (default is 0.5 fraction of the total capacity).

    Returns:
        float: The stress factor value.
    """
    return k_1 * (x - x_ref) + k_2


def exponential_soc_k_relation(x, k, x_ref=0.5):
    """
    Exponential SOC stress factor model.

    Args:
        x (float): The SOC value.
        k (float): The rate constant.
        x_ref (float): The reference SOC value (default is 0.5 fraction of the total capacity).

    Returns:
        float: The stress factor value.
    """
    factor = k * (x - x_ref)
    return _exponential(x, x_ref, factor)


def exponential_high_soc_k_relation(x, k, x_ref=0.85):
    """
    Exponential high SOC stress factor model.

    Args:
        x (float): The SOC value.
        k (float): The rate constant.
        x_ref (float): The reference SOC value (default is 0.85 fraction of the total capacity).

    Returns:
        float: The stress factor value.
    """
    factor = k * (x - x_ref)
    out = _exponential(x, x_ref, factor)
    out = np.where(x < x_ref, 1.0, out)
    return out


def exponential_temperature_k_relation(x, k, x_ref=298.0):
    """
    Exponential temperature stress factor model.

    Args:
        x (float): The temperature value.
        k (float): The rate constant.
        x_ref (float): The reference temperature value (default is 298.0 K).

    Returns:
        float: The stress factor value.
    """
    factor = k * (x - x_ref) * (x_ref / x)
    return _exponential(x, x_ref, factor)


def linear_time_k_relation(x, k_1, k_2, x_ref=1.0):
    """
    Linear time stress factor model.

    Args:
        x (float): The time value.
        k_1 (float): The rate constant.
        k_2 (float): The offset.
        x_ref (float): The reference time value (default is 1.0 second).

    Returns:
        float: The stress factor value.
    """
    return k_1 * x / x_ref + k_2


def proportional_time_k_relation(x, k_1, k_2=0.0, x_ref=1.0):
    """
    Proportional time stress factor model.

    Args:
        x (float): The time value.
        k_1 (float): The rate constant.
        k_2 (float): The offset. Not used.
        x_ref (float): The reference time value (default is 1.0 second).

    Returns:
        float: The stress factor value.
    """
    return k_1 * x / x_ref


def constant_time_k_relation(x, k_1, k_2=0.0, x_ref=1.0):
    """
    Constant time stress factor model.

    Args:
        x (float): The time value. Not used.
        k_1 (float): The constant stress factor value.
        k_2 (float): The offset. Not used.
        x_ref (float): The reference time value (default is 1.0 second). Not used.

    Returns:
        float: The stress factor value.
    """

    return k_1


def exponential_dod_k_relation(x, k_1, k_2, k_3=0.0, x_ref=1.0):
    """
    Exponential DOD stress factor model.

    Args:
        x (float): The DOD value.
        k_1 (float): The rate constant.
        k_2 (float): The rate constant.
        k_3 (float): The offset.
        x_ref (float): The reference DOD value (default is 1.0 in fraction of the total capacity).

    Returns:
        float: The stress factor value.
    """
    return k_1 * x / x_ref * np.exp(k_2 * x) + k_3


def empirical_dod_k_relation(x, k_1, k_2, k_3, x_ref=1.0):
    """
    Empirical DOD stress factor model.

    Args:
        x (float): The DOD value.
        k_1 (float): The rate constant.
        k_2 (float): The rate constant.
        k_3 (float): The offset.
        x_ref (float): The reference DOD value (default is 1.0 in fraction of the total capacity).

    Returns:
        float: The stress factor value.
    """
    return np.divide(1, k_1 * np.power(x / x_ref, k_2) + k_3)


def quadratic_dod_k_relation(x, k_1, k_2, x_ref=1.0):
    """
    Quadratic (power-law) DOD stress factor model.

    This is the form Xu et al. (2018) use for NMC cells (their eq. 30):
    ``S_delta(delta) = k_1 * delta ** k_2``. Like the empirical and
    exponential DoD forms it returns a per-cycle rate magnitude (it is *not*
    normalized to 1 at any reference DoD).

    Args:
        x (float): The DOD value.
        k_1 (float): The rate constant.
        k_2 (float): The exponent.
        x_ref (float): The reference DOD value (default is 1.0 in fraction of the total capacity).

    Returns:
        float: The stress factor value.
    """
    return k_1 * np.power(x / x_ref, k_2)
