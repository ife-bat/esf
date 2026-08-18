"""
This module contains classes for fitting data to obtain the stress factors and the "SEI" factors.

The non-linear part ("SEI" factors) is first fitted to the data, and then the linear contribution for
the dataset is then extracted.
After that, the secondary fitting is performed to extract the stress factors parameters
(the linear part is a product of the different stress factors). To be able to extract the model parameters
for a given stress factor, either the experimental data need to be obtained at a set of reference values fixing all
the other stress factors, or the linear contribution must be scaled by the other stress factors.

Typically, the stress factors for SoC and temperature are obtained from setting a reference value
(e.g. SoC=0.5, T=298K), while the other ones are obtained by rescaling the linear
contribution by the other stress factors.

Model:

    (cycling) L = 1 - alpha_sei * exp(-N * beta_sei * deg_per_cyc) - (1 - alpha_sei) * exp(-N * deg_per_cyc)

    (calendaring)
    L = 1 - alpha_sei * exp(-t * beta_sei * deg_per_time_unit) - (1 - alpha_sei) * exp(-t * deg_per_time_unit)

    with:

        L is the degradation (or loss). L = 1 - SoH. So L = 0 for a new battery.

        alpha_sei is generally between 2% and 16%
        beta_sei must be > 1, (SEI film formation occurs at the beginning of the cells life).

    For calendar experiments we assume:

        deg_per_time_unit = time_deg_model(t) * soc_stress_model(SoC) * temp_stress_model(T)

    It is also possible to split for example the soc_stress_model into two parts (or more):

        deg_per_time_unit = time_deg_model(t) * soc1_stress_model(SoC) * soc2_stress_model(SoC) * temp_stress_model(T)

    For cycling data, in addition to the calendar aging, we also need to consider the additional
    degradation due to the actual cycling:

        deg_per_cyc = time_deg_model(N) * soc_stress_model(SoC) * temp_stress_model(T) * cycle_stress_model(N)

    Here, the cycle_stress_model(N) is typically a function of the depth of discharge (DoD) and the cycle number (N).

    Stress factor models:

        temp_stress_model(T) = exp(k_T*(T-T_ref))

            considering 2 points at (T_ref, SoC_B) and (T_A, SoC_B):
            deg_per_time_unit(point_A) / deg_per_time_unit(point_ref) = temp_stress_model(T_A)
                                                                      = exp(k_T*(T_A-T_ref))
            because: temp_stress_model(T_ref) = 1

            =>  k_T = ln(deg_per_time_unit(point_A) / deg_per_time_unit(point_ref)) / (T_A-T_ref)

        soc_stress_model(SoC) = 1 + k_soc * (SoC - SoC_ref)


        time_deg_model(t) = k_t * t

            => k_t = calendar_deg / (t  * soc_stress_model(SoC) * temp_stress_model(T))
                  = deg_per_time_unit / (soc_stress_model(SoC) * temp_stress_model(T))

                    where t_e is the duration of calendar experiments


        dod_stress_model(DoD)

            For LMO and NMC batteries:

                N_cycles_at_80_SoH = 1 / (k_d1 * DoD^k_d2 + k_d3)

            For LFP batteries:

                N_cycles_at_80_SoH = k_d1 * DoD * e^(k_d2 * DoD)

"""

# Note for developers:
#   PrimaryFit uses dictionaries for storing objects so be careful when renaming for example prms items to make sure
#   that the change is propagated also to the fit-functions etc.

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import uncertainties
from lmfit import Model
from lmfit.model import ModelResult
from scipy.optimize import fsolve

import esf.models.base_models as models
from esf.io.data import (
    SampleData,
)
from esf.models import fit_plotting
from esf.settings.parameters import (
    CYCLE_UNIT,
    DOD_UNIT,
    L_UNIT,
    Q_,
    RATE_UNIT,
    SF_COL,
    SOC_UNIT,
    SOH_UNIT,
    TEMPERATURE_UNIT,
    TIME_UNIT,
    Columns,
    DataType,
    ESFParams,
    Regime,
)
from esf.utils.converters import to_numpy_float64

logger = logging.getLogger(__name__)

SINGLE = "single"
RESTRICT_Y_AXIS_FOR_MODEL_RESULT_PLOT = True
DEVELOPMENT_MODE = True
DEG_RATE_NAME = "deg_rate"

def is_cycling_regime(regime) -> bool:
    """True if ``regime`` denotes cycling degradation.

    Accepts a :class:`~esf.settings.parameters.Regime` enum or a string; only
    the leading ``cycling`` / ``calend`` token is significant (e.g.
    ``"cycling_vs_temperature"`` counts as cycling).
    """
    return str(regime).startswith("cycling")


def is_calendar_regime(regime) -> bool:
    """True if ``regime`` denotes calendar degradation."""
    return str(regime).startswith("calend")


def coerce_regime(regime):
    """Normalize a regime to a :class:`~esf.settings.parameters.Regime` enum.

    Accepts a ``Regime`` or a string; only the leading ``cycling`` / ``calend``
    / ``op`` token is significant, so decorated strings like
    ``"cycling_vs_temperature"`` map to ``Regime.CYCLING``. ``None`` passes
    through (a fit may be constructed before its regime is known).
    """
    if regime is None or isinstance(regime, Regime):
        return regime
    if is_cycling_regime(regime):
        return Regime.CYCLING
    if is_calendar_regime(regime):
        return Regime.CALENDAR
    if str(regime).startswith("op"):
        return Regime.OPERATIONAL
    raise ValueError(f"unrecognized regime: {regime!r}")


# Columns the non-reference DoD fit needs on every data point so it can strip
# the temperature, SoC and calendar-time stress factors out of the measured
# cycle life (Xu et al. 2018, eqs. 20/31; decision 5).
DOD_NON_REFERENCE_COLUMNS = ("T", "SoC", "t_cycle")

_DOD_NON_REFERENCE_MISSING_COLUMNS = (
    "The non-reference DoD stress-factor fit removes the temperature, SoC and "
    "calendar-time stress factors from the measured cycle life (Xu et al. 2018 "
    "eqs. 20/31), so every data point must carry temperature 'T' (K), mean "
    "state of charge 'SoC' (fraction) and cycle duration 't_cycle' (s). "
    "Missing column(s): {missing}. Provide them, or fit at reference "
    "conditions with is_at_reference=True."
)

class FitResult:
    """
    This class is used to store the results of the fits.
    """

    def __init__(self, with_uncertainty=True):
        self.fit_objects = {}
        self.fit_object_type = None
        self.with_uncertainty = with_uncertainty

    def __str__(self):
        if not self.fit_objects:
            return "No fits"
        return f"Fits: {self.fit_objects}"

    def __repr__(self):
        if not self.fit_objects:
            return "No fits"
        return f"Fits: {self.fit_objects}"

    def add_fit(self, fit_object, key=SINGLE):
        self.fit_objects[key] = fit_object

    def get_fit(self, key=SINGLE, default=None):
        values = self.fit_objects.keys()
        if not values:
            raise ValueError("No fit objects")

        return self.fit_objects.get(key, default)

    def get_average_parameter(self, parameter, allow_none=False):
        _v = 0.0
        try:
            for _key, fit_object in self.fit_objects.items():
                if isinstance(fit_object, dict):
                    _v += fit_object[parameter]
            else:
                if not self.with_uncertainty:
                    # only picking the values, not the lmfit parameters for now (losing the standard errors):
                    _v += fit_object.params[parameter].value
                else:
                    _v += fit_object.uvars[parameter]
        except KeyError as err:
            if allow_none:
                return None
            raise ValueError(
                f"Parameter {parameter} not found in fit objects"
            ) from err
        return _v / len(self.fit_objects)

    def get_parameter(self, parameter, key=SINGLE):
        fit_object = self.get_fit(key)
        try:
            if isinstance(fit_object, dict):
                _v = fit_object[parameter]
            else:
                if not self.with_uncertainty:
                    # only picking the values, not the lmfit parameters for now (losing the standard errors):
                    _v = fit_object.params[parameter].value
                else:
                    _v = fit_object.uvars[parameter]
        except KeyError as err:
            raise ValueError(
                f"Parameter {parameter} not found in fit objects"
            ) from err
        return _v


class BaseProcessor:
    """
    Base class for processing data before fitting or calculating parameters.
    """

    colors = ["black", "red", "blue", "green", "orange", "purple", "grey", "brown"]
    markers = ["o", "x", "v", "s", "+", "d", "p", "*"]

    def __init__(
        self,
        data: pd.DataFrame | SampleData,
        prms: ESFParams,
        regime: str | Regime | None = None,
        *args,
        **kwargs,
    ):
        if isinstance(data, SampleData):
            raise ValueError(
                "Handling SampleData is not yet implemented, use pd.DataFrame instead."
            )

        if data.empty:
            raise ValueError("There is no data! Please provide some data.")

        self.data = data
        self.prms = prms
        self.regime = regime
        self._experimental = kwargs.pop("experimental", False)
        self.verbose = kwargs.pop("verbose", False)

        prms_mode = prms.mode
        if prms_mode == "uncertainties":
            self.with_uncertainty = True
        else:
            self.with_uncertainty = False

        logger.info(f"{data.head()=}")
        logger.info(f"{prms=}")

        # setting defaults etc. (TODO: update with Enums):
        self.x_col = kwargs.pop("x_col", "t")
        self.y_col = kwargs.pop("y_col", "L")
        self.y_err_col = kwargs.pop("y_err_col", "y_err")
        self.z_col = kwargs.pop("z_col", "SoC")

        self.modified_x_col = None
        self.modified_y_col = None

        self.z_val = None
        self.z_unit = None
        self.z_name = None

        # for storing the for example maximum cycle number for each data-set:
        self.x_values = []

        self.time_unit = TIME_UNIT
        self.temperature_unit = TEMPERATURE_UNIT
        self.cycle_unit = CYCLE_UNIT
        self.soc_unit = SOC_UNIT
        self.dod_unit = DOD_UNIT
        self.soh_unit = SOH_UNIT
        self.l_unit = L_UNIT
        self.sf_col = SF_COL
        self.rate_unit = RATE_UNIT

        self.split_on_uid = True

        # whether fit() writes its results into prms; set by fit(apply=...)
        self._apply_results = True

    @property
    def regime(self):
        """The fit's regime, always a :class:`~esf.settings.parameters.Regime`
        enum (decorated strings passed in are coerced on assignment)."""
        return self._regime

    @regime.setter
    def regime(self, value):
        self._regime = coerce_regime(value)

    @property
    def is_cycling(self) -> bool:
        """Whether this fit is for cycling (vs calendar) degradation."""
        return self.regime is Regime.CYCLING

    @property
    def is_calendar(self) -> bool:
        """Whether this fit is for calendar (vs cycling) degradation."""
        return self.regime is Regime.CALENDAR

    def get_esf_unit(self, key: str) -> str:
        # TODO: make this a function e.g. get_base_unit(class_name, key)
        if key in ["T", "temperature", "Temperature"]:
            return self.temperature_unit
            # return getattr(self, "temperature_unit")
        elif key in ["t", "time", "Time"]:
            return self.time_unit
        elif key in ["D", "dod", "DoD"]:
            return self.dod_unit
        elif key in ["N", "FEC", "cycle", "Cycle"]:
            return self.cycle_unit
        elif key in ["SOC", "soc", "SoC", "high_soc", "High_SOC"]:
            return self.soc_unit
        elif key in ["L", "l", "degradation", "Degradation"]:
            return self.l_unit
        elif key in ["R", "r", "rate", "Rate", "c-rate", "C-rate"]:
            return self.rate_unit
        else:
            raise ValueError(f"{key=} is not supported")

    def _pick_x(self, d):
        x_col = self.get_x_col()
        return d[x_col].max()

    def get_x_col(self):
        return self.modified_x_col or self.x_col

    def get_max_x_value(self):
        try:
            return self.data[self.get_x_col()].max()
        except KeyError as err:
            raise ValueError(f"Column {self.get_x_col()} not found in data") from err

    def get_y_col(self):
        return self.modified_y_col or self.y_col

    def color(self, number):
        # yes, I know....
        number = number % len(self.colors)
        return self.colors[number]

    def echo(self, text="", *args, **kwargs):
        # every message goes to the debug log; the console only sees them in
        # verbose mode (library code is silent by default)
        if text:
            logger.debug(text)
        if self.verbose is True:
            print(text, *args, **kwargs)

    def _convert_value(self, value, unit_from, unit_to):
        self.echo(f"      converting -> {value=} {unit_from=} {unit_to=}")
        if str(unit_from) == str(unit_to):
            return value
        if str(unit_from) == "dimensionless":
            return value
        if str(unit_to) == "dimensionless":
            return value
        if str(unit_from).lower() in ["frac", "fract.", "fraction"] or str(
            unit_to
        ).lower() in [
            "frac",
            "fract.",
            "fraction",
        ]:
            self.echo("It is not allowed to convert fractions to other units")
            return value
        try:
            original = Q_(value, unit_from)
            converted = original.to(unit_to)
            self.echo(f"      converted -> {converted=}")
            return converted.magnitude
        except Exception as e:
            self.echo(f"Error converting {value} from {unit_from} to {unit_to}: {e}")
            return value


class BaseFit(BaseProcessor):
    """
    Base class for fitting the degradation model to data.

    Parameters:
        data (pd.DataFrame): The data to be fitted. It should contain the columns specified in x_col, y_col, z_col, and y_err_col.
        prms (ESFParams): The parameters of the degradation model.

    Attributes:
        fit_result (PrimaryFit): An object that stores the results of the fits.
        prms (ESFParams): The updated parameters after the fit.
    Usage:
        1. Instantiate the class with the data and parameters.
        2. Call the fit() method to perform the fits and update the parameters.
        3. Access the fit results through the fit_result attribute.
        4. Access the updated parameters through the prms attribute.
    """

    colors = ["black", "red", "blue", "green", "orange", "purple", "grey", "brown"]
    markers = ["o", "x", "v", "s", "+", "d", "p", "*"]

    def __init__(
        self,
        data: pd.DataFrame | SampleData,
        prms: ESFParams,
        model_type: str | None = None,
        model_sub_type: str | None = None,
        model=None,
        parameter_dict=None,
        fit_function=None,
        fit_method=None,
        nan_policy=None,
        fit_function_has_y_err=None,
        fit_independent_var=None,
        re_normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(data, prms, *args, **kwargs)
        if isinstance(data, SampleData):
            raise ValueError(
                "Handling SampleData is not yet implemented, use pd.DataFrame instead."
            )

        if data.empty:
            raise ValueError("There is no data! Please provide some data.")

        self.model_type = model_type
        self.model_sub_type = model_sub_type
        self.re_normalize = re_normalize
        self.parameter_dict = parameter_dict or None

        self.fit_function = fit_function or None
        self.fit_method = fit_method or "leastsq"
        self.nan_policy = nan_policy or "omit"
        self.fit_function_has_y_err = fit_function_has_y_err or True
        self.fit_independent_var: str | None = fit_independent_var or None

        self.fit_result = FitResult(with_uncertainty=self.with_uncertainty)

        # setting the models in order of priority:
        # 1. If the models are provided as arguments, use them.
        # 2. If the models are not provided, check if they are in the prms object and create them.
        # 3. If the models are not in the prms object, create them using the standard methods ().
        self._model = model
        self._esf_model_object = None

        self.default_fit_function = None

        self.split_on_uid = False
        self._stress_values = None
        self._reference_value = None
        self._degradation_rates = None
        self.is_stress_factor_fit = False

    def _pick_z(self, g_val):
        # TODO: check if this is still needed
        # a bit of a hack to pick the z value from the key in the fit_objects dict
        if "uid" in self.data.columns and isinstance(g_val, tuple):
            return g_val[1]
        return

    def get_fit_function(self) -> Callable:
        # using a simple getter method to allow for setting fit function outside the initializer of the subclass
        if self.fit_function is not None:
            return self.fit_function
        return self.default_fit_function

    def get_parameter_dict(self) -> dict:
        # using a simple getter method to allow for setting fit parameters outside the initializer of the subclass
        if self.parameter_dict is not None:
            return self.parameter_dict
        return self.default_parameter_dict()

    def default_parameter_dict(self) -> dict:
        raise NotImplementedError(
            "default_parameter_dict must be implemented in the subclass"
        )

    def get_independent_variable(self):
        x = (
            self.get_x_col()
            if self.fit_independent_var is None
            else self.fit_independent_var
        )
        return x

    def preprocess_degradation_rates(
        self,
        z_values: np.ndarray,
        degradation_rates: np.ndarray,
        reference_value: float,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, float]:
        raise NotImplementedError(
            "preprocess_degradation_rates must be implemented in the subclass"
        )

    @property
    def model(self) -> Model:
        if self._model is None:
            self._model = self.create_model()
        return self._model

    def create_model(self) -> Model:
        fit_function = self.get_fit_function()
        independent_var = self.get_independent_variable()
        model = Model(
            fit_function,
            independent_vars=[independent_var],
        )
        return model

    def _fit(self, data, *args, **kwargs) -> tuple[ModelResult, dict]:
        first_iter = kwargs.pop("first_iter", True)
        parameter_dict = kwargs.pop("parameter_dict", None)
        parameter_overrides = kwargs.pop("parameter_overrides", None)
        self.preprocess_degradation_rates()

        x_col = self.get_x_col()
        y_col = self.get_y_col()
        _data_df = data[[x_col, y_col]].sort_values(by=x_col)
        x = _data_df[x_col]
        y = _data_df[y_col]

        independent_var = self.get_independent_variable()

        model = self.model
        if parameter_dict is None:
            parameter_dict = self.get_parameter_dict()

        title = f"Performing fit with {model.name}"
        if first_iter:
            self.echo()
            self.echo("=" * min(len(title), 80))
            self.echo(title)
            self.echo("=" * min(len(title), 80))

            parameter_dict = self._update_parameter_dict(parameter_dict, kwargs)
            if parameter_overrides:
                parameter_dict = self.apply_parameter_overrides(
                    parameter_dict, parameter_overrides
                )

            self.echo("\nParameters")
            self.echo("-" * min(len(title), 80))

            for k, v in parameter_dict.items():
                self.echo(f" {k}: {v}")
        else:
            self.echo("\nRe-fitting model")
            self.echo("-" * min(len(title), 80))

        params = model.make_params(**parameter_dict)

        fit_kwargs = dict(
            params,
            nan_policy=self.nan_policy,
            method=self.fit_method,
        )

        fit_kwargs[independent_var] = x

        if self.fit_function_has_y_err:
            w = 1.0 / data[self.y_err_col]
            fit_kwargs["weights"] = w
        result = model.fit(
            y,
            params,
            **fit_kwargs,
        )
        self.echo("\nBest values")
        self.echo("-" * min(len(title), 80))
        for k, v in result.best_values.items():
            self.echo(f" {k}: {v}")
        return result, parameter_dict

    def apply_parameter_overrides(self, parameter_dict, overrides, strict=True):
        """Merge user overrides into an lmfit parameter dict.

        Two key forms are accepted:

        - ``"name"`` with a dict value replaces the whole parameter spec, e.g.
          ``{"x_ref": {"value": 1.0, "vary": False}}``
        - ``"name__attribute"`` with a scalar value sets one attribute of an
          existing spec, e.g. ``{"x_ref__value": 1.0}`` or ``{"k__max": 2.0}``

        Args:
            parameter_dict: the lmfit parameter dict to update (mutated).
            overrides: mapping of override keys (see above) to values.
            strict: if True (the explicit ``parameter_overrides`` API), raise
                ``ValueError`` on unknown parameter names or malformed values;
                if False (the legacy ``fit(**kwargs)`` path), log and skip them.

        Returns:
            The updated parameter dict.
        """
        for key, value in overrides.items():
            name, _, attribute = key.partition("__")
            if name not in parameter_dict:
                message = (
                    f"unknown fit parameter {name!r}; "
                    f"valid parameters: {sorted(parameter_dict)}"
                )
                if strict:
                    raise ValueError(message)
                self.echo(f" -> {message}")
                continue
            old = parameter_dict[name]
            if attribute:
                self.echo(f" -> overwriting {name}.{attribute}: {old} -> {value}")
                parameter_dict[name][attribute] = value
            elif isinstance(value, dict):
                self.echo(f" -> overwriting {name}: {old} -> {value}")
                parameter_dict[name] = value
            else:
                message = (
                    f"override for {name!r} must be a dict "
                    f"(got {type(value).__name__}); "
                    f"use '{name}__value' to set a single attribute"
                )
                if strict:
                    raise ValueError(message)
                self.echo(f" -> {message}")
        return parameter_dict

    def _update_parameter_dict(self, parameter_dict, kwargs):
        # legacy lenient path: overrides arrive as raw fit(**kwargs); unknown
        # keys are logged and skipped (they may be unrelated kwargs)
        return self.apply_parameter_overrides(parameter_dict, kwargs, strict=False)

    def normalize_degradation_rates(self, x, degradation_rates, reference_value):
        """Safe normalization of degradation rates to a reference value.

        This method selects the closest value to the reference value in the x array
        and normalizes the degradation rates to this value.
        """

        # Find the reference value in the x array (or select closest if missing):
        difference = np.abs(x - reference_value)
        idx = np.argmin(difference)
        updated_reference_value = x[idx]
        ref_degradation_rate = degradation_rates[idx]

        # Divide by ref_degradation_rate to get the relative degradation rates:
        normalized_degradation_rates = degradation_rates / ref_degradation_rate

        return normalized_degradation_rates, updated_reference_value

    def get_results(self):
        raise NotImplementedError("get_results must be implemented in the subclass")

    def _report_fit_result(self, result, title=None):
        if not self.verbose:
            return
        if isinstance(result, ModelResult):
            self.echo(80 * "-")
            if title is not None:
                self.echo(title)
            self.echo(result.fit_report())
            self.echo(80 * "-")
            result_fig = result.plot(title=title)
            if RESTRICT_Y_AXIS_FOR_MODEL_RESULT_PLOT:
                # Until I have figured out why the error-bars in the plot are so big, I restrict
                # the y-axis to the y-data range to be able to see anything at all:
                try:
                    error, line = result_fig.axes[1].lines
                    data = line.get_ydata()
                    y0 = min(data)
                    y1 = max(data)
                    delta_y = y1 - y0
                    y0 -= delta_y * 0.1
                    y1 += delta_y * 0.1
                    result_fig.axes[1].set_ylim(y0, y1)
                except Exception as e:
                    self.echo(str(e))
        else:
            self.echo(f"Fit result is not a ModelResult: {result}")

    def fit(self, *args, apply=True, **kwargs):
        """Run the fit.

        If ``apply`` is True (default) the fitted values are written into
        ``prms``. With ``apply=False`` the fit runs but leaves ``prms``
        untouched; read the results with ``fitted_parameters()`` /
        ``parameter_uncertainty()``.
        """
        self._apply_results = apply
        if self.is_stress_factor_fit:
            self.fit_stress_factor(*args, **kwargs)
        else:
            self.fit_nonlinear(*args, **kwargs)

    def fit_nonlinear(self, *args, **kwargs):
        pdict = None
        verbose = kwargs.pop("verbose", None)
        if verbose is not None:
            self.verbose = verbose

        def _fit_title(z_val: Any = None) -> str:
            regime = "cyc" if self.is_cycling else "cal"
            _title = f"{self.__class__.__name__}"
            if self._model is not None:
                _title += f" - {self._model.name}({regime})"
            if z_val is not None:
                if isinstance(z_val, float):
                    _title += f" - {self.z_col}={z_val:0.2f}"
                else:
                    _title += f"\n{z_val}"
            return _title

        logger.info("Performing single fit")
        # TODO: figure out what is strange with the error-bars in the lmfit plots
        if self.split_on_uid and "uid" in self.data.columns:
            warnings.warn("Split_on_uid is not properly tested yet")
            for uid, d in self.data.groupby("uid"):
                result, pdict = self._fit(d, *args, parameter_dict=pdict, **kwargs)
                self.fit_result.add_fit(result, uid)
                self._report_fit_result(result, title=_fit_title(uid))
        else:
            result, pdict = self._fit(self.data, *args, parameter_dict=pdict, **kwargs)
            self.fit_result.add_fit(result)
            self._report_fit_result(
                result, title=_fit_title(self.data[self.z_col].mean())
            )
        if self._apply_results:
            self.update_prms()

    def fit_stress_factor(self, *args, **kwargs):
        """
        Fit the stress factor model to the data.

        Args:
            *args: Additional arguments to pass to the fit method.
            **kwargs: Additional keyword arguments to pass to the fit method.

        """

        x_col = self.get_x_col()
        y_col = self.get_y_col()

        _data_df = self.data[[x_col, y_col]].sort_values(by=x_col)

        # Convert the x and y columns to numpy arrays:
        x = _data_df[x_col].values
        degradation_rates = _data_df[y_col].values

        # Some stress factor models needs to be evaluated at a specific reference value.
        # This is the case for both the temperature and the soc stress factor models.
        reference_value = self.reference_value

        if self.verbose:
            print("Fit input data:")
            print(f" {degradation_rates=}\n {reference_value=}\n {x=}")

        # For SoC and temperature stress factor models, the degradation rates are normalized to the reference value.
        # The data should contain the measurement done at the reference value. If not, it finds the closest x-value and picks
        # that degradation rate for normalization. This is done with the normalize_degradation_rates method.
        # For the time stress factor model, the degradation rates of all the other stress factors must be removed and thus we
        # have to calculate them at each x-value (using the simple normalize_degradation_rates method will not work).
        degradation_rates, reference_value = self.preprocess_degradation_rates(
            x, degradation_rates, reference_value
        )

        # Setting these as attributes for later use in simulations etc. Doing it here
        # instead of inside the preprocess method since the user is allowed to override
        # the preprocess method:
        self._degradation_rates = degradation_rates
        self._reference_value = reference_value
        self._stress_values = x

        # Fit the secondary model (Note! Forcing the independent variable to be "x" for the secondary fit):
        if self.verbose:
            print("Fit input data after preprocessing:")
            print(f" {degradation_rates=}\n {reference_value=}\n {x=}")

        result = self._stress_factor_fit(
            degradation_rates,
            reference_value,  # remark that this is the reference value defined in the actual model.
            x,
            *args,
            **kwargs,
        )
        self.fit_result.add_fit(result)
        self._report_fit_result(result, title="Result")
        if self._apply_results:
            self.update_prms()

        return result

    def _stress_factor_fit(
        self,
        normalized_degradation_rates: np.ndarray,
        reference_value: float,
        x_values: np.array,
        *args,
        **kwargs,
    ):
        parameter_overrides = kwargs.pop("parameter_overrides", None)

        model = self.model
        title = f"Performing fit with {model.name}"
        self.echo()
        self.echo("=" * min(len(title), 80))
        self.echo(title)
        self.echo("=" * min(len(title), 80))

        self.echo(f" -> {model=}")
        self.echo(f"Reference value: {reference_value}")

        model_parameter_dict = self.get_parameter_dict()
        # Note! the convertion of the reference value to the correct unit is done in the get_parameter_dict method
        self.echo(f" -> {model_parameter_dict=}")
        model_parameter_dict = self._update_parameter_dict(model_parameter_dict, kwargs)
        if parameter_overrides:
            model_parameter_dict = self.apply_parameter_overrides(
                model_parameter_dict, parameter_overrides
            )
        self.echo(f" -> {model_parameter_dict=}")
        params = model.make_params(**model_parameter_dict)
        self.echo(f" -> {params=}")

        # Note! Forcing the independent variable to be "x" for the secondary fit.
        fit_kwargs = dict(
            params=params,
            nan_policy=self.nan_policy,
            method=self.fit_method,
            x=x_values,
        )
        logger.debug(f"{fit_kwargs=}\n{model_parameter_dict=}\n{model=}\n{params=}")
        result = model.fit(
            normalized_degradation_rates,
            **fit_kwargs,
        )

        self.echo("\nBest values")
        self.echo("-" * min(len(title), 80))
        for k, v in result.best_values.items():
            self.echo(f" {k}: {v}")

        return result

    def _get_stress_factor_at_reference_conditions(self):
        x = self._reference_value
        self.echo(f" -> {x=}")

        independent_var = self.get_independent_variable()
        model_result = self.fit_result.get_fit()
        model = self.model
        sim_kwargs = {independent_var: x}
        ref_degradation_rate = model.eval(params=model_result.params, **sim_kwargs)
        self.echo(f" -> {ref_degradation_rate=}")
        return ref_degradation_rate

    def _simulate(
        self,
        x: np.ndarray | None = None,
        steps=100,
        split_x_on_z=None,
        *args,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        x_col = self.get_x_col()
        y_col = self.sf_col if self.is_stress_factor_fit else self.get_y_col()

        if x is None:
            if split_x_on_z is None:
                # if x is not given, we assume that all the sims are done on the same x values:
                split_x_on_z = False

            x = np.linspace(self.data[x_col].min(), self.data[x_col].max(), steps)

        independent_var = self.get_independent_variable()
        model_result = self.fit_result.get_fit()
        model = self.model
        sim_kwargs = {independent_var: x}
        y_sim = model.eval(params=model_result.params, **sim_kwargs)

        simulation_df = pd.DataFrame({x_col: x, y_col: y_sim})

        return simulation_df

    def simulate(
        self,
        x: np.ndarray | None = None,
        z: np.ndarray | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Evaluate the fitted model on x (defaults to the data range)."""
        if z is not None:
            raise NotImplementedError(
                "simulating on a secondary variable (z) is not supported "
                "for this fit class"
            )
        return self._simulate(*args, x=x, **kwargs)

    def plot_results(self, *, axes: list | None = None, **kwargs):
        """Render the fit result. Delegates to :mod:`esf.models.fit_plotting`."""
        return fit_plotting.plot_single_fit(self, axes=axes, **kwargs)

    def parameter_uncertainty(self):
        """The fitted parameters' covariance, keyed by ESFParams names.

        Returns a :class:`esf.uncertainty.ParameterUncertainty` (Tier 0). Merge
        the results of several fits with ``+`` and feed them to
        :class:`esf.uncertainty.ParameterEnsemble` for Monte Carlo propagation.
        """
        from esf.uncertainty import extract_uncertainty

        return extract_uncertainty(self)


class StressFactorFit(BaseFit):

    def __init__(
        self,
        data,
        prms,
        regime="calend",
        create_from_prms=True,
        model_type="soc",
        x_col="SoC",
        reference_value=None,
        reference_value_unit=None,
        fit_function_has_y_err=False,
        esf_model_unit=None,
        fit_method="leastsq",
        update_degradation_parameter=True,
        *args,
        **kwargs,
    ):
        """StressFactorFit class initializer:

        Arguments:
            data (pd.DataFrame): The data to be fitted.
            prms (ESFParams): The parameters of the degradation model.
            regime (str): The regime of the data.
            create_from_prms (bool): Whether to create the fit from the prms object.
            x_col (str): The column name of the independent variable.
            reference_value (float): The reference value of the independent variable (e.g. if SoC it is typically 0.5).
            reference_value_unit (str): The unit of the reference value (defaults to the unit of x_col).
            esf_model_unit (str): The unit of the esf model. Used to convert the reference value to the unit used in the model.
                It is currently not allowed to implement models with units that differse
                from the ones defined in the parameters.py file (e.g. SOC_UNIT, TEMPERATURE_UNIT, etc.).
            fit_function_has_y_err = False (bool)
        """
        super().__init__(
            data,
            prms,
            *args,
            model_type=model_type,  # might not be needed
            **kwargs,
        )
        self.x_col = x_col
        self.x_unit = self.prms.get_unit(self.x_col)
        self._alternative_reference_unit = reference_value_unit
        self.y_col = DEG_RATE_NAME
        self.regime = regime
        self.fit_method = fit_method
        self._create_from_prms = create_from_prms

        self.is_stress_factor_fit = True
        self.update_degradation_parameter = update_degradation_parameter

        # This should only be set to True if you want to divide the fit parameters
        # by "x_ref" before updating the prms object. Handle with caution:
        self.re_normalize = False

        self.reference_value = reference_value
        self.fit_function_has_y_err = fit_function_has_y_err
        self.esf_model_unit = esf_model_unit
        self.set_mode()
        self.fit_independent_var = self.fit_independent_var_from_prms()

    def fitted_parameters(self) -> dict:
        """The fitted stress-model parameters, keyed by ESFParams attribute name.

        This is a pure read of the fit result: it does **not** modify ``prms``.
        ``update_prms`` writes these values into ``prms``; callers who want the
        results without the side effect use this method instead.
        """
        result = self.fit_result.get_fit()
        if not isinstance(result, ModelResult):
            raise ValueError("Fit result is not a ModelResult object")

        esf_model = self._esf_model_object
        if esf_model is None:
            return {}

        esf_params = esf_model.esf_parameters[1:]
        params = esf_model.original_parameters[1:]

        # the fit works in x_ref-normalized space; divide back out when the
        # model carries an x_ref parameter and re-normalization is requested
        if "x_ref" in params and self.re_normalize:
            ref = self.fit_result.get_parameter("x_ref")
        else:
            ref = 1.0

        return {
            ep: self.fit_result.get_parameter(p) / ref
            for ep, p in zip(esf_params, params, strict=False)
        }

    def update_prms(self, *args, **kwargs):

        result = self.fit_result.get_fit()
        if not isinstance(result, ModelResult):
            raise ValueError("Fit result is not a ModelResult object")

        # if the secondary model is created from the prms object, we can
        # update the prms object directly. If not, either the subclass needs
        # to override this method or the user needs to update the prms object manually.
        fitted = self.fitted_parameters()
        if not fitted:
            logger.debug("No secondary esf model object")
            warnings.warn(
                "You do not have a secondary esf model object (ModelItem). You will have to override the "
                "update_prms method in the sub-class directly if you want to transfer the fit parameters"
                "automatically to the ESFParams object."
            )
            self.echo(result.fit_report())
            return

        title = "Updating prms with fit results"
        self.echo()
        self.echo("=" * min(len(title), 80))
        self.echo(title)
        self.echo("=" * min(len(title), 80))

        for ep, value in fitted.items():
            try:
                logger.info(f" -> {ep} = {value}")
                self.echo(f" -> {ep} = {value}")
                self.prms.set(ep, value)
            except Exception as e:
                logger.error(f"Error updating {ep=}: {e}")
                self.echo(f"Error updating {ep=}: {e}")
                self.echo(f"{self.fit_result=}")

        # also updating the soc_deg_rate_at_reference_conditions and temperature_deg_rate_at_reference_conditions
        if self.update_degradation_parameter:
            self.echo("Updating degradation parameters at reference conditions")
            ref_degradation_rate = self._get_stress_factor_at_reference_conditions()
            if self.x_col.lower() == "soc":
                self.prms.soc_deg_rate_at_reference_conditions = ref_degradation_rate
                self.echo(
                    f" -> soc_deg_rate_at_reference_conditions = {ref_degradation_rate}"
                )
            elif self.x_col == "T":
                self.prms.temperature_deg_rate_at_reference_conditions = (
                    ref_degradation_rate
                )
                self.echo(
                    f" -> temperature_deg_rate_at_reference_conditions = {ref_degradation_rate}"
                )
        self.echo("Marking prms as changed")
        self.prms.mark_changed()

    def fit_independent_var_from_prms(self):
        if self._create_from_prms:
            try:
                return self._esf_model_object.independent_var
            except Exception as e:
                self.echo("Failed to get independent variable from prms")
                self.echo(f" -> {self._esf_model_object=}")
                self.echo(str(e))
                self.echo("Returning default independent variable")
        return "x"

    def convert_reference_value(self, value, to_unit=None):
        """
        Convert the reference value in the esf model to the unit provided by the parameters (users units).

        Args:
            value (float): The reference value to convert.
            to_unit (str): The unit to convert the reference value to.

        Returns:
            float: The converted reference value.
        """
        self.echo("Convert reference value")
        if self.esf_model_unit is None:
            raise ValueError("esf_model_unit is not set")
        if to_unit is None:
            if self._alternative_reference_unit is None:
                # converting to the same unit as the x column
                to_unit = self.x_unit
            else:
                # converting to the unit specified by the user / sub-class
                # (e.g. for time stress factor fit, we need to calculate the degradation rates from all the other stress factors)
                to_unit = self._alternative_reference_unit
        self.echo("Going to convert using self._convert_value")
        self.echo(f" -> {value=} {self.esf_model_unit=} {to_unit=}")
        new_value = self._convert_value(
            value, unit_from=self.esf_model_unit, unit_to=to_unit
        )
        self.echo(f" -> {new_value=}")
        return new_value

    def get_parameter_dict(self) -> dict:
        if self.parameter_dict is not None:
            return self.parameter_dict
        if self._create_from_prms:
            try:
                prm_dict = self._esf_model_object.get_parameter_dict()
                # convert the reference value to the unit of the x column
                prm_dict["x_ref"]["value"] = self.convert_reference_value(
                    prm_dict["x_ref"]["value"]
                )
                return prm_dict
            except Exception as e:
                self.echo("Failed to create parameter dict from prms")
                self.echo(str(e))
                self.echo("Creating parameter dict from default method")
        return self.default_parameter_dict()


class SoCSFfit(StressFactorFit):
    """SoC Stress Factor class:

    Arguments:
        data (pd.DataFrame): The data to be fitted. This must be a pd.DataFrame with columns "SoC" and "deg_rate".
        prms (ESFParams): The parameters of the degradation model.
        regime (str): The regime of the data.
        create_from_prms (bool): Whether to create the fit from the prms object.
    """
    def __init__(
        self,
        data,
        prms,
        regime="calend",
        create_from_prms=True,
        update_degradation_parameter=True,
        *args,
        **kwargs,
    ):

        super().__init__(
            data,
            prms,
            *args,
            regime=regime,
            model_type="soc",
            x_col="SoC",
            create_from_prms=create_from_prms,
            reference_value=prms.reference_soc,
            fit_function_has_y_err=False,
            esf_model_unit=SOC_UNIT,
            update_degradation_parameter=update_degradation_parameter,
            **kwargs,
        )

    def set_mode(self):
        if self.is_cycling:
            model_regime = "Cycling"
        elif self.is_calendar:
            model_regime = "Calendar"
        else:
            raise ValueError(f"{self.regime=} is not supported")
        self.default_fit_function = self.prms.stress_model_function(model_regime, "soc")
        self.model_sub_type = self.prms.stress_model_label(model_regime, "soc")
        self._esf_model_object = self.prms.stress_model(model_regime, "soc")

    def preprocess_degradation_rates(
        self, z_values, degradation_rates, reference_value, *args, **kwargs
    ) -> tuple[np.ndarray, float]:
        return self.normalize_degradation_rates(
            z_values, degradation_rates, reference_value
        )

    def default_parameter_dict(self) -> dict:
        return {
            "k": dict(
                value=0.001,
                vary=True,
                min=-0.1,
                max=4.0,
            ),
            "x_ref": dict(
                value=to_numpy_float64(self.prms.reference_soc),
                vary=False,
            ),
        }


class TemperatureSFfit(StressFactorFit):
    """Temperature Stress Factor class:

    Arguments:
        data (pd.DataFrame): The data to be fitted. This must be a pd.DataFrame with columns "T" and "deg_rate".
        prms (ESFParams): The parameters of the degradation model.
        regime (str): The regime of the data.
        create_from_prms (bool): Whether to create the fit from the prms object.
    """

    def __init__(
        self,
        data,
        prms,
        regime="calend",
        create_from_prms=True,
        update_degradation_parameter=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            data,
            prms,
            *args,
            regime=regime,
            model_type="temperature",
            x_col="T",
            create_from_prms=create_from_prms,
            reference_value=prms.reference_temperature,
            fit_function_has_y_err=False,
            esf_model_unit=TEMPERATURE_UNIT,
            update_degradation_parameter=update_degradation_parameter,
            **kwargs,
        )

    def set_mode(self):
        if self.is_cycling:
            model_regime = "Cycling"
        elif self.is_calendar:
            model_regime = "Calendar"
        else:
            raise ValueError(f"{self.regime=} is not supported")
        self.default_fit_function = self.prms.stress_model_function(
            model_regime, "temperature"
        )
        self.model_sub_type = self.prms.stress_model_label(model_regime, "temperature")
        self._esf_model_object = self.prms.stress_model(model_regime, "temperature")

    def default_parameter_dict(self) -> dict:
        return {
            "k": dict(
                value=0.001,
                vary=True,
                min=-0.1,
                max=4.0,
            ),
            "x_ref": dict(
                value=to_numpy_float64(self.prms.reference_temperature),
                vary=False,
            ),
        }

    def preprocess_degradation_rates(
        self, z_values, degradation_rates, reference_value, *args, **kwargs
    ) -> tuple[np.ndarray, float]:
        return self.normalize_degradation_rates(
            z_values, degradation_rates, reference_value
        )


class TimeSFCalc(BaseProcessor):

    def __init__(
        self,
        data,
        prms,
        x_col="SoC",
        y_col=DEG_RATE_NAME,
        regime="calend",
        create_from_prms=True,
        reference_time=None,
        reference_time_unit=None,
        *args,
        **kwargs,
    ):
        if x_col == "SoC":
            reference_value = prms.reference_soc
            reference_value_unit = prms.soc_unit
        elif x_col == "T":
            reference_value = prms.reference_temperature
            reference_value_unit = prms.temperature_unit
        else:
            raise ValueError(f"{x_col=} is not supported")

        super().__init__(
            data,
            prms,
            *args,
            regime=regime,
            x_col=x_col,
            y_col=y_col,
            **kwargs,
        )
        self._create_from_prms = create_from_prms
        self.reference_value = reference_value
        self.reference_value_unit = reference_value_unit
        self.reference_time = reference_time or prms.reference_calendar_time
        self.reference_time_unit = reference_time_unit or prms.time_unit
        self.calculated_k_time = None
        self.echo(self.__str__())

    def __str__(self):
        txt = f"TimeSFCalc(\n\tregime={self.regime}, x_col={self.x_col}, y_col={self.y_col}, \n"
        txt += f"\treference_value={self.reference_value}, reference_value_unit={self.reference_value_unit}\n"
        if self.prms.mode == "uncertainties":
            txt += f"\tmode={self.prms.mode}\n"
        txt += f"\treference_time={self.reference_time} {self.reference_time_unit}\n"
        txt += f"\tcalculated_k_time={self.calculated_k_time}\n"
        return txt

    def calc(self, apply=True):
        """
        Calculate the relasionship

        k_time = F / (t * S_temperature * S_SoC * S_something_else)

        The "x" values are either the temperature or the SoC, and the degradation rate is rescaled
        by the other stress factor(s).

        k_time = (F / (t * S_not_x)) / S_x = (F / S_not_x) / (S_x * t)

        With ``apply=False`` the result is available as ``self.calculated_k_time``
        but is not written into ``prms``.
        """
        x_model_simulated_values, normalized_degradation_rates = (
            self.preprocess_degradation_rates(
                x_values=self.data[self.x_col].values,
                degradation_rates=self.data[self.y_col].values,
                x_reference_value=self.reference_value,
            )
        )
        k_times = normalized_degradation_rates / (
            self.reference_time * x_model_simulated_values
        )

        # TODO: find a better way to handle case where we have uncertainties.
        k_times = np.mean(k_times)
        k_times_stdev = np.std(k_times)
        self.echo("--------------------------------")
        self.echo(f"{k_times=} {k_times_stdev=}")
        self.echo("--------------------------------")

        if self.prms.mode == "uncertainties":
            k_times = uncertainties.ufloat(k_times, k_times_stdev)
        self.calculated_k_time = k_times
        if not (self.is_calendar or self.is_cycling):
            raise ValueError(f"{self.regime=} is not supported")
        if not apply:
            return
        if self.is_calendar:
            self.prms.k_1_time_calendar = k_times
        else:
            self.prms.k_1_time_cycling = k_times
        self.prms.mark_changed()

    def preprocess_degradation_rates(
        self, x_values, degradation_rates, x_reference_value, **kwargs
    ):
        """Rescale the degradation rates so that other contributions are removed.

        Currently, we only support data of the form degradation_rates vs SoC and
        degradation_rates vs temperature. The only time stress factor model that can be used
        is the proportional model.

        """

        self.echo("\n>>> Preprocessing degradation rates")
        self.echo(f"{x_values=}")
        self.echo(f"{degradation_rates=}")
        self.echo(f"{x_reference_value=}")
        self.echo(f"{self.reference_time=}")

        if self.is_cycling:
            warnings.warn("It is not recommended to use this method for cycling data!")
            regime = "Cycling"
        elif self.is_calendar:
            regime = "Calendar"
        else:
            raise ValueError(f"{self.regime=} is not supported")

        if not self._create_from_prms:
            raise ValueError("Not creating model from prms, using default parameters")

        self.echo("\n>>> Creating model from prms")

        # make sure that the time stress function is the proportional model
        t_model_info = self._get_models_info(regime, ["time"])[0]
        t_stress_function = t_model_info["func"]
        if t_stress_function.__name__ != "proportional_time_k_relation":
            raise ValueError("The time stress function must be the proportional model")

        # TODO: move this to its own method
        model_set = self.prms.get_model_set()
        model_sub_set = model_set[regime]
        if "time" not in model_sub_set:
            raise ValueError(f"time is not in {regime} {model_sub_set=}")

        # get the model for the x values
        if self.x_col == "SoC":
            x_model_name = "soc"
        elif self.x_col == "T":
            x_model_name = "temperature"
        else:
            raise ValueError(f"{self.x_col=} is not supported")

        x_model_info = self._get_models_info(regime, [x_model_name])[0]
        x_stress_function = x_model_info["func"]
        x_stress_values = x_model_info["values"]
        x_p_values = x_stress_values[:-1]
        x_p_ref = x_stress_values[-1]

        x_model_simulated_values = x_stress_function(
            x_values, *x_p_values, x_ref=x_p_ref
        )

        normalized_t_value = self.reference_time

        self.echo(f"{x_model_simulated_values=}")
        self.echo(f"{normalized_t_value=}")

        scaling_models = [m for m in model_sub_set if m != "time" and m != x_model_name]
        scaling_models_info = self._get_models_info(regime, scaling_models)

        stress_factor = 1.0
        for model_info in scaling_models_info:
            stress_function = model_info["func"]
            stress_values = model_info["values"]
            p_values = stress_values[:-1]
            p_ref = stress_values[-1]

            individual_stress_factor = stress_function(
                p_ref,
                *p_values,
                x_ref=p_ref,
            )
            stress_factor = stress_factor * individual_stress_factor
        normalized_degradation_rates = degradation_rates / stress_factor

        return (
            x_model_simulated_values,
            normalized_degradation_rates,
        )

    def _get_time_stress_model_info(self, function_type: str = "Proportional"):
        # not used, but kept in case we will allow for other time stress models.
        self.echo("--------------------------------")
        self.echo(" Getting time stress model info")
        self.echo("--------------------------------")

        if self.is_cycling:
            regime = "Cycling"
        elif self.is_calendar:
            regime = "Calendar"
        else:
            raise ValueError(f"{self.regime=} is not supported")

        model_function = self.prms.get_model_function(regime, "time", function_type)

        parameter_dict = self.prms.get_model_parameter_dict(
            regime, "time", function_type
        )

        esf_parameters = self.prms.get_model_esf_parameters(
            regime, "time", function_type
        )
        esf_parameters_values = self.prms.get_all(esf_parameters)
        self.echo(f"{model_function=}")
        self.echo(f"{parameter_dict=}")
        self.echo(f"{esf_parameters_values=}")

        return model_function, parameter_dict, esf_parameters_values

    def _get_models_info(self, regime: str, models: list):
        models_info = []
        for factor in models:
            model_info = dict(
                name=f"{factor}_{regime.lower()}_model",
                stress_factor=factor,
                func=self.prms.stress_model_function(regime, factor),
                values=self.prms.stress_model_parameter_values(regime, factor),
                prms_unit=self.prms.get_unit(factor),
                esf_unit=self.get_esf_unit(factor),
            )
            models_info.append(model_info)
        return models_info


class DoDSFfit(StressFactorFit):
    """Fit the DoD stress factor to cycling data.


    To find the DoD stress factor, we need "cycle life before reaching EOL" (typically 80% SoH). In the paper,
    the plot showing an example of this (fig 3c) it is written "Cycle life before reaching 80% EoL at reference conditions".

    Columns needed in the data:
    - N: number of cycles before reaching EOL
    - DoD: depth of discharge

    Additional information needed for fitting at non-reference conditions:
    - t: time (either as a column or with enough information so that it can be calculated)

    There are two ways to find the parameters for the DoD stress factor. One option is to ase the reference conditions
    and find the degradation rate at reference conditions.
    A more general method is to use the temperature, SoC, and time stress model to eliminate all other stress factors
    from the degradation data. Given the temperature, SoC, and time for one cycle (condition A), we can calculate the
    DoD stress factor as:

    S_dod = (f_d_1[A] / (S_T[A] * S_soc[A])) - S_t[t_p, A]

    where:
        f_d_1[A] is the degradation rate at condition A
        S_T[A] is the temperature stress factor at condition A
        S_soc[A] is the SoC stress factor at condition A
        S_t[t_p, A] is the time stress factor at condition A

        t_p is the time at the point of the DoD stress factor calculation

    We can then use these DoD stress values to fit the DoD stress function and find the DoD stress factor parameters.


    """

    def __init__(
        self,
        data,
        prms,
        regime="cycling",
        create_from_prms=True,
        is_at_reference=True,
        update_degradation_parameter=False,
        *args,
        **kwargs,
    ):
        self.is_at_reference = is_at_reference
        super().__init__(
            data,
            prms,
            *args,
            regime=regime,
            model_type="dod",
            x_col="DoD",
            create_from_prms=create_from_prms,
            reference_value=prms.reference_dod,
            fit_function_has_y_err=False,
            esf_model_unit=DOD_UNIT,
            update_degradation_parameter=update_degradation_parameter,
            **kwargs,
        )

    def set_mode(self):
        if not self.is_cycling:
            raise ValueError("DoD stress factor fit is only supported for cycling data")
        self.x_col = "DoD"
        # the raw measured column is cycle life N; preprocess() converts it to a
        # per-cycle rate S_dod before the fit sees it
        self.y_col = "N"
        self.default_fit_function = self.prms.stress_model_function("Cycling", "dod")
        self.model_sub_type = self.prms.stress_model_label("Cycling", "dod")
        self._esf_model_object = self.prms.stress_model("Cycling", "dod")

    def _stress_factor(self, regime: str, factor: str, x: np.ndarray) -> np.ndarray:
        """Evaluate a fitted stress-factor model at the given operating points.

        Used by the non-reference path to reconstruct S_T, S_soc and S_t from
        the parameters already stored in ``prms``.
        """
        func = self.prms.stress_model_function(regime, factor)
        values = self.prms.stress_model_parameter_values(regime, factor)
        return func(np.asarray(x, dtype=np.float64), *values)

    def preprocess_degradation_rates(
        self, x_values, degradation_rates, reference_value, *args, **kwargs
    ) -> tuple[np.ndarray, float]:
        """Turn cycle-life data into the per-cycle DoD stress factor.

        Reference-conditions path (``is_at_reference=True``): every other
        stress factor is 1, so the DoD model output *is* the per-cycle
        degradation rate (decision 1). The linear inversion of cycle life N to
        that rate (decision 2) is::

            S_dod(delta) = full_degradation_level / N(delta)

        with the EOL loss = full_degradation_level = 0.2 (decision 3). The
        reference value is not used for normalization (the DoD factor is not
        anchored to 1 at any reference DoD, decision 4); it is returned
        unchanged only to satisfy the pipeline contract.

        Non-reference path (``is_at_reference=False``): the cycles were run at
        assorted temperatures, SoCs and durations, so the temperature, SoC and
        calendar-time stress factors are stripped out first (Xu et al. 2018,
        eqs. 20/31; decision 5)::

            f_d,1 = full_degradation_level / N
            S_dod(delta) = f_d,1 / (S_T(T) * S_soc(SoC)) - S_t(t_cycle)

        S_T and S_soc are the *cycling* temperature and SoC models and S_t is
        the *calendar* time model, all read from ``prms`` -- so those stages
        must be fitted before this one. The additive time term matches the
        paper's split: calendar aging over the cycle's wall-clock duration is
        subtracted, while temperature and SoC scale multiplicatively. The data
        must carry the columns in ``DOD_NON_REFERENCE_COLUMNS``.
        """
        cycle_life = np.asarray(degradation_rates, dtype=np.float64)
        f_d1 = self.prms.full_degradation_level / cycle_life
        if self.is_at_reference:
            return f_d1, reference_value

        missing = [c for c in DOD_NON_REFERENCE_COLUMNS if c not in self.data.columns]
        if missing:
            raise ValueError(
                _DOD_NON_REFERENCE_MISSING_COLUMNS.format(missing=missing)
            )
        # x_values is the DoD column sorted ascending; align the auxiliary
        # conditions to the same order so they match f_d1 point for point.
        aligned = self.data.sort_values(by=self.x_col)
        s_temperature = self._stress_factor("Cycling", "temperature", aligned["T"])
        s_soc = self._stress_factor("Cycling", "soc", aligned["SoC"])
        s_time = self._stress_factor("Calendar", "time", aligned["t_cycle"])
        stress_factor = f_d1 / (s_temperature * s_soc) - s_time
        return stress_factor, reference_value

    def default_parameter_dict(self):
        # only reached if create_from_prms is False; mirror the registry defaults
        return self._esf_model_object.get_parameter_dict()


class NonlinearFit(BaseFit):
    """
    Fit the Non-linear model (aka. the SEI model) to cycling data to extract alpha_sei and beta_sei
    or to find the degradation rates (deg_per_cyc or deg_per_time_unit).

    Best practice:
    --------------
    1) Find the SEI parameters for the reference conditions.
    2) Fix the SEI parameters and find the degradation rates for the other conditions.

    Parameters
    ----------
    data : pd.DataFrame
        Cycling data with columns "N", "L", "SoC" (only one data-set allowed)
    prms : ESFParams
        Parameters of the degradation model
    """

    def __init__(
        self,
        data,
        prms,
        z_col="T",
        z_val=None,
        regime="calend",
        freeze_sei_parameters_for_cycling_data=True,
        update_degradation_parameter=False,
        is_reference=False,
        calculate_n_at_eol=False,
        split_on_uid=False,
        *args,
        **kwargs,
    ):
        super().__init__(data, prms, *args, model_type="sei", **kwargs)
        self.y_col = "L"
        self.z_col = z_col
        self.z_val = z_val
        self.z_unit = prms.get_unit(z_col)
        self.regime = regime
        self.is_reference = is_reference
        self.update_degradation_parameter = update_degradation_parameter
        self.calculate_n_at_eol = calculate_n_at_eol
        self.split_on_uid = split_on_uid

        self.single_fit = True
        self.fit_method = "leastsq"

        self.freeze_sei_parameters_for_cycling_data = (
            freeze_sei_parameters_for_cycling_data
        )
        self.set_mode()

    def set_mode(self):
        if self.is_cycling:
            self.x_col = "N"
            self.default_fit_function = models.nonlinear_cycle_model

        elif self.is_calendar:
            self.x_col = "t"
            self.default_fit_function = models.nonlinear_cal_model
        else:
            raise ValueError(f"{self.regime=} is not supported")

    def default_parameter_dict(self):
        if self.parameter_dict is not None:
            return self.parameter_dict
        _vary_sei = self.is_reference
        _vary_x_ref = False  # probably never needed
        _vary_degradation_parameter = True
        _degradation_parameter_value = 0.00001
        _degradation_parameter_limits = (0, 1)

        _degradation_parameter = (
            "deg_per_cyc" if self.is_cycling else "deg_per_time_unit"
        )
        _reference_x = (
            self.prms.reference_cycle
            if self.is_cycling
            else self.prms.reference_calendar_time
        )

        # alpha_sei is the fraction of capacity consumed during SEI formation
        # and beta_sei a positive rate ratio; leaving them unbounded lets the
        # optimizer wander into exp-overflow territory (lmfit then dies with
        # "The array returned by a function changed size between calls") or
        # converge to unphysical, degenerate solutions.
        parameter_dict = {
            "alpha_sei": dict(
                value=to_numpy_float64(self.prms.sei_alpha),
                vary=_vary_sei,
                min=0.0,
                max=1.0,
            ),
            "beta_sei": dict(
                value=to_numpy_float64(self.prms.sei_beta),
                vary=_vary_sei,
                min=0.0,
                max=1.0e4,
            ),
            _degradation_parameter: dict(
                value=_degradation_parameter_value,
                min=_degradation_parameter_limits[0],
                max=_degradation_parameter_limits[1],
                vary=_vary_degradation_parameter,
            ),
            "x_ref": dict(
                value=to_numpy_float64(_reference_x),
                vary=_vary_x_ref,
            ),
        }

        return parameter_dict

    def preprocess_degradation_rates(self, *args, **kwargs):
        if self.is_reference:
            if "SoC" not in self.data.columns:
                warnings.warn(
                    "No SoC column found in the data. Assuming reference conditions are met."
                )
            elif "T" not in self.data.columns:
                warnings.warn(
                    "No T column found in the data. Assuming reference conditions are met."
                )
            else:
                self.echo("Checking the reference conditions")
                reference_soc = self.data["SoC"].mean()
                if not np.isclose(reference_soc, self.prms.reference_soc):
                    warnings.warn(
                        f"Reference SoC is not {self.prms.reference_soc}: {reference_soc=}. This is unusual."
                    )
                reference_temperature = self.data["T"].mean()
                if not np.isclose(
                    reference_temperature, self.prms.reference_temperature
                ):
                    warnings.warn(
                        f"Reference temperature is not {self.prms.reference_temperature}: {reference_temperature=}. This is unusual."
                    )

    @property
    def _lmfit_rate_name(self) -> str:
        # the degradation-rate parameter name used by the lmfit model
        return "deg_per_cyc" if self.is_cycling else "deg_per_time_unit"

    @property
    def _prms_rate_name(self) -> str:
        # the corresponding ESFParams field name (note: the cycling field is
        # ``deg_per_cycle``, not the lmfit ``deg_per_cyc``)
        return "deg_per_cycle" if self.is_cycling else "deg_per_time_unit"

    def fitted_parameters(self) -> dict:
        """The fitted SEI values, keyed by ESFParams attribute name.

        Pure read of the fit result; does **not** modify ``prms``. Always
        includes the SEI parameters used (fixed or fitted) and the degradation
        rate for the regime.
        """
        return {
            "sei_alpha": self.fit_result.get_average_parameter("alpha_sei"),
            "sei_beta": self.fit_result.get_average_parameter("beta_sei"),
            self._prms_rate_name: self.fit_result.get_average_parameter(
                self._lmfit_rate_name
            ),
        }

    def update_prms(self):
        self.echo("Updating prms")
        self.echo("----------------------------------------------")
        if self.split_on_uid and "uid" in self.data.columns:
            warnings.warn("split_on_uid is not fully tested for NonlinearFit")

        fitted = self.fitted_parameters()

        if self.is_reference:
            self.prms.sei_alpha = fitted["sei_alpha"]
            self.prms.sei_beta = fitted["sei_beta"]
            logger.info(f"Got sei_alpha={fitted['sei_alpha']} and sei_beta={fitted['sei_beta']}")

        degradation_rate = fitted[self._prms_rate_name]
        if self.is_cycling and self.calculate_n_at_eol:
            self._calculate_n_eol(degradation_rate)
        if self.update_degradation_parameter:
            self.echo(f"updating degradation parameter: {degradation_rate=}")
            self.prms.set(self._prms_rate_name, degradation_rate)

    def _get_max_x_value(self):
        return self.data[self.x_col].max()

    def _calculate_n_eol(self, deg_per_cyc):
        # Estimate the number of cycles for complete degradation:
        starting_estimate = np.array(
            self.prms.full_degradation_cycle_no_estimate
        )  # only one value, but `scipy.optimize.fsolve` requires an array
        function_arguments = (
            self.prms.sei_alpha,
            self.prms.sei_beta,
            deg_per_cyc,
            self.prms.full_degradation_level,
        )
        function_arguments = tuple(to_numpy_float64(x) for x in function_arguments)
        model_function = models.residuals_solve_cycle_num
        result = fsolve(
            model_function,
            starting_estimate,
            args=function_arguments,
        )
        cyc_num = result[0]
        logger.info(f"Estimated cycle number for complete degradation: {cyc_num:0.0f}")
        logger.info(
            f"Assuming {self.prms.full_degradation_level=} and "
            f"initial guess {self.prms.full_degradation_cycle_no_estimate=}"
        )
        if self.update_degradation_parameter:
            self.echo(f"Updating degradation parameter: {cyc_num=}")
            self.prms.full_degradation_cycle_no = cyc_num
        else:
            self.echo(f"Degradation parameter (not updating prms): {cyc_num=}")


class NonlinearMultiFit:
    """This class is used to fit the Nonlinear model to a data-set grouped by the values in the z_col.

    Args:
        data (pd.DataFrame): The data to fit the model to.
        prms (ESFParams): The parameters of the model.
        z_col (str): The column name of the variable to group the data by.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        fit_results (list): A list of fit results.

    Example:
        >>> # Instantiate the class with data and parameters
        >>> fit = NonlinearMultiFit(data, params, 'temperature')
        >>> # Perform fits and update parameters
        >>> fit.fit()
        >>> # Access results and updated parameters
        >>> results = fit.fit_results
        >>> updated_params = fit.prms

    Note:
        Typically used to find SEI parameters, or fixing SEI parameters and finding
        degradation rates (deg_per_cyc or deg_per_time_unit).
    """

    def __init__(self, data, prms, z_col, *args, **kwargs):
        self.data = data
        self.prms = prms
        self.z_col = z_col
        self.fit_results = []
        self.fit_class = NonlinearFit

        # Extract regime from kwargs (NonlinearMultiFit is not a BaseProcessor,
        # so coerce here rather than through the property setter)
        self.regime = coerce_regime(kwargs.pop("regime", Regime.CALENDAR))

        # Store remaining kwargs
        self.fit_class_args = kwargs
        self.verbose = kwargs.get("verbose", False)

    def fit(self):
        """
        Fit the nonlinear model to each group of data defined by z_col.

        This method creates a NonlinearFit instance for each unique value in z_col,
        fits the model to that subset of data, and stores the results.
        """
        if self.z_col not in self.data.columns:
            raise ValueError(f"Column {self.z_col} not found in data")

        # Group data by z_col
        grouped_data = self.data.groupby(self.z_col)

        # Clear previous results
        self.fit_results = []

        # Log the fitting process
        if self.verbose:
            print(
                f"Fitting nonlinear model to {len(grouped_data)} groups based on {self.z_col}"
            )

        # Fit each group
        for z_val, group_data in grouped_data:
            if self.verbose:
                print(f"Fitting group with {self.z_col}={z_val}")

            # Create a copy of the parameters for this fit
            group_prms = self.prms.copy()

            # Create a NonlinearFit instance for this group
            # Note: We're careful not to pass parameters twice
            fit_instance = self.fit_class(
                data=group_data,
                prms=group_prms,
                regime=self.regime,
                z_col=self.z_col,
                **self.fit_class_args,
            )

            # Perform the fit
            fit_instance.fit()

            # Store the results

            result = {
                "z_col": self.z_col,
                "z_val": z_val,
                "z_unit": group_prms.get_unit(self.z_col),
                "x_col": fit_instance.get_x_col(),
                "x_max": fit_instance.get_max_x_value(),
                "x_unit": group_prms.get_unit(fit_instance.get_x_col()),
                "fit_instance": fit_instance,
                "fit_result": fit_instance.fit_result,
                "fit_parameters": {
                    "alpha_sei": fit_instance.fit_result.get_average_parameter(
                        "alpha_sei", allow_none=True
                    ),
                    "beta_sei": fit_instance.fit_result.get_average_parameter(
                        "beta_sei", allow_none=True
                    ),
                    "deg_per_time_unit": fit_instance.fit_result.get_average_parameter(
                        "deg_per_time_unit", allow_none=True
                    ),
                    "deg_per_cyc": fit_instance.fit_result.get_average_parameter(
                        "deg_per_cyc", allow_none=True
                    ),
                },
                "esf_parameters": {
                    "alpha_sei": fit_instance.prms.sei_alpha,
                    "beta_sei": fit_instance.prms.sei_beta,
                    "deg_per_time_unit": getattr(
                        fit_instance.prms, "deg_per_time_unit", None
                    ),
                    "deg_per_cyc": getattr(fit_instance.prms, "deg_per_cycle", None),
                },
            }

            self.fit_results.append(result)

        return self.fit_results

    def plot_results(
        self,
        title: bool | str | None = None,
        figsize: tuple | None = (6, 6),
        legend_loc: str | None = "center right",
        autoscale_residuals: bool | None = True,
    ):
        """Plot the fitting results for all groups.

        Delegates to :mod:`esf.models.fit_plotting`.
        """
        return fit_plotting.plot_multi_fit(
            self,
            title=title,
            figsize=figsize,
            legend_loc=legend_loc,
            autoscale_residuals=autoscale_residuals,
        )

    def get_parameters_dataframe(
        self, only_degradation_rates=True, keep_original_degradation_rate_name=False
    ):
        """
        Return a DataFrame containing the fitted parameters for each group.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns for z_val and all fitted parameters
        """
        if not self.fit_results:
            raise ValueError("No fit results. Call fit() first.")

        import pandas as pd

        if is_cycling_regime(self.regime):
            deg_value_to_keep = "deg_per_cyc"
            x_col_max_name = "N_max"
        else:
            deg_value_to_keep = "deg_per_time_unit"
            x_col_max_name = "t_max"

        # Extract parameters from each fit result
        data = []
        for result in self.fit_results:
            row = {
                f"{result['x_col']}_max": result["x_max"],
                f"{self.z_col}": result["z_val"],
            }
            row.update(result["fit_parameters"])
            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        if self.verbose:
            print("=" * 80)
            print("Parameters dataframe:")
            print(df)
            print("=" * 80)
        if only_degradation_rates:
            df = df[[x_col_max_name, self.z_col, deg_value_to_keep]]
        if not keep_original_degradation_rate_name:
            df = df.rename(columns={deg_value_to_keep: DEG_RATE_NAME})

        return df

    def simulate(self, x=None, z=None):
        """
        Simulate the model for given x and z values.

        Parameters:
        -----------
        x : array-like, optional
            Values for the independent variable (time or cycles)
        z : array-like, optional
            Values for the grouping variable (z_col)

        Returns:
        --------
        pd.DataFrame
            DataFrame with simulated values
        """
        if not self.fit_results:
            raise ValueError("No fit results. Call fit() first.")

        import pandas as pd

        # If z is not provided, use the z values from the fit results
        if z is None:
            z = [result["z_val"] for result in self.fit_results]
        elif not hasattr(z, "__iter__"):
            z = [z]

        # Create a list to store simulation results
        sim_dfs = []

        # Simulate for each z value
        for z_val in z:
            # Find the closest fit result
            closest_result = min(
                self.fit_results, key=lambda r: abs(r["z_val"] - z_val)
            )

            # Get the fit instance
            fit_instance = closest_result["fit_instance"]

            # Simulate
            sim_df = fit_instance.simulate(x=x)
            sim_df[self.z_col] = z_val

            sim_dfs.append(sim_df)

        # Concatenate all simulation results
        return pd.concat(sim_dfs, ignore_index=True)


def sei_fit_at_reference_conditions(
    prms: ESFParams,
    data: pd.DataFrame,
    data_type: DataType = DataType.CALENDAR_VS_TEMPERATURE,
    verbose: bool = False,
    parameter_overrides: dict | None = None,
    apply: bool = True,
    **kwargs,
):
    # performs a secondary fit for the SEI parameters at reference conditions
    # with apply=False the fit runs without writing into prms (read the
    # results with fit.fitted_parameters() / fit.parameter_uncertainty())
    # data: pd.DataFrame containing the degradation rates.

    # Should update prms and return the fit object
    # Then the user can plot the results using (not implemented yet) fit.plot_results() or plot_fit(fit_object)

    original_prms = prms.copy()
    reference_temperature = prms.reference_temperature
    reference_soc = prms.reference_soc

    regime = data_type.regime
    z_col = data_type.z
    z_val = reference_temperature if z_col == Columns.TEMPERATURE else reference_soc

    if verbose:
        print("=" * 80)
        print("Performing SEI fit at reference conditions")
        print("=" * 80)
        print()
        print("Reference conditions:")
        print(
            f"Reference temperature: {reference_temperature} {prms.temperature_unit}"
        )
        print(f"Reference SoC: {reference_soc} {prms.soc_unit}")

        print(f"Data type: {regime}")
        if is_cycling_regime(regime):
            print(f"Reference cycle number: {prms.reference_cycle}  {prms.cycle_unit}")
        else:
            print(
                f"Reference calendar time: {prms.reference_calendar_time} {prms.time_unit}"
            )

    fit = NonlinearFit(
        data,
        prms,
        z_col=z_col,
        z_val=z_val,
        regime=regime,
        is_reference=True,
        update_degradation_parameter=True,
        verbose=verbose,
        **kwargs,
    )
    fit.fit(parameter_overrides=parameter_overrides, apply=apply)

    if verbose:
        fit.plot_results()
        print(fit.fit_result)
        original_prms.compare(prms, verbose=True)

    return fit


def degradation_rates_fit(
    prms: ESFParams,
    data: pd.DataFrame,
    data_type: DataType = DataType.CALENDAR_VS_TEMPERATURE,
    return_fit_object: bool = False,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame | NonlinearMultiFit:
    _prms = prms.copy()

    regime = data_type.regime
    z_col = data_type.z

    reference_temperature = prms.reference_temperature
    reference_soc = prms.reference_soc

    if verbose:
        print("=" * 80)
        print("Performing degradation rates fit")
        print("=" * 80)
        print()
        print("Reference conditions:")
        print(
            f"Reference temperature: {reference_temperature} {prms.temperature_unit}"
        )
        print(f"Reference SoC: {reference_soc} {prms.soc_unit}")

        print(f"Regime: {regime}")
        print(f"Z column: {z_col}")
        if is_cycling_regime(regime):
            print(f"Cycle unit: {prms.cycle_unit}")
            print(f"Reference cycle number: {prms.reference_cycle}")
        else:
            print(f"Time unit: {prms.time_unit}")
            print(
                f"Reference calendar time: {prms.reference_calendar_time} {prms.time_unit}"
            )

    fit = NonlinearMultiFit(
        data,
        prms,
        z_col=z_col,
        regime=regime,
        is_reference=False,
        verbose=verbose,
        update_degradation_parameter=False,
        **kwargs,
    )
    fit.fit()
    df = fit.get_parameters_dataframe()
    if verbose:
        fit.plot_results()
        print(fit.fit_results)
        print("\nFit results:")
        print("----------------------------------------------")
        print(df)
        print()
    if return_fit_object:
        return fit
    return df


def _print_sf_fit_header(prms, data_type, _title):
    print("=" * 80)
    print(_title)
    print("=" * 80)
    print()
    print("Reference conditions:")
    print(f"Reference SoC: {prms.reference_soc} {prms.soc_unit}")

    print(
        f"Reference temperature: {prms.reference_temperature} {prms.temperature_unit}"
    )
    print(f"Regime: {data_type.regime}")
    if is_cycling_regime(data_type.regime):
        print(f"Cycle unit: {prms.cycle_unit}")
        print(f"Reference cycle number: {prms.reference_cycle}")
    else:
        print(f"Time unit: {prms.time_unit}")
        print(
            f"Reference calendar time: {prms.reference_calendar_time} {prms.time_unit}"
        )


def _report_sf_fit(verbose, fit):
    if verbose:
        fit.plot_results()
        print(fit.fit_result)


def soc_stress_factor_fit(
    prms: ESFParams,
    data: pd.DataFrame,
    data_type: DataType = DataType.CALENDAR_VS_SOC,
    verbose: bool = False,
    update_degradation_parameter: bool = True,
    parameter_overrides: dict | None = None,
    apply: bool = True,
    **kwargs,
):
    _title = "Performing SOC stress factor fit"

    if verbose:
        _print_sf_fit_header(prms, data_type, _title)

    fit = SoCSFfit(
        data,
        prms,
        regime=data_type.regime,
        verbose=verbose,
        update_degradation_parameter=update_degradation_parameter,
        **kwargs,
    )
    fit.fit(parameter_overrides=parameter_overrides, apply=apply)
    _report_sf_fit(verbose, fit)
    return fit


def temperature_stress_factor_fit(
    prms: ESFParams,
    data: pd.DataFrame,
    data_type: DataType = DataType.CALENDAR_VS_TEMPERATURE,
    verbose: bool = False,
    update_degradation_parameter: bool = True,
    parameter_overrides: dict | None = None,
    apply: bool = True,
    **kwargs,
):
    _title = "Performing temperature stress factor fit"

    if verbose:
        _print_sf_fit_header(prms, data_type, _title)

    fit = TemperatureSFfit(
        data,
        prms,
        regime=data_type.regime,
        verbose=verbose,
        update_degradation_parameter=update_degradation_parameter,
        **kwargs,
    )
    fit.fit(parameter_overrides=parameter_overrides, apply=apply)
    _report_sf_fit(verbose, fit)
    return fit


def time_stress_factor_calc(
    prms: ESFParams,
    data: pd.DataFrame,
    data_type: DataType = DataType.CALENDAR_VS_SOC,
    verbose: bool = False,
    apply: bool = True,
    **kwargs,
):
    _title = "Performing time stress factor calculation"

    if verbose:
        _print_sf_fit_header(prms, data_type, _title)

    sf_calc = TimeSFCalc(
        data,
        prms,
        x_col=data_type.z,
        regime=data_type.regime,
        verbose=verbose,
        **kwargs,
    )
    sf_calc.calc(apply=apply)
    return sf_calc


def calculate_time_stress_factor(prms: ESFParams):
    """
    Calculate the time stress factor for a given data set.

    S(t) = k_t * t

    Once the temperature and SoC stress model coefficients (k_T, k_soc) are known,
    the time stress factor can be calculated as using the result of calendar aging tests.

    k_t = f_d_t[T=T_ref, SoC=SoC_ref] / t * S_T(T_ref) * S_soc(SoC_ref)

    t is the duration of the calendar aging test.

    Remark: This function only works if you have the all the relevant parameters correct in
    your prms object.
    """

    warnings.warn(
        "This function is unreliable. Use time_stress_factor_calc instead.",
        DeprecationWarning,
    )

    # First find the degradation rate and reference time
    f = prms.deg_per_time_unit
    print(f" -> {f=}")
    t = prms.reference_calendar_time
    print(f" -> {t=}")

    # Then find the stress factors for the temperature and SoC at the reference conditions
    stress_factor_temperature = prms.k_temperature_calendar
    stress_factor_soc = prms.k_soc_calendar

    print(f" -> {stress_factor_temperature=}")
    print(f" -> {stress_factor_soc=}")

    # Then calculate the time stress factor
    k_t = f / (t * stress_factor_temperature * stress_factor_soc)

    print(f" -> {k_t=}")

    prms.k_1_time_calendar = k_t
    return k_t


def dod_stress_factor_fit(
    prms: ESFParams,
    data: pd.DataFrame,
    data_type: DataType = DataType.CYCLE_VS_DOD,
    verbose: bool = False,
    update_degradation_parameter: bool = False,
    is_at_reference: bool = True,
    parameter_overrides: dict | None = None,
    apply: bool = True,
    **kwargs,
):
    """Fit the DoD stress factor to cycle-life-vs-DoD data.

    The data must have a ``DoD`` column and a cycle-life ``N`` column. With
    ``is_at_reference=True`` (default) the cycles are assumed to be at the
    reference SoC and temperature, so only DoD varies. With
    ``is_at_reference=False`` the temperature, SoC and calendar-time stress
    factors are removed first (Xu et al. 2018, eqs. 20/31), which requires the
    columns ``T`` (K), ``SoC`` (fraction) and ``t_cycle`` (s) per point and the
    temperature/SoC/time stages already fitted in ``prms``. The DoD model form
    follows the chemistry (Empirical/Exponential/Quadratic; decision 6). The
    fitted ``k_*_dod`` constants land in ``prms``. See ``DoDSFfit``.
    """
    _title = "Performing dod stress factor fit"

    if verbose:
        _print_sf_fit_header(prms, data_type, _title)

    fit = DoDSFfit(
        data,
        prms,
        regime=data_type.regime,
        verbose=verbose,
        update_degradation_parameter=update_degradation_parameter,
        is_at_reference=is_at_reference,
        **kwargs,
    )
    fit.fit(parameter_overrides=parameter_overrides, apply=apply)
    _report_sf_fit(verbose, fit)
    return fit
