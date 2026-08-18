import copy
import json
import logging
import pathlib
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, make_dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd
import uncertainties

from esf.settings.units import Q_, ureg  # noqa: F401  (re-exported for consumers)

logger = logging.getLogger(__name__)

THIS_FOLDER = pathlib.Path(__file__).parent
DATA_FOLDER = (THIS_FOLDER / "../../data/").resolve()
DST_DATA_FOLDER = (DATA_FOLDER / "Ageing_Data_Org/DST_cycles/").resolve()
PRMS_FOLDER = (DATA_FOLDER / "DegModelPara/").resolve()
ESF_PARAMS_VERSION: int = 1

TEMPERATURE_UNIT = "K"  # or "degC"

TIME_UNIT = "seconds"  # maybe it should be hours instead?
CYCLE_UNIT = "N"
SOC_UNIT = "frac"
DOD_UNIT = "frac"
SOH_UNIT = "frac"
L_UNIT = "frac"
RATE_UNIT = "C"  # not used yet

SF_COL = "SF"

CATEGORICAL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# TODO: implement this in the simulators:


class ESFParamsModelTypeExtension(StrEnum):
    """Enum for enforcing the correct naming convention for model type parameters"""

    CALENDAR = "_calendar_model"
    CYCLING = "_cycling_model"


MODEL_SETS = {
    "default": {
        "Calendar": ["time", "soc", "temperature"],
        "Cycling": ["dod", "soc", "temperature", "high_soc"],
    },
    "original": {
        "Calendar": ["time", "soc", "temperature"],
        "Cycling": ["dod", "soc", "temperature"],
    },
    "all": {
        "Calendar": ["time", "soc", "temperature"],
        "Cycling": ["dod", "soc", "temperature", "high_soc"],
    },
}


# Xu et al. (2018) use a different DoD stress-factor form per chemistry
# (decision 6 in the DoD-fitting decision record): the empirical form for LMO,
# the exponential form for LFP, and the quadratic (power-law) form for NMC.
# Used to pick ``dod_cycling_model`` from ``battery_chemistry`` when the model
# label is not set explicitly.
DOD_MODEL_BY_CHEMISTRY = {
    "LMO": "Empirical",
    "LFP": "Exponential",
    "NMC": "Quadratic",
}
DEFAULT_DOD_MODEL = "Empirical"


class Regime(StrEnum):
    """Enum for data types."""

    CALENDAR = "calend"
    CYCLING = "cycling"
    OPERATIONAL = "operational"


class Columns(StrEnum):
    """Enum for columns."""

    TIME = "time"
    SOC = "SoC"
    TEMPERATURE = "T"
    DOD = "DoD"


class DataType(StrEnum):
    """Enum for data types."""

    CALENDAR_VS_TEMPERATURE = "cal-vs-temp"
    CALENDAR_VS_SOC = "cal-vs-soc"
    CYCLE_VS_TEMPERATURE = "cyc-vs-temp"
    CYCLE_VS_DOD = "cyc-vs-dod"
    OPERATIONAL_DATA = "op-data"
    # TODO: add more data types (e.g. cycle vs. c-rate)
    # - if so, also update the z, data_type, and is_nlfd properties
    # - and add the new data types to the MINIMUM_COLUMNS dict in esf/io/data.py
    # - and maybe more places in the data class
    # - Also, remember to update the Columns enum in esf/settings/parameters.py

    @property
    def z(self) -> str:
        """Get the z column name from the DataType. Only works for non-operational data."""
        if self.value.endswith("-temp"):
            return Columns.TEMPERATURE
        elif self.value.endswith("-soc"):
            return Columns.SOC
        elif self.value.endswith("-dod"):
            return Columns.DOD
        return ""

    @property
    def regime(self) -> str:
        """Get the data type from the DataType."""
        if self.value.startswith("cal-"):
            return Regime.CALENDAR
        elif self.value.startswith("cyc-"):
            return Regime.CYCLING
        elif self.value.startswith("op-"):
            return Regime.OPERATIONAL
        return ""

    @property
    def is_nlfd(self) -> bool:
        """Check if the data type is a non-linear fit data type."""
        return self.value.endswith("-temp") or self.value.endswith("-soc")


@dataclass
class ESFParams:
    """
    Dataclass for ESF parameters.

    Note that the dataclass needs to obey the following conventions:
        1. the model types should end with the appropriate name as
            defined in the ESFParamsModelTypeExtension

    Methods:
        __repr__(): Returns a string representation of the dataclass.
        get_model(model_type, model_label): Returns a model from the model register.
        clean_up(): Not implemented.
        compare(other): Compares the dataclass with another dataclass.
        to_dict(): Converts the dataclass to a dictionary.
        to_json(): Converts the dataclass to a JSON string.
        from_json(json_str): Creates a dataclass from a JSON string.
        save_json(file_path): Saves the dataclass as a JSON file.
        load_json(file_path): Loads a dataclass from a JSON
        to_frame(): Converts the dataclass to a pandas DataFrame.
        to_short_frame(): Converts fit-able variables in the dataclass to a pandas DataFrame.
        update(dict): Updates the dataclass with a dictionary. To be used when adding custom parameters.
        mark_changed(): Marks the dataclass as changed (i.e. updates the last_updated variable).
    """

    # no validation is implemented
    # no unit tests are implemented yet
    version: int = ESF_PARAMS_VERSION
    mode: str = "default"  # "default", "uncertainties"
    user: str = "user"
    date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    sample_data_info: str = "sample data info"
    battery_chemistry: str = "not specified"
    # For choosing the set of models to use:
    degradation_model: str = "default"

    time_unit: str = TIME_UNIT
    temperature_unit: str = TEMPERATURE_UNIT
    cycle_unit: str = CYCLE_UNIT
    soc_unit: str = SOC_UNIT
    dod_unit: str = DOD_UNIT
    soh_unit: str = SOH_UNIT
    l_unit: str = L_UNIT
    rate_unit: str = RATE_UNIT

    # Default model types for calendar degradation:
    temperature_calendar_model: str = "Exponential"  # old
    time_calendar_model: str = "Proportional"  # old
    soc_calendar_model: str = "Exponential"
    high_soc_calendar_model: str = "Exponential"

    # Default model types for cycling degradation:
    temperature_cycling_model: str = "Exponential"  # old
    soc_cycling_model: str = "Linear"
    high_soc_cycling_model: str = "Exponential"
    # Left empty on purpose: __post_init__ resolves it from battery_chemistry
    # (decision 6) unless the user sets a label explicitly. An explicit label
    # always wins.
    dod_cycling_model: str = ""
    rate_cycling_model: str = "Linear"

    # Validity of the parameters:
    temperature_range_min: float = 0.0
    temperature_range_max: float = 100.0
    soc_range_min: float = 0.0
    soc_range_max: float = 1.0
    dod_range_min: float = 0.0
    dod_range_max: float = 1.0
    soh_range_min: float = 0.0
    soh_range_max: float = 1.0
    l_range_min: float = 0.0
    l_range_max: float = 1.0
    rate_range_min: float = 0.0
    rate_range_max: float = 10.0

    # Experimental information:
    number_of_cells_used: int = 1
    reference_for_datasets: str = "not specified"
    reference_for_models: str = "not specified"

    # Nonlinear degradation model parameters:
    sei_alpha: float = 0.0575763976
    sei_beta: float = 113.383941797
    reference_cycle: float = 1.0
    reference_calendar_time: float = 3600 * 24.0  # 1 day in seconds

    # Linear degradations obtained at reference conditions (reference_soc and reference_temperature):
    deg_per_cycle: float = 0.00002
    deg_per_time_unit: float = 3.5120068288541084e-05

    # These are not actually used by the esf model, but might be useful for debugging or apps etc:
    soc_deg_rate_at_reference_conditions: float = 1.0
    temperature_deg_rate_at_reference_conditions: float = 1.0
    duration_calendar_at_reference_conditions: float = 157788000.0

    # how much degradation (L-value) is considered full degradation:
    full_degradation_level: float = 0.2
    full_degradation_cycle_no: float = 3699.0
    full_degradation_cycle_no_estimate: float = 200.0

    # Temperature model parameters:
    reference_temperature: float = 298.15
    k_temperature_calendar: float = 0.0698782039
    k_temperature_cycling: float = 0.0698782039

    k_temperature_slope: float = 0.0001  # not used?
    k_temperature_intercept: float = 0.0001  # not used?

    # SOC model parameters:
    reference_soc: float = 0.5
    k_soc_calendar: float = 1.0249254361
    k_soc_cycling: float = 1.0249254361

    # High SOC model parameters:
    reference_high_soc: float = 0.85
    k_high_soc_calendar: float = 1.391411435
    k_high_soc_cycling: float = 1.391411435

    # Time model parameters:
    reference_time: float = 1.0
    k_1_time_calendar: float = 0.0000000004
    k_2_time_calendar: float = 0.0

    # DOD model parameters:
    reference_dod: float = 1.0
    k_1_dod: float = 173050.0
    k_2_dod: float = -0.501
    k_3_dod: float = -147740.0

    # Rate model parameters (not implemented yet):
    reference_rate: float = 1.0
    k_1_rate: float = 0.0001
    k_2_rate: float = 0.0001
    k_3_rate: float = 0.0001

    # ------------------------------------------------------------------------------------------------
    #        additional parameters already added
    # ------------------------------------------------------------------------------------------------
    k_linear_1_soc_calendar: float = 0.0001  # added linear soc model
    k_linear_2_soc_calendar: float = 0.0001  # added linear soc model
    k_linear_1_soc_cycling: float = 0.0001  # added linear soc model
    k_linear_2_soc_cycling: float = 0.0001  # added linear soc model

    # ------------------------------------------------------------------------------------------------
    #        additional parameters to be added (for example for additional stress factors models)
    # ------------------------------------------------------------------------------------------------
    # names of parameters added at runtime via update(); recorded so that
    # save_json/load_json can round-trip them (see load_json)
    custom_parameter_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        # decision 6: pick the DoD stress-factor form from the chemistry unless
        # the user set an explicit label. Unknown chemistries fall back to the
        # empirical (LMO) form, which is what the codebase used historically.
        if not self.dod_cycling_model:
            self.dod_cycling_model = DOD_MODEL_BY_CHEMISTRY.get(
                self.battery_chemistry, DEFAULT_DOD_MODEL
            )

    def mark_changed(self):
        self.last_updated = datetime.now().isoformat()

    def pprint(self) -> None:
        txt = 80 * "=" + "\n"
        txt += "ESFParams".center(80) + "\n"
        txt += 80 * "-" + "\n"
        txt += f"{'  Parameter':<37} | {'Value':<22} | {'Type'}\n"
        txt += 80 * "=" + "\n"

        for key, value in self.__dict__.items():
            is_bad = False
            dtype = type(value)
            if dtype is str:
                formatted_value = f'"{value}"'
            elif dtype in (list, tuple, np.ndarray):
                txt += "\nWARNING! list, tuple, or ndarray observed:\n"
                formatted_value = f"{value}"
                is_bad = True
            elif dtype in [float, np.float64, np.float32]:
                formatted_value = f"{value:.8f}"
            else:
                formatted_value = f"{value}"
            formatted_value = formatted_value.ljust(22)
            txt += f"  {key.ljust(35)} | {formatted_value} | {type(value)}\n"
            if is_bad:
                txt += "\n"
        txt += 80 * "-" + "\n"
        print(txt)

    def get_model_set(self) -> dict:
        model_set_name = self.degradation_model
        return MODEL_SETS[model_set_name]

    def pprint_model_set(self) -> None:
        txt = 80 * "=" + "\n"
        txt += "ESFParams".center(80) + "\n"
        txt += 80 * "-" + "\n"
        txt += f"  Using model set: {self.degradation_model}\n"
        txt += 80 * "=" + "\n"

        for regime in ("Calendar", "Cycling"):
            txt += f"\n  {regime} model:\n"
            for factor, function, values in self.stress_models(regime):
                txt += 80 * "-" + "\n"
                txt += f"    Stress factor: {factor}\n"
                txt += f"    Model label: {self.stress_model_label(regime, factor)}\n"
                txt += f"    Model function: {function}\n"
                txt += f"    Model parameters: {values}\n"
        txt += 80 * "=" + "\n"
        print(txt)

    def clean_up(self):
        print("Sorry, this method is not implemented yet.")

    def compare(self, other: "ESFParams", verbose=False) -> bool:
        if verbose:
            print(80 * "=")
            print("Comparing ESFParams".center(80))
            print(80 * "=")
            print(f" {'variable':25s} | {'self':27} | other")
            print(80 * "-")
        is_equal = True
        for key, value in self.__dict__.items():
            if key not in other.__dict__:
                if verbose:
                    print(f"WARNING: Key {key} not found in other")
                is_equal = False
                continue
            if value != other.__dict__[key]:
                if verbose:
                    print(f" {key:25s} | {value:27} | {other.__dict__[key]}")
                is_equal = False
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            if verbose:
                print("WARNING: Keys are not equal")
                print(
                    f"self ^ other: {set(self.__dict__.keys()) ^ set(other.__dict__.keys())}"
                )
            is_equal = False
        if verbose:
            print(80 * "-")
        return is_equal

    def get_unit(self, key: str) -> str:
        if key in ["T", "temperature", "Temperature"]:
            return self.temperature_unit
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
            print(f"WARNING: Unit for {key} not found")
            return None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        try:
            return json.dumps(self.to_dict())
        except Exception as e:
            print(e)

    def save_json(self, file_path: str | pathlib.Path):
        # Same file format as DataFrame.to_json (a {"Value": {...}} mapping, so
        # load_json and existing parameter files keep working), but written with
        # the json module: pandas' double_precision=10 default silently corrupts
        # small parameters (e.g. k_1_time_calendar = 4.14e-10 became 4e-10).
        def _encode(value):
            if isinstance(value, uncertainties.UFloat):
                return {
                    "nominal_value": value.nominal_value,
                    "std_dev": value.std_dev,
                }
            return float(value)

        with open(file_path, "w") as f:
            json.dump({"Value": self.to_dict()}, f, default=_encode)

    def to_frame(self):
        return pd.DataFrame.from_dict(self.to_dict(), orient="index", columns=["Value"])

    def to_short_frame(self):
        return pd.DataFrame.from_dict(
            {
                "sei_alpha": self.sei_alpha,
                "sei_beta": self.sei_beta,
                "deg_per_cycle": self.deg_per_cycle,
                "deg_per_time_unit": self.deg_per_time_unit,
                "reference_temperature": self.reference_temperature,
                "k_temperature_slope": self.k_temperature_slope,
                "k_temperature_intercept": self.k_temperature_intercept,
                "k_temperature_calendar": self.k_temperature_calendar,
                "reference_soc": self.reference_soc,
                "k_soc_calendar": self.k_soc_calendar,
                "reference_high_soc": self.reference_high_soc,
                "k_high_soc_cycling": self.k_high_soc_cycling,
                "reference_time": self.reference_time,
                "k_1_time_calendar": self.k_1_time_calendar,
                "k_dod_1": self.k_1_dod,
                "k_dod_2": self.k_2_dod,
                "k_dod_3": self.k_3_dod,
            },
            orient="index",
            columns=["Value"],
        )

    def convert_units(self, unit="temperature", unit_from="seconds", unit_to="hours"):
        """Convert units for parameters.

        Args:
            unit (str): Type of unit to convert (e.g. "temperature")
            unit_from (str): Current unit
            unit_to (str): Target unit
        """

        if unit_from == unit_to:
            print(f"Units are already {unit_from}")
            return

        # Dictionary mapping unit types to their conversion settings
        unit_converters = {
            "temperature": {
                "unit_attr": "temperature_unit",  # attribute storing the current unit
                "params": ["reference_temperature"],  # parameters to convert
                "valid_units": ["K", "degC"],  # allowed units
            },
            # Can be extended with other unit types like:
            # "time": {
            #     "unit_attr": "time_unit",
            #     "params": ["reference_time", "duration_calendar"],
            #     "valid_units": ["seconds", "hours", "days"],
            # },
        }

        if unit not in unit_converters:
            print(f"Cannot convert units for {unit}")
            return

        converter = unit_converters[unit]
        if (
            unit_from not in converter["valid_units"]
            or unit_to not in converter["valid_units"]
        ):
            print(f"Cannot convert {unit} from {unit_from} to {unit_to}")
            print(f"Supported units are: {converter['valid_units']}")
            return

        current_unit = getattr(self, converter["unit_attr"])
        if current_unit != unit_from:
            print(f"Current unit is {current_unit}, not {unit_from}")
            return

        # Convert each parameter
        for param in converter["params"]:
            current_value = getattr(self, param)
            new_value = float(Q_(current_value, unit_from).to(unit_to).magnitude)
            setattr(self, param, new_value)

        # Update the unit attribute
        setattr(self, converter["unit_attr"], unit_to)

    @staticmethod
    def _to_uncertainty(value: dict):
        """This helper method converts a dictionary item as created by pandas to an uncertainty object."""
        val = value["nominal_value"]
        std = value["std_dev"]
        return uncertainties.ufloat(val, std)

    @classmethod
    def load_json(cls, file_path: str | pathlib.Path) -> "ESFParams":
        # read with the json module (pandas' default reader is not
        # round-trip exact for floats)
        with open(file_path) as f:
            payload = json.load(f)
        return cls._from_parameter_mapping(payload["Value"], source=file_path)

    @classmethod
    def from_json(cls, json_str: str) -> "ESFParams":
        return cls._from_parameter_mapping(json.loads(json_str), source="json string")

    @classmethod
    def _from_parameter_mapping(cls, mapping: dict, source="mapping") -> "ESFParams":
        """Build an instance from a {name: value} mapping.

        Unknown keys that the file declares in ``custom_parameter_names`` are
        restored as custom parameters (they were added with ``update()``
        before saving); any other unknown key is a stale/legacy parameter and
        is skipped with a warning so old parameter files keep loading.
        """
        valid_keys = {f.name for f in fields(cls)}
        declared_custom = set(mapping.get("custom_parameter_names") or [])
        parameters = {}
        custom = {}
        unknown_keys = []
        for key, value in mapping.items():
            if isinstance(value, dict):
                value = cls._to_uncertainty(value)
            if key in valid_keys:
                parameters[key] = value
            elif key in declared_custom:
                custom[key] = value
            else:
                unknown_keys.append(key)
        if unknown_keys:
            warnings.warn(
                f"Ignoring unknown parameters in {source}: {sorted(unknown_keys)}"
            )
        # custom_parameter_names is rebuilt by update() below
        parameters.pop("custom_parameter_names", None)
        instance = cls(**parameters)
        if custom:
            instance.update(custom)
        return instance

    @staticmethod
    def _get_model_object(model_regime: str, model_type: str, model_label: str):
        from esf import _mr as model_register

        model = model_register.get(model_regime, model_type, model_label)
        if model is None:
            raise AttributeError(
                f"Model not found: {model_regime}, {model_type}, {model_label}"
            )
        return model

    @classmethod
    def get_model(
        cls, model_regime: str, model_type: str, model_label: str
    ) -> Callable | None:
        try:
            m = cls._get_model_object(model_regime, model_type, model_label)
        except AttributeError as e:
            print(e)
            return None
        return m

    @classmethod
    def get_model_function(
        cls,
        model_regime: str,
        model_type: str,
        model_label: str,
        wrap_for_uncertainties: bool = False,
        raise_on_error=False,
    ) -> Callable | None:
        try:
            m = cls._get_model_object(model_regime, model_type, model_label).model
        except AttributeError as e:
            print(e)
            if raise_on_error:
                raise e
            return None

        if wrap_for_uncertainties:
            return uncertainties.wrap(m)
        return m

    @classmethod
    def get_model_esf_parameters(
        cls,
        model_regime: str,
        model_type: str,
        model_label: str,
        include_independent: bool = False,
    ) -> Callable | None:
        try:
            parameters = cls._get_model_object(
                model_regime, model_type, model_label
            ).esf_parameters
        except AttributeError as e:
            print(e)
            return None
        if not include_independent:
            parameters = parameters[1:]
        return parameters

    @classmethod
    def get_model_parameter_dict(
        cls, model_regime: str, model_type: str, model_label: str
    ) -> Callable | None:
        try:
            parameter_dict = cls._get_model_object(
                model_regime, model_type, model_label
            ).fit_parameter_dict
        except AttributeError as e:
            print(e)
            return None
        return parameter_dict

    def copy(self):
        return copy.deepcopy(self)

    def get(self, key: str, of_type: str | None = None):
        if of_type is not None:
            return getattr(self, key)[of_type]
        return getattr(self, key)

    def set(self, key: str, value):
        if not hasattr(self, key):
            print(f"Unknown attribute:{key}")
            print(
                " - note that this attribute will not be handled as an ESF parameter."
            )
        setattr(self, key, value)

    def update(self, d: dict):
        """Update existing parameters, or add new custom parameters.

        Unknown keys are promoted to real dataclass fields (so they take part
        in ``to_dict``/``save_json``) and recorded in
        ``custom_parameter_names`` so ``load_json``/``from_json`` can restore
        them (see G9 in the round-3 review).
        """
        new_fields = []
        new = {}
        for key, value in d.items():
            if not hasattr(self, key):
                logger.info(f"Adding new custom attribute: {key}")
                new_fields.append((key, type(value), field(default=value)))
                new[key] = value
            else:
                setattr(self, key, value)
        if new_fields:
            # base the extended class on the *current* class so that custom
            # parameters from earlier update() calls keep their field status
            # (previously each call re-based on ESFParams, silently dropping
            # earlier custom fields from serialization)
            self.__class__ = make_dataclass(
                "uESFParams",
                fields=new_fields,
                bases=(type(self),),
            )
            for key, value in new.items():
                setattr(self, key, value)
            self.custom_parameter_names = list(self.custom_parameter_names) + [
                key for key, *_ in new_fields
            ]

    def get_all(self, keys: list[str]):
        return [self.get(key) for key in keys]

    # --- generic stress-model access -------------------------------------
    #
    # Every stress factor of both regimes is resolved the same way: the
    # configured label (e.g. "Exponential") is stored in the attribute
    # "<factor>_<regime>_model", and (regime, factor, label) identifies the
    # model in the register. The methods below replace the former ~36
    # hand-written properties; the old property names still resolve through
    # __getattr__ (deprecated).

    def stress_model_label(self, regime: str, factor: str) -> str:
        """The configured model label (e.g. "Exponential") for a stress factor.

        Args:
            regime: "Calendar" or "Cycling"
            factor: e.g. "time", "soc", "temperature", "dod", "high_soc"
        """
        return self.get(f"{factor}_{regime.lower()}_model")

    def stress_model(self, regime: str, factor: str):
        """The registered ModelItem for a stress factor (None if not registered)."""
        return self.get_model(regime, factor, self.stress_model_label(regime, factor))

    def stress_model_function(self, regime: str, factor: str, **kwargs):
        """The model function for a stress factor."""
        return self.get_model_function(
            regime, factor, self.stress_model_label(regime, factor), **kwargs
        )

    def stress_model_parameter_names(
        self, regime: str, factor: str, include_independent: bool = False
    ):
        """The ESFParams attribute names holding the model's parameters."""
        return self.get_model_esf_parameters(
            regime,
            factor,
            self.stress_model_label(regime, factor),
            include_independent=include_independent,
        )

    def stress_model_parameter_values(self, regime: str, factor: str):
        """The current parameter values for a stress factor's model."""
        return self.get_all(self.stress_model_parameter_names(regime, factor))

    def stress_model_fit_parameters(self, regime: str, factor: str):
        """The lmfit parameter dictionary for a stress factor's model."""
        return self.get_model_parameter_dict(
            regime, factor, self.stress_model_label(regime, factor)
        )

    def stress_models(self, regime: str):
        """All (factor, function, parameter_values) of the active model set."""
        return [
            (
                factor,
                self.stress_model_function(regime, factor),
                self.stress_model_parameter_values(regime, factor),
            )
            for factor in self.get_model_set()[regime]
        ]

    @staticmethod
    def _parse_stress_model_alias(name: str):
        """Parse deprecated property names like 'soc_calendar_stress_model_function'."""
        for suffix in (
            "_stress_model_parameters_values",
            "_stress_model_parameters",
            "_stress_model_function",
            "_stress_model",
        ):
            if name.endswith(suffix):
                head = name[: -len(suffix)]
                break
        else:
            return None
        if head.endswith("_calendar"):
            regime, factor = "Calendar", head[: -len("_calendar")]
        elif head.endswith("_cycling"):
            regime, factor = "Cycling", head[: -len("_cycling")]
        elif head == "dod":
            # the dod aliases never carried the regime in their name
            regime, factor = "Cycling", "dod"
        else:
            return None
        if not factor:
            return None
        return regime, factor, suffix

    def __getattr__(self, name: str):
        # only called when normal attribute lookup fails, i.e. for the
        # deprecated generated property names
        parsed = self._parse_stress_model_alias(name)
        if parsed is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        regime, factor, suffix = parsed
        replacement = {
            "_stress_model": f"stress_model({regime!r}, {factor!r})",
            "_stress_model_function": f"stress_model_function({regime!r}, {factor!r})",
            "_stress_model_parameters": f"stress_model_fit_parameters({regime!r}, {factor!r})",
            "_stress_model_parameters_values": f"stress_model_parameter_values({regime!r}, {factor!r})",
        }[suffix]
        warnings.warn(
            f"'{name}' is deprecated; use {replacement} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if suffix == "_stress_model":
            return self.stress_model(regime, factor)
        if suffix == "_stress_model_function":
            return self.stress_model_function(regime, factor)
        if suffix == "_stress_model_parameters":
            return self.stress_model_fit_parameters(regime, factor)
        return self.stress_model_parameter_values(regime, factor)

    @property
    def sim_parameters(self):
        # TODO: update this so that it returns parameters based on the models used
        return {
            "sei_alpha": self.sei_alpha,
            "sei_beta": self.sei_beta,
            "reference_temperature": self.reference_temperature,
            "k_temperature_cycling": self.k_temperature_cycling,
            "reference_soc": self.reference_soc,
            "k_soc_calendar": self.k_soc_calendar,
            "reference_time": self.reference_time,
            "k_1_time_calendar": self.k_1_time_calendar,
            "k_1_dod": self.k_1_dod,
            "k_2_dod": self.k_2_dod,
            "k_3_dod": self.k_3_dod,
        }

    @property
    def sim_stress_models(self) -> dict:
        # TODO: Should be a dictionary with model
        #  names as keys and model functions as values.
        pass

    @property
    def sim_stress_model_parameters(self) -> dict:
        """

        Returns:
            dict: Dictionary with stress model parameters.
        Example:
        output = {
            "time_stress_model": (k_time, reference_time),
            "soc_stress_model": (k_soc, reference_soc),
            "temperature_stress_model": (k_temperature, reference_temperature),
            "dod_stress_mode: (k_dod_1, k_dod_2, k_dod_3)
        }
        """
        # TODO: update this so that it returns parameters based on the models used.
        pass


def get_example_params():
    # need to make sure the parameters for cycling vs calendar are consistent
    return ESFParams(
        sample_data_info="Example parameters from the original ESF paper by Xu et al.",
        # temperatures are kelvin throughout (see the units convention in the
        # README); the k_temperature constants from the paper are
        # kelvin-parameterized, so combining them with degC references gave
        # wrong stress factors
        temperature_unit="K",
        time_unit="seconds",
        cycle_unit="N",
        temperature_calendar_model="Exponential",
        temperature_cycling_model="Exponential",
        soc_calendar_model="Exponential",
        soc_cycling_model="Exponential",
        high_soc_calendar_model="Exponential",
        high_soc_cycling_model="Exponential",
        time_calendar_model="Proportional",
        dod_cycling_model="Empirical",
        sei_alpha=0.0575,
        sei_beta=121.0,
        k_1_dod=1.4e5,
        k_2_dod=-5.01e-1,
        k_3_dod=-1.23e5,
        k_soc_calendar=1.04,
        k_soc_cycling=1.04,
        k_high_soc_calendar=10.2,
        k_high_soc_cycling=10.2,
        reference_soc=0.5,
        k_temperature_calendar=6.93e-2,
        k_temperature_cycling=6.93e-2,
        reference_temperature=298.15,  # K
        k_1_time_calendar=4.14e-10,  # pr. second
        reference_calendar_time=3600 * 24.0,
    )


def get_example_params_from_original_repo():

    return ESFParams(
        sample_data_info="Example parameters from the code by original ESF paper by Xu et al.",
        degradation_model="original",
        temperature_unit="K",
        time_unit="seconds",
        cycle_unit="N",
        temperature_calendar_model="Exponential",
        temperature_cycling_model="Exponential",
        soc_calendar_model="Exponential",
        soc_cycling_model="Exponential",
        time_calendar_model="Proportional",
        dod_cycling_model="Empirical",
        sei_alpha=5.87e-02,
        sei_beta=1.06e02,
        k_1_dod=1.47e04,
        k_2_dod=-1.65e00,
        k_3_dod=3.61e02,
        k_soc_calendar=1.01e00,
        reference_soc=0.5,
        k_high_soc_cycling=10.2,
        reference_high_soc=0.85,
        k_temperature_cycling=6.71e-02,
        reference_temperature=298.15,  # K
        k_1_time_calendar=4.11e-10,  # pr. second
        reference_calendar_time=3600 * 24.0,
    )
