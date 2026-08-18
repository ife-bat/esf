import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from lmfit import Model

from esf.models import base_models


@dataclass
class ModelItem:
    """Dataclass for models."""

    name: str
    model_regime: str
    model_type: str
    label: str
    description: str
    original_parameters: list[str]
    esf_parameters: list[str]
    esf_reference_parameter: str = "x_ref"
    fit_parameter_dict: dict = field(default_factory=dict)
    model: Callable = field(default=None, repr=False)

    def get_parameter_dict(self, reference_value: float | None = None) -> dict:
        # deep copy: a shallow copy shares the nested per-parameter dicts, so
        # fits that adjust e.g. x_ref would corrupt the registry defaults for
        # every later fit
        parameter_dict = copy.deepcopy(self.fit_parameter_dict)
        if reference_value is not None and reference_value in parameter_dict:
            parameter_dict[self.esf_reference_parameter] = dict(
                value=reference_value, vary=False
            )
        return parameter_dict

    @property
    def independent_vars(self):
        return self.original_parameters[0:1]

    @property
    def independent_var(self):
        return self.original_parameters[0]

    @property
    def lmfit_model(self):
        return Model(self.model, independent_vars=self.independent_vars)


class ModelRegister:
    """Registry of stress-factor models, keyed by (regime, type, label)."""

    def __init__(self):
        self.models = {}

    def register(
        self,
        model,
        model_regime,
        model_type,
        model_label,
        parameter_names,
        parameter_dict=None,
    ):
        """
        Register a model.

        Args:
            model: model function
            model_regime: model regime ("Cycling" or "Calendar")
            model_type: model type (e.g., dod, soc, temperature)
            model_label: model subtype (e.g., Linear, Exponential)
            parameter_names: the ESFParams attribute name for each function
                parameter, starting with the independent variable
                (e.g. ["x", "k_soc_calendar", "reference_soc"])
            parameter_dict: parameter dictionary used for the lmfit-model fitting (must use the same parameters
                names as in the model)
        """
        if not parameter_names:
            raise ValueError(
                f"parameter_names is required when registering {model.__name__}"
            )
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(
                f"duplicate parameter names for {model.__name__}: {parameter_names}"
            )
        key = (model_regime, model_type, model_label)
        if key in self.models:
            raise ValueError(f"model already registered: {key}")
        self.models[key] = ModelItem(
            name=model.__name__,
            model_regime=model_regime,
            model_type=model_type,
            label=model_label,
            description=model.__doc__,
            original_parameters=model.__code__.co_varnames,
            esf_parameters=parameter_names,
            fit_parameter_dict=parameter_dict,
            model=model,
        )

    def get(self, model_regime, model_type, model_label):
        """Look up a registered model; returns None if not registered."""
        return self.models.get((model_regime, model_type, model_label))

    def __repr__(self):
        from pprint import pformat

        txt = "-------------\n"
        txt += "ModelRegister\n"
        txt += "-------------\n"
        for key, value in self.models.items():
            txt += f"{key}:\n   {pformat(value)}\n"
        return txt


def run():
    # TODO: make sure that the model register uses the correct model names and parameter names vs ESFParameter
    # Should utilze enums here!
    model_register = ModelRegister()
    model_register.register(
        base_models.empirical_dod_k_relation,
        "Cycling",
        "dod",
        "Empirical",
        parameter_names=["x", "k_1_dod", "k_2_dod", "k_3_dod", "reference_dod"],
        # Bounds are deliberately generic: the previous k_3 window
        # [-1.23e6, -1.23e4] was tuned to LMO and rejected other chemistries
        # (e.g. LFP has a positive k_3). k_2 stays negative (the empirical
        # cycle-life law decreases with DoD); k_1 stays positive.
        parameter_dict=dict(
            k_1=dict(value=140000.0, vary=True, min=0.0),
            k_2=dict(value=-0.501, vary=True, min=-10.0, max=-0.001),
            k_3=dict(value=-123000.0, vary=True),
            x_ref=dict(value=1.0, vary=False),
        ),
    )
    model_register.register(
        base_models.exponential_dod_k_relation,
        "Cycling",
        "dod",
        "Exponential",
        parameter_names=["x", "k_1_dod", "k_2_dod", "k_3_dod", "reference_dod"],
        parameter_dict=dict(
            k_1=dict(value=1e04, vary=True),
            k_2=dict(value=-0.5, vary=True),
            k_3=dict(value=0.0, vary=False),
            x_ref=dict(value=1.0, vary=False),
        ),
    )
    model_register.register(
        base_models.quadratic_dod_k_relation,
        "Cycling",
        "dod",
        "Quadratic",
        # NMC form (Xu et al. eq. 30). Only two parameters; k_3_dod is unused
        # for this form. S = k_1 * delta^k_2 is a rate magnitude, so k_1 > 0 and
        # k_2 > 0 (the per-cycle rate grows with DoD -> cycle life falls).
        parameter_names=["x", "k_1_dod", "k_2_dod", "reference_dod"],
        parameter_dict=dict(
            k_1=dict(value=1e-05, vary=True, min=0.0),
            k_2=dict(value=1.5, vary=True, min=0.0, max=10.0),
            x_ref=dict(value=1.0, vary=False),
        ),
    )
    model_register.register(
        base_models.exponential_soc_k_relation,
        "Calendar",
        "soc",
        "Exponential",
        parameter_names=["x", "k_soc_calendar", "reference_soc"],
        parameter_dict=dict(
            k=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            x_ref=dict(value=0.5, vary=False),
        ),
    )
    model_register.register(
        base_models.exponential_soc_k_relation,
        "Cycling",
        "soc",
        "Exponential",
        parameter_names=["x", "k_soc_cycling", "reference_soc"],
        parameter_dict=dict(
            k=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            x_ref=dict(value=0.5, vary=False),
        ),
    )
    model_register.register(
        base_models.exponential_high_soc_k_relation,
        "Calendar",
        "high_soc",
        "Exponential",
        parameter_names=["x", "k_high_soc_calendar", "reference_high_soc"],
        parameter_dict=dict(
            k=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            x_ref=dict(value=0.85, vary=False),
        ),
    )
    model_register.register(
        base_models.exponential_high_soc_k_relation,
        "Cycling",
        "high_soc",
        "Exponential",
        parameter_names=["x", "k_high_soc_cycling", "reference_high_soc"],
        parameter_dict=dict(
            k=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            x_ref=dict(value=0.85, vary=False),
        ),
    )
    model_register.register(
        base_models.linear_soc_k_relation,
        "Calendar",
        "soc",
        "Linear",
        parameter_names=[
            "x",
            "k_linear_1_soc_calendar",
            "k_linear_2_soc_calendar",
            "reference_soc",
        ],
        parameter_dict=dict(
            k_1=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            k_2=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            x_ref=dict(value=0.5, vary=False),
        ),
    )
    model_register.register(
        base_models.linear_soc_k_relation,
        "Cycling",
        "soc",
        "Linear",
        parameter_names=[
            "x",
            "k_linear_1_soc_cycling",
            "k_linear_2_soc_cycling",
            "reference_soc",
        ],
        parameter_dict=dict(
            k_1=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            k_2=dict(value=0.001, vary=True, min=-0.1, max=4.0),
            x_ref=dict(value=0.5, vary=False),
        ),
    )
    model_register.register(
        base_models.exponential_temperature_k_relation,
        "Cycling",
        "temperature",
        "Exponential",
        parameter_names=["x", "k_temperature_cycling", "reference_temperature"],
        parameter_dict=dict(
            k=dict(value=0.001, vary=True, min=0.00001, max=0.99),
            x_ref=dict(value=298.15, vary=False),
        ),
    )
    model_register.register(
        base_models.exponential_temperature_k_relation,
        "Calendar",
        "temperature",
        "Exponential",
        parameter_names=["x", "k_temperature_calendar", "reference_temperature"],
        parameter_dict=dict(
            k=dict(value=0.001, vary=True, min=0.00001, max=0.99),
            x_ref=dict(value=298.15, vary=False),
        ),
    )
    # Note: "Proportional" means proportional to x (not necessarily time), so to
    #    get same time model as used in the reference, you need to use the
    #    "Constant" model.
    model_register.register(
        base_models.proportional_time_k_relation,  # could use linear_time_k_relation here instead
        "Calendar",
        "time",
        "Proportional",
        parameter_names=[
            "x",
            "k_1_time_calendar",
            "k_2_time_calendar",
            "reference_time",
        ],
        parameter_dict=dict(
            k_1=dict(value=1e-10, vary=True, min=-0.1, max=4.0),
            k_2=dict(
                value=0.0, vary=False
            ),  # intercept is zero for a proportional function
            x_ref=dict(value=86400.0, vary=False),  # reference time set to 1 day
        ),
    )
    model_register.register(
        base_models.constant_time_k_relation,  # could use linear_time_k_relation here instead
        "Calendar",
        "time",
        "Constant",
        parameter_names=[
            "x",
            "k_1_time_calendar",
            "k_2_time_calendar",
            "reference_time",
        ],
        parameter_dict=dict(
            k_1=dict(value=5e-10, vary=True, min=-0.1, max=4.0),
            k_2=dict(value=0.0, vary=False),  # slope is zero for a constant function
            x_ref=dict(
                value=1.0, vary=False
            ),  # reference time set to 1 (needed for the automatic post-processing)
        ),
    )
    model_register.register(
        base_models.linear_time_k_relation,
        "Calendar",
        "time",
        "Linear",
        parameter_names=[
            "x",
            "k_1_time_calendar",
            "k_2_time_calendar",
            "reference_time",
        ],
        parameter_dict=dict(
            k_1=dict(value=5e-10, vary=True, min=-0.1, max=4.0),
            k_2=dict(value=1.0, vary=True, min=-0.1, max=4.0),
            x_ref=dict(value=3600 * 24, vary=False),
        ),
    )
    return model_register


if __name__ == "__main__":
    from rich import print

    model_register = run()
    print(model_register)
