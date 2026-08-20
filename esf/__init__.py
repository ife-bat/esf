"""esf - empirical stress factor degradation model for Li-ion batteries.

The two core workflows, importable straight from the package:

Fitting (data -> parameters)::

    import esf

    data = esf.SampleData()
    data.add_data(frame, data_type=esf.DataType.CALENDAR_VS_TEMPERATURE)
    data.calculate_life_fraction()
    prms = esf.get_example_params()
    esf.sei_fit_at_reference_conditions(prms, data.calendar_life_vs_temperature(...))

Simulation (parameters + drive cycle -> degradation)::

    result = esf.drive_cycle_degradation_calculator(drive_cycle_frame, prms, cycle_numbers=...)

Prediction from field data (operational trace + parameters -> degradation)::

    op = esf.OperationalData.from_field_dataframe(field_frame)   # SoC/C-rate/T/SoH over time
    result = op.predict(prms)

Everything else (individual fit classes, plotting, converters) stays importable
from its defining module.
"""

from esf.models import register_models

try:
    from importlib.metadata import version

    __version__ = version("esf")
except ImportError:
    __version__ = "unknown"

# the global model register; must exist before the fitting module is used
# (ESFParams resolves stress models against it at call time)
_mr = register_models.run()

from esf.io.data import OperationalData, SampleData, example_sample_data
from esf.models.fitting import (
    NonlinearFit,
    NonlinearMultiFit,
    degradation_rates_fit,
    dod_stress_factor_fit,
    sei_fit_at_reference_conditions,
    soc_stress_factor_fit,
    temperature_stress_factor_fit,
    time_stress_factor_calc,
)
from esf.settings.parameters import (
    DataType,
    ESFParams,
    Regime,
    get_example_params,
    get_example_params_from_original_repo,
)
from esf.simulations.cycles import (
    drive_cycle_001,
    drive_cycle_002,
    load_drive_cycle,
)
from esf.simulations.degradation import (
    calc_cycle_at_end_of_life,
    drive_cycle_degradation_calculator,
)
from esf.simulations.dst_cycle import (
    DSTCycleDeg,
    dst_cycles_experimental_data_v_lims,
    dst_cycles_from_experimental_data,
)
from esf.uncertainty import (
    ParameterEnsemble,
    ParameterUncertainty,
    simulate_with_uncertainty,
)

__all__ = [
    # parameters and vocabulary
    "DataType",
    "ESFParams",
    "Regime",
    "get_example_params",
    "get_example_params_from_original_repo",
    # data handling
    "SampleData",
    "OperationalData",
    "example_sample_data",
    # fitting
    "NonlinearFit",
    "NonlinearMultiFit",
    "degradation_rates_fit",
    "sei_fit_at_reference_conditions",
    "soc_stress_factor_fit",
    "temperature_stress_factor_fit",
    "dod_stress_factor_fit",
    "time_stress_factor_calc",
    # simulation
    "DSTCycleDeg",
    "calc_cycle_at_end_of_life",
    "drive_cycle_001",
    "drive_cycle_002",
    "drive_cycle_degradation_calculator",
    "dst_cycles_experimental_data_v_lims",
    "dst_cycles_from_experimental_data",
    "load_drive_cycle",
    # uncertainty propagation
    "ParameterEnsemble",
    "ParameterUncertainty",
    "simulate_with_uncertainty",
]
