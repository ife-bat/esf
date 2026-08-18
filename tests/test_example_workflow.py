"""Executes the documentation example so docs/fitting-architecture.md cannot rot.

scripts/examples/full_fitting_workflow.py generates synthetic aging data,
runs the complete two-stage fitting procedure, asserts that every fitted
constant matches the value that generated the data, and simulates a drive
cycle with the result.
"""

import importlib.util
import pathlib


def test_full_fitting_workflow_example():
    script = (
        pathlib.Path(__file__).parent.parent
        / "scripts"
        / "examples"
        / "full_fitting_workflow.py"
    )
    spec = importlib.util.spec_from_file_location("full_fitting_workflow", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prms, simulated = module.main()

    # main() already asserts parameter recovery; check the simulation output
    assert not simulated.empty
    assert (simulated["loss"] >= 0).all()
    assert (simulated["loss"].diff().dropna() > 0).all()
