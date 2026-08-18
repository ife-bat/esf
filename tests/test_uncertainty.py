"""Tests for uncertainty propagation (esf.uncertainty).

Covers Tier 0 (capturing a fit's covariance) and Tier 1 (the Monte Carlo
ensemble + band), including a cross-check of the sampler against first-order
error propagation from the `uncertainties` package.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import uncertainties
from uncertainties import umath

import esf
from esf.models.base_models import exponential_temperature_k_relation
from esf.uncertainty import ParameterEnsemble, ParameterUncertainty


# ---------------------------------------------------------------------------
# ParameterUncertainty mechanics
# ---------------------------------------------------------------------------
class TestParameterUncertainty:
    def _example(self):
        return ParameterUncertainty(
            names=["a", "b"],
            nominal=np.array([1.0, 10.0]),
            covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
            bounds=[(0.0, 2.0), (0.0, 20.0)],
        )

    def test_std_is_sqrt_diagonal(self):
        u = self._example()
        assert u.std["a"] == pytest.approx(0.2)
        assert u.std["b"] == pytest.approx(0.3)

    def test_report_columns(self):
        report = self._example().report()
        assert list(report.columns) == ["value", "std", "rel_std"]
        assert report.loc["a", "rel_std"] == pytest.approx(0.2)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ParameterUncertainty(["a"], np.array([1.0, 2.0]), np.array([[1.0]]))

    def test_block_diagonal_merge(self):
        a = ParameterUncertainty(["a"], np.array([1.0]), np.array([[0.04]]))
        b = ParameterUncertainty(["b"], np.array([2.0]), np.array([[0.09]]))
        merged = a + b
        assert merged.names == ["a", "b"]
        np.testing.assert_allclose(merged.covariance, [[0.04, 0.0], [0.0, 0.09]])

    def test_merge_rejects_overlap(self):
        a = ParameterUncertainty(["a"], np.array([1.0]), np.array([[0.04]]))
        with pytest.raises(ValueError, match="appear in both"):
            _ = a + a

    def test_sample_statistics_match(self):
        u = self._example()
        rng = np.random.default_rng(0)
        samples = u.sample(50_000, rng)
        np.testing.assert_allclose(samples.mean(axis=0), u.nominal, atol=0.02)
        np.testing.assert_allclose(np.cov(samples.T), u.covariance, atol=0.01)

    def test_sample_respects_bounds(self):
        u = ParameterUncertainty(
            ["a"], np.array([0.05]), np.array([[0.1**2]]), bounds=[(0.0, np.inf)]
        )
        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # near-bound -> rejection warning
            samples = u.sample(5_000, rng)
        assert np.all(samples >= 0.0)


# ---------------------------------------------------------------------------
# Tier 0: capturing a fit's covariance
# ---------------------------------------------------------------------------
class TestExtractUncertainty:
    def _noisy_temperature_fit(self, noise=0.04, seed=0):
        rng = np.random.default_rng(seed)
        temperature = np.array([288.15, 298.15, 308.15, 318.15, 328.15])
        rates = 4.5e-4 * exponential_temperature_k_relation(
            temperature, 0.0693, x_ref=298.15
        )
        rates = rates * (1 + rng.normal(0, noise, len(temperature)))
        prms = esf.get_example_params()
        fit = esf.temperature_stress_factor_fit(
            prms,
            pd.DataFrame({"T": temperature, "deg_rate": rates}),
            data_type=esf.DataType.CALENDAR_VS_TEMPERATURE,
        )
        return fit

    def test_captures_named_covariance(self):
        u = self._noisy_temperature_fit().parameter_uncertainty()
        assert u.names == ["k_temperature_calendar"]
        assert u.std["k_temperature_calendar"] > 0
        # the fitted value is the nominal
        assert u.nominal[0] == pytest.approx(0.0693, rel=0.1)

    def test_bounds_are_captured(self):
        u = self._noisy_temperature_fit().parameter_uncertainty()
        low, high = u.bounds[0]
        assert low < u.nominal[0] < high


# ---------------------------------------------------------------------------
# sampler cross-check against first-order propagation
# ---------------------------------------------------------------------------
def test_sampler_matches_first_order_propagation():
    # S(k) = exp(k * delta); first-order std_S = |delta * exp(k0*delta)| * std_k
    k0, std_k, delta = 1.3, 0.05, 0.3
    u = ParameterUncertainty(["k"], np.array([k0]), np.array([[std_k**2]]))
    rng = np.random.default_rng(0)
    samples = u.sample(40_000, rng)[:, 0]
    mc_std = np.exp(samples * delta).std()

    analytic = uncertainties.ufloat(k0, std_k)
    first_order_std = umath.exp(analytic * delta).std_dev

    assert mc_std == pytest.approx(first_order_std, rel=0.05)


# ---------------------------------------------------------------------------
# Tier 1: ensemble + simulate_with_uncertainty
# ---------------------------------------------------------------------------
class TestEnsembleAndSimulation:
    def _sei_uncertainty(self, prms, rel=0.05):
        names = ["sei_alpha", "sei_beta", "k_1_dod"]
        nominal = np.array([prms.sei_alpha, prms.sei_beta, prms.k_1_dod])
        covariance = np.diag((rel * nominal) ** 2)
        bounds = [(0.0, 1.0), (0.0, 1e4), (0.0, np.inf)]
        return ParameterUncertainty(names, nominal, covariance, bounds)

    def test_ensemble_overwrites_only_sampled_params(self):
        prms = esf.get_example_params()
        u = self._sei_uncertainty(prms)
        ensemble = ParameterEnsemble(prms, u, n=10, seed=0)
        members = list(ensemble)
        assert len(members) == 10
        for member in members:
            assert member.sei_alpha != prms.sei_alpha or member is not prms
            # a parameter not in the uncertainty is untouched
            assert member.k_soc_calendar == prms.k_soc_calendar

    def test_ensemble_is_deterministic(self):
        prms = esf.get_example_params()
        u = self._sei_uncertainty(prms)
        a = ParameterEnsemble(prms, u, n=100, seed=42)
        b = ParameterEnsemble(prms, u, n=100, seed=42)
        np.testing.assert_allclose(a.samples, b.samples)

    def test_simulation_band_is_ordered_and_grows(self):
        prms = esf.get_example_params()
        ensemble = ParameterEnsemble(prms, self._sei_uncertainty(prms), n=300, seed=0)
        drive_cycle = esf.drive_cycle_001(verbose=False)
        cycle_numbers = np.linspace(1, 1000, 5)
        band = esf.simulate_with_uncertainty(
            esf.drive_cycle_degradation_calculator,
            ensemble,
            sim_kwargs=dict(df=drive_cycle, cycle_numbers=cycle_numbers),
            value_columns=["soh"],
            index_columns=["cycle_number"],
        )
        assert list(band.columns) == [
            "cycle_number",
            "soh_p05",
            "soh_p50",
            "soh_p95",
        ]
        assert (band["soh_p05"] <= band["soh_p50"]).all()
        assert (band["soh_p50"] <= band["soh_p95"]).all()
        width = (band["soh_p95"] - band["soh_p05"]).to_numpy()
        # widen with cycling (after the first, ~zero-loss point)
        assert width[-1] > width[1] > 0

    def test_simulation_band_is_deterministic(self):
        prms = esf.get_example_params()
        u = self._sei_uncertainty(prms)
        drive_cycle = esf.drive_cycle_001(verbose=False)
        kwargs = dict(df=drive_cycle, cycle_numbers=np.linspace(1, 500, 3))
        band1 = esf.simulate_with_uncertainty(
            esf.drive_cycle_degradation_calculator,
            ParameterEnsemble(prms, u, n=100, seed=7),
            sim_kwargs=kwargs,
            value_columns=["loss"],
        )
        band2 = esf.simulate_with_uncertainty(
            esf.drive_cycle_degradation_calculator,
            ParameterEnsemble(prms, u, n=100, seed=7),
            sim_kwargs=kwargs,
            value_columns=["loss"],
        )
        pd.testing.assert_frame_equal(band1, band2)

    def test_variable_length_outputs_need_align_on(self):
        # a sim function whose output length depends on the parameters
        prms = esf.get_example_params()
        u = ParameterUncertainty(
            ["sei_alpha"], np.array([prms.sei_alpha]), np.array([[0.005**2]]),
            bounds=[(0.0, 1.0)],
        )
        ensemble = ParameterEnsemble(prms, u, n=20, seed=0)

        def variable_sim(prms):
            length = 5 + int(prms.sei_alpha * 100)
            x = np.linspace(0, 1, length)
            return pd.DataFrame({"x": x, "y": prms.sei_alpha * x})

        with pytest.raises(ValueError, match="differ in length"):
            esf.simulate_with_uncertainty(variable_sim, ensemble)

        band = esf.simulate_with_uncertainty(
            variable_sim, ensemble, align_on="x", value_columns=["y"]
        )
        assert "y_p50" in band.columns
        assert (band["y_p05"] <= band["y_p95"]).all()


def test_public_api_exports():
    for name in ("ParameterUncertainty", "ParameterEnsemble", "simulate_with_uncertainty"):
        assert name in esf.__all__
        assert getattr(esf, name) is not None
