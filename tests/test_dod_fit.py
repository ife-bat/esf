"""Tests for the DoD stress-factor fit.

The modeling decisions behind this fit are recorded in the DoD-fitting
decision record (decisions 1-8, summarised in the ``DoDSFfit`` docstrings and
``development/2026-07-13_round4-plan.md`` B1). The headline acceptance test
(decision 8) fits the published LMO cycle-life data and recovers the paper's
constants; a synthetic recovery test proves the fit converges from an
arbitrary start rather than just sitting on the registry defaults. The
chemistry-driven model selection (decision 6) and the non-reference
stress-removal path (decision 5, Xu eqs. 20/31) are covered here too.
"""

import numpy as np
import pandas as pd
import pytest

import esf
from esf.models.base_models import (
    empirical_dod_k_relation,
    quadratic_dod_k_relation,
)
from esf.models.fitting import DoDSFfit, dod_stress_factor_fit
from esf.settings.parameters import DATA_FOLDER, DataType

# published LMO DoD constants (Xu et al. 2018), also the registry defaults
PUBLISHED = {"k_1_dod": 1.4e5, "k_2_dod": -0.501, "k_3_dod": -1.23e5}


def _load_cycle_life(chemistry):
    path = (
        DATA_FOLDER
        / "Ageing_Data_Org"
        / "cycling_degradation"
        / f"cycle_nb_at_80_SoH_{chemistry}.csv"
    )
    return pd.read_csv(path).rename(columns={f"Cycle_Nb_{chemistry}": "N"})


def load_lmo_cycle_life():
    return _load_cycle_life("LMO")


class TestLMOAcceptance:
    """Decision 8: fitting cycle_nb_at_80_SoH_LMO.csv recovers the published
    constants within ~2%."""

    def test_recovers_published_constants(self):
        prms = esf.ESFParams(battery_chemistry="LMO")  # Empirical DoD model
        # perturb the start: the registry initials *are* the published values,
        # so the fit must actually converge, not just stay put
        dod_stress_factor_fit(
            prms,
            load_lmo_cycle_life(),
            data_type=DataType.CYCLE_VS_DOD,
            parameter_overrides={
                "k_1": {"value": 8e4},
                "k_2": {"value": -0.40},
                "k_3": {"value": -8e4},
            },
        )
        for name, published in PUBLISHED.items():
            fitted = getattr(prms, name)
            assert fitted == pytest.approx(published, rel=0.02), name

    def test_eol_loss_is_full_degradation_level(self):
        # decision 3: the EOL loss used to invert N -> rate is 0.20, not
        # deg_at_eol = 0.188. Fitting with the wrong value shifts k_1/k_3.
        df = load_lmo_cycle_life()
        prms = esf.ESFParams()
        prms.full_degradation_level = 0.188  # the wrong value on purpose
        dod_stress_factor_fit(prms, df, data_type=DataType.CYCLE_VS_DOD)
        # k_2 (the shape) is loss-independent, but k_1 drifts well past 2%
        assert prms.k_2_dod == pytest.approx(PUBLISHED["k_2_dod"], rel=0.02)
        assert abs(prms.k_1_dod - PUBLISHED["k_1_dod"]) / PUBLISHED["k_1_dod"] > 0.03


class TestSyntheticRecovery:
    """Generate cycle life from known constants, fit, recover them exactly."""

    def test_round_trip(self):
        k1_true, k2_true, k3_true = 1.6e5, -0.55, -1.3e5
        loss_eol = 0.2
        dod = np.linspace(0.05, 0.95, 40)
        s_dod = empirical_dod_k_relation(dod, k1_true, k2_true, k3_true)
        cycle_life = loss_eol / s_dod
        frame = pd.DataFrame({"DoD": dod, "N": cycle_life})

        prms = esf.ESFParams()
        assert prms.full_degradation_level == loss_eol
        dod_stress_factor_fit(
            prms,
            frame,
            data_type=DataType.CYCLE_VS_DOD,
            parameter_overrides={
                "k_1": {"value": 1e5},
                "k_2": {"value": -0.4},
                "k_3": {"value": -1e5},
            },
        )
        assert prms.k_1_dod == pytest.approx(k1_true, rel=1e-3)
        assert prms.k_2_dod == pytest.approx(k2_true, rel=1e-3)
        assert prms.k_3_dod == pytest.approx(k3_true, rel=1e-3)


class TestFitMechanics:
    def test_fitted_parameters_read_side(self):
        prms = esf.ESFParams()
        fit = dod_stress_factor_fit(
            prms, load_lmo_cycle_life(), data_type=DataType.CYCLE_VS_DOD
        )
        fitted = fit.fitted_parameters()
        assert set(fitted) == {"k_1_dod", "k_2_dod", "k_3_dod", "reference_dod"}

    def test_only_cycling_regime_supported(self):
        frame = load_lmo_cycle_life()
        with pytest.raises(ValueError, match="only supported for cycling"):
            DoDSFfit(frame, esf.ESFParams(), regime="calend")


class TestChemistryModelSelection:
    """Decision 6: the DoD stress-factor form follows the chemistry unless the
    user overrides it explicitly."""

    @pytest.mark.parametrize(
        "chemistry, label",
        [
            ("LMO", "Empirical"),
            ("LFP", "Exponential"),
            ("NMC", "Quadratic"),
            ("not specified", "Empirical"),
            ("something else", "Empirical"),
        ],
    )
    def test_label_from_chemistry(self, chemistry, label):
        prms = esf.ESFParams(battery_chemistry=chemistry)
        assert prms.dod_cycling_model == label

    def test_explicit_label_wins(self):
        prms = esf.ESFParams(battery_chemistry="NMC", dod_cycling_model="Empirical")
        assert prms.dod_cycling_model == "Empirical"

    def test_selection_survives_json_round_trip(self):
        prms = esf.ESFParams(battery_chemistry="NMC")
        restored = esf.ESFParams.from_json(prms.to_json())
        assert restored.dod_cycling_model == "Quadratic"

    def test_nmc_uses_quadratic_function(self):
        prms = esf.ESFParams(battery_chemistry="NMC")
        assert (
            prms.stress_model_function("Cycling", "dod").__name__
            == "quadratic_dod_k_relation"
        )


class TestNMCQuadratic:
    """Decision 6: the NMC quadratic (power-law) DoD form fits the NMC
    cycle-life data and round-trips synthetic data."""

    def test_fits_real_nmc_data(self):
        prms = esf.ESFParams(battery_chemistry="NMC")
        fit = dod_stress_factor_fit(
            prms, _load_cycle_life("NMC"), data_type=DataType.CYCLE_VS_DOD
        )
        # k_3_dod is unused by the quadratic form
        assert set(fit.fitted_parameters()) == {
            "k_1_dod",
            "k_2_dod",
            "reference_dod",
        }
        # the power-law recovers the NMC data essentially exactly (R^2 ~ 1)
        df = _load_cycle_life("NMC").sort_values("DoD")
        measured = prms.full_degradation_level / df["N"].to_numpy()
        modelled = quadratic_dod_k_relation(
            df["DoD"].to_numpy(), prms.k_1_dod, prms.k_2_dod, prms.reference_dod
        )
        ss_res = np.sum((measured - modelled) ** 2)
        ss_tot = np.sum((measured - measured.mean()) ** 2)
        assert 1 - ss_res / ss_tot > 0.999
        assert prms.k_2_dod > 0  # rate grows with DoD

    def test_synthetic_round_trip(self):
        k1_true, k2_true = 8.0e-5, 1.7
        loss_eol = 0.2
        dod = np.linspace(0.05, 0.95, 40)
        s_dod = quadratic_dod_k_relation(dod, k1_true, k2_true)
        frame = pd.DataFrame({"DoD": dod, "N": loss_eol / s_dod})

        prms = esf.ESFParams(battery_chemistry="NMC")
        dod_stress_factor_fit(
            prms,
            frame,
            data_type=DataType.CYCLE_VS_DOD,
            parameter_overrides={"k_1": {"value": 1e-5}, "k_2": {"value": 1.0}},
        )
        assert prms.k_1_dod == pytest.approx(k1_true, rel=1e-3)
        assert prms.k_2_dod == pytest.approx(k2_true, rel=1e-3)


class TestNonReferencePath:
    """Decision 5: at non-reference conditions the temperature, SoC and
    calendar-time stress factors are stripped out before fitting the DoD form
    (Xu et al. 2018, eqs. 20/31)."""

    @staticmethod
    def _make_non_reference_frame(prms, k1, k2, k3, seed=1):
        """Forward-model cycle life at scattered conditions (eq. 20)."""
        rng = np.random.default_rng(seed)
        dod = np.linspace(0.05, 0.95, 30)
        temperature = prms.reference_temperature + rng.uniform(-10, 20, dod.size)
        soc = np.clip(prms.reference_soc + rng.uniform(-0.3, 0.4, dod.size), 0.05, 0.99)
        t_cycle = rng.uniform(1800, 7200, dod.size)

        def sf(regime, factor, x):
            func = prms.stress_model_function(regime, factor)
            values = prms.stress_model_parameter_values(regime, factor)
            return func(np.asarray(x, dtype=float), *values)

        s_temperature = sf("Cycling", "temperature", temperature)
        s_soc = sf("Cycling", "soc", soc)
        s_time = sf("Calendar", "time", t_cycle)
        s_dod = empirical_dod_k_relation(dod, k1, k2, k3)
        f_d1 = (s_dod + s_time) * s_soc * s_temperature  # eq. 20
        cycle_life = prms.full_degradation_level / f_d1
        return pd.DataFrame(
            {"DoD": dod, "N": cycle_life, "T": temperature, "SoC": soc, "t_cycle": t_cycle}
        )

    def test_recovers_constants_after_stress_removal(self):
        k1_true, k2_true, k3_true = 1.4e5, -0.501, -1.23e5
        prms = esf.ESFParams(battery_chemistry="LMO")
        frame = self._make_non_reference_frame(prms, k1_true, k2_true, k3_true)
        dod_stress_factor_fit(
            prms,
            frame,
            data_type=DataType.CYCLE_VS_DOD,
            is_at_reference=False,
            parameter_overrides={
                "k_1": {"value": 1e5},
                "k_2": {"value": -0.4},
                "k_3": {"value": -1e5},
            },
        )
        assert prms.k_1_dod == pytest.approx(k1_true, rel=1e-3)
        assert prms.k_2_dod == pytest.approx(k2_true, rel=1e-3)
        assert prms.k_3_dod == pytest.approx(k3_true, rel=1e-3)

    def test_ignoring_stress_removal_biases_the_fit(self):
        # if the same non-reference data is fitted *as if* at reference, the
        # unremoved temperature/SoC/time spread corrupts the constants
        k1_true, k2_true, k3_true = 1.4e5, -0.501, -1.23e5
        prms = esf.ESFParams(battery_chemistry="LMO")
        frame = self._make_non_reference_frame(prms, k1_true, k2_true, k3_true)
        dod_stress_factor_fit(
            prms, frame, data_type=DataType.CYCLE_VS_DOD, is_at_reference=True
        )
        assert abs(prms.k_1_dod - k1_true) / k1_true > 0.05

    def test_missing_columns_raise(self):
        frame = load_lmo_cycle_life()  # only DoD and N
        with pytest.raises(ValueError, match="t_cycle"):
            DoDSFfit(frame, esf.ESFParams(), is_at_reference=False).fit()
