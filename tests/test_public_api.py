"""Tests for the package-level API surface and the finisher-session fixes:
the public exports, the explicit parameter_overrides API (F5), and the
generator-based Data iterator (D7)."""

import numpy as np
import pandas as pd
import pytest


class TestPublicApi:
    def test_all_exports_resolve(self):
        import esf

        assert esf.__version__
        for name in esf.__all__:
            assert getattr(esf, name) is not None, name

    def test_core_workflow_names_are_exported(self):
        import esf

        # the two workflows the package exists for must be reachable top-level
        for name in (
            "SampleData",
            "ESFParams",
            "get_example_params",
            "sei_fit_at_reference_conditions",
            "degradation_rates_fit",
            "drive_cycle_degradation_calculator",
        ):
            assert name in esf.__all__


class TestParameterOverrides:
    """The explicit parameter_overrides API (F5). Strict: typos raise."""

    def soc_frame(self):
        from esf.models.base_models import exponential_soc_k_relation

        soc = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
        rates = 3.6e-5 * exponential_soc_k_relation(soc, 1.3)
        return pd.DataFrame({"SoC": soc, "deg_rate": rates})

    def test_full_spec_override(self):
        import esf

        prms = esf.get_example_params()
        fit = esf.soc_stress_factor_fit(
            prms,
            self.soc_frame(),
            data_type=esf.DataType.CALENDAR_VS_SOC,
            parameter_overrides={"k": {"value": 0.5, "vary": False}},
        )
        # k was fixed, so the (wrong) start value must survive the fit
        assert fit.fit_result.get_fit().best_values["k"] == pytest.approx(0.5)

    def test_attribute_override(self):
        import esf

        prms = esf.get_example_params()
        fit = esf.soc_stress_factor_fit(
            prms,
            self.soc_frame(),
            data_type=esf.DataType.CALENDAR_VS_SOC,
            parameter_overrides={"k__max": 0.9},
        )
        # true k is 1.3; the tightened bound must cap the fit
        assert fit.fit_result.get_fit().best_values["k"] == pytest.approx(0.9, abs=1e-6)

    def test_unknown_parameter_raises(self):
        import esf

        prms = esf.get_example_params()
        with pytest.raises(ValueError, match="unknown fit parameter 'k_typo'"):
            esf.soc_stress_factor_fit(
                prms,
                self.soc_frame(),
                data_type=esf.DataType.CALENDAR_VS_SOC,
                parameter_overrides={"k_typo": {"value": 1.0}},
            )

    def test_non_dict_full_spec_raises(self):
        import esf

        prms = esf.get_example_params()
        with pytest.raises(ValueError, match="must be a dict"):
            esf.soc_stress_factor_fit(
                prms,
                self.soc_frame(),
                data_type=esf.DataType.CALENDAR_VS_SOC,
                parameter_overrides={"k": 1.0},
            )

    def test_legacy_kwargs_path_still_lenient(self):
        # the old fit(**{"x_ref": {...}}) style must keep working and keep
        # ignoring unknown keys (they may be unrelated kwargs)
        import esf
        from esf.models.fitting import SoCSFfit

        prms = esf.get_example_params()
        fit = SoCSFfit(self.soc_frame(), prms, regime="calend")
        fit.fit(k=dict(value=0.001, vary=True, min=-0.1, max=4.0), not_a_param=123)
        assert prms.k_soc_calendar == pytest.approx(1.3, rel=1e-4)


class TestDataIterator:
    """Data.__iter__ is a generator: independent, nestable iterations (D7)."""

    def _sample_data(self):
        import esf

        t = np.linspace(1, 100, 5)
        frames = []
        for subset, soh_offset in (("A", 0.0), ("B", 0.05)):
            frame = pd.DataFrame({"t": t, "SoH": 0.99 - 0.001 * t - soh_offset})
            frame["T"] = 298.15
            frame["subset"] = subset
            frames.append(frame)
        data = esf.SampleData()
        data.add_data(
            pd.concat(frames), data_type=esf.DataType.CALENDAR_VS_TEMPERATURE
        )
        return data

    def test_iterates_all_uids(self):
        data = self._sample_data()
        uids = [entry["uid"] for entry in data]
        assert len(uids) == 2
        assert len(set(uids)) == 2

    def test_nested_iteration_is_independent(self):
        data = self._sample_data()
        pairs = [(a["uid"], b["uid"]) for a in data for b in data]
        # 2 uids -> the full 2x2 product; the old stateful iterator lost the
        # outer position after the inner loop exhausted it
        assert len(pairs) == 4

    def test_empty_data_iterates_nothing(self):
        import esf

        assert list(esf.SampleData()) == []
