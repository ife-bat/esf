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

    def test_dst_reference_data_is_reachable_top_level(self):
        """The DST loader and its SoC windows are public.

        The UI consumes both; before 0.2.0 they were only importable from
        esf.simulations.dst_cycle, i.e. internal by the versioning policy while
        being public in practice.
        """
        import esf

        for name in (
            "dst_cycles_from_experimental_data",
            "dst_cycles_experimental_data_v_lims",
            "drive_cycle_002",
        ):
            assert name in esf.__all__

    def test_dst_loader_defaults_to_the_bundled_data(self):
        import esf

        frame = esf.dst_cycles_from_experimental_data()
        assert list(frame.columns) == ["N", "SoH", "label"]
        assert frame["label"].nunique() == 7
        assert not frame.empty

    def test_dst_loader_accepts_a_string_path(self):
        import esf
        from esf.settings.parameters import DST_DATA_FOLDER

        frame = esf.dst_cycles_from_experimental_data(str(DST_DATA_FOLDER))
        assert frame["label"].nunique() == 7

    def test_dst_windows_match_the_loaded_labels(self):
        import esf

        v_min, v_max = esf.dst_cycles_experimental_data_v_lims()
        assert len(v_min) == len(v_max) == 7
        labels = set(esf.dst_cycles_from_experimental_data()["label"])
        for lo, hi in zip(v_min, v_max, strict=True):
            assert f"{lo}-{hi} @ 20°C" in labels

    def test_dst_loader_is_quiet(self, capsys):
        """It used to print a dataframe head on every call."""
        import esf

        esf.dst_cycles_from_experimental_data()
        assert capsys.readouterr().out == ""

    def test_both_example_drive_cycles_have_the_same_shape(self):
        import esf

        one = esf.drive_cycle_001(verbose=False)
        two = esf.drive_cycle_002(verbose=False)
        assert list(one.columns) == list(two.columns)
        assert not two.empty

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
