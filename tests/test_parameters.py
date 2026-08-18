"""Tests for ESFParams and the model register (esf.settings.parameters,
esf.models.register_models)."""

import numpy as np
import pytest

from esf.settings.parameters import (
    MODEL_SETS,
    DataType,
    ESFParams,
    ESFParamsModelTypeExtension,
    Regime,
    get_example_params,
    get_example_params_from_original_repo,
)


@pytest.fixture
def prms():
    return get_example_params()


class TestBasics:
    def test_default_construction(self):
        p = ESFParams()
        assert p.reference_soc == 0.5
        assert p.degradation_model in MODEL_SETS

    def test_get_and_set(self, prms):
        assert prms.get("reference_soc") == pytest.approx(0.5)
        prms.set("reference_soc", 0.6)
        assert prms.reference_soc == pytest.approx(0.6)

    def test_get_all_preserves_order(self, prms):
        values = prms.get_all(["sei_alpha", "sei_beta"])
        assert values == [prms.sei_alpha, prms.sei_beta]

    def test_copy_is_independent(self, prms):
        clone = prms.copy()
        clone.sei_alpha = 999.0
        assert prms.sei_alpha != 999.0

    def test_compare(self, prms):
        assert prms.compare(prms.copy())
        other = prms.copy()
        other.sei_alpha = 999.0
        assert not prms.compare(other)

    def test_update_existing_key(self, prms):
        prms.update({"sei_alpha": 0.1})
        assert prms.sei_alpha == pytest.approx(0.1)

    def test_update_new_key_adds_custom_attribute(self, prms):
        prms.update({"k_custom_factor": 1.23})
        assert prms.get("k_custom_factor") == pytest.approx(1.23)

    def test_get_unit(self, prms):
        assert prms.get_unit("T") == prms.temperature_unit
        assert prms.get_unit("time") == prms.time_unit
        assert prms.get_unit("DoD") == prms.dod_unit
        assert prms.get_unit("soc") == prms.soc_unit


class TestSerialization:
    def test_json_string_round_trip(self, prms):
        restored = ESFParams.from_json(prms.to_json())
        assert prms.compare(restored)

    def test_json_file_round_trip(self, prms, tmp_path):
        file_path = tmp_path / "prms.json"
        prms.save_json(file_path)
        restored = ESFParams.load_json(file_path)
        assert prms.compare(restored)

    def test_json_file_round_trip_keeps_full_precision(self, prms, tmp_path):
        # regression test: pandas-based saving used to truncate to 10 digits,
        # turning k_1_time_calendar = 4.14e-10 into 4e-10
        file_path = tmp_path / "prms.json"
        prms.save_json(file_path)
        restored = ESFParams.load_json(file_path)
        assert restored.k_1_time_calendar == prms.k_1_time_calendar
        assert restored.deg_per_time_unit == prms.deg_per_time_unit

    def test_json_file_round_trip_with_uncertainties(self, prms, tmp_path):
        import uncertainties

        prms.sei_alpha = uncertainties.ufloat(0.0575, 0.003)
        file_path = tmp_path / "prms.json"
        prms.save_json(file_path)
        restored = ESFParams.load_json(file_path)
        assert isinstance(restored.sei_alpha, uncertainties.UFloat)
        assert restored.sei_alpha.nominal_value == pytest.approx(0.0575)
        assert restored.sei_alpha.std_dev == pytest.approx(0.003)

    def test_custom_parameters_survive_file_round_trip(self, prms, tmp_path):
        # regression test (G9): update() promoted custom keys to fields so
        # they were saved, but load_json used to drop them on load
        prms.update({"k_custom_factor": 1.23})
        file_path = tmp_path / "prms.json"
        prms.save_json(file_path)
        restored = ESFParams.load_json(file_path)
        assert restored.get("k_custom_factor") == pytest.approx(1.23)
        assert "k_custom_factor" in restored.custom_parameter_names
        assert restored.compare(prms)

    def test_custom_parameters_survive_string_round_trip(self, prms):
        prms.update({"k_custom_factor": 4.56})
        restored = ESFParams.from_json(prms.to_json())
        assert restored.get("k_custom_factor") == pytest.approx(4.56)

    def test_repeated_updates_all_survive(self, prms, tmp_path):
        # regression test: update() used to re-base the extended class on
        # ESFParams each call, silently dropping earlier custom fields from
        # serialization
        prms.update({"k_first": 1.0})
        prms.update({"k_second": 2.0})
        assert prms.to_dict()["k_first"] == pytest.approx(1.0)
        assert prms.to_dict()["k_second"] == pytest.approx(2.0)
        file_path = tmp_path / "prms.json"
        prms.save_json(file_path)
        restored = ESFParams.load_json(file_path)
        assert restored.get("k_first") == pytest.approx(1.0)
        assert restored.get("k_second") == pytest.approx(2.0)

    def test_undeclared_unknown_keys_still_dropped(self, tmp_path):
        # stale/legacy keys (not declared in custom_parameter_names) must
        # keep being skipped with a warning, as before
        import json

        prms = ESFParams()
        payload = {"Value": prms.to_dict()}
        payload["Value"]["deg_per_cyc"] = 0.123  # legacy renamed key
        file_path = tmp_path / "prms.json"
        file_path.write_text(json.dumps(payload))
        with pytest.warns(UserWarning, match="Ignoring unknown parameters"):
            restored = ESFParams.load_json(file_path)
        assert not hasattr(restored, "deg_per_cyc")

    def test_removed_deg_at_eol_key_still_loads(self, tmp_path):
        # deg_at_eol was a vestigial, never-computed field (removed A2); an
        # old parameter file that still carries it must load (the stale key is
        # dropped with a warning) rather than crash
        import json

        prms = ESFParams()
        payload = {"Value": prms.to_dict()}
        payload["Value"]["deg_at_eol"] = 0.188  # the removed field
        file_path = tmp_path / "prms.json"
        file_path.write_text(json.dumps(payload))
        with pytest.warns(UserWarning, match="Ignoring unknown parameters"):
            restored = ESFParams.load_json(file_path)
        assert not hasattr(restored, "deg_at_eol")
        assert restored.full_degradation_level == prms.full_degradation_level

    def test_to_frame_contains_all_fields(self, prms):
        df = prms.to_frame()
        assert set(prms.to_dict()) == set(df.index)

    def test_to_short_frame_uses_calendar_value_for_calendar_row(self):
        # regression test: the calendar row used to read the cycling value
        p = ESFParams(k_temperature_calendar=0.111, k_temperature_cycling=0.222)
        df = p.to_short_frame()
        assert df.loc["k_temperature_calendar", "Value"] == pytest.approx(0.111)


class TestUnitConversion:
    def test_temperature_kelvin_to_celsius(self):
        p = ESFParams(temperature_unit="K", reference_temperature=298.15)
        p.convert_units(unit="temperature", unit_from="K", unit_to="degC")
        assert p.temperature_unit == "degC"
        assert p.reference_temperature == pytest.approx(25.0)

    def test_no_op_when_units_are_equal(self):
        p = ESFParams(temperature_unit="K", reference_temperature=298.15)
        p.convert_units(unit="temperature", unit_from="K", unit_to="K")
        assert p.reference_temperature == pytest.approx(298.15)


class TestModelResolution:
    @pytest.mark.parametrize("model_set_name", sorted(MODEL_SETS))
    def test_all_models_in_model_sets_resolve(self, model_set_name):
        """Every stress factor of every model set must resolve to a callable
        with resolvable parameter names for the default ESFParams."""
        p = ESFParams(degradation_model=model_set_name)
        model_set = MODEL_SETS[model_set_name]
        extensions = {
            "Calendar": ESFParamsModelTypeExtension.CALENDAR,
            "Cycling": ESFParamsModelTypeExtension.CYCLING,
        }
        for regime, factors in model_set.items():
            for factor in factors:
                label = p.get(f"{factor}{extensions[regime]}")
                function = p.get_model_function(
                    regime, factor, label, raise_on_error=True
                )
                assert callable(function)
                parameter_names = p.get_model_esf_parameters(regime, factor, label)
                values = p.get_all(parameter_names)
                assert all(np.isfinite(float(v)) for v in values)

    def test_unknown_model_raises(self):
        with pytest.raises(AttributeError):
            ESFParams._get_model_object("Cycling", "dod", "DoesNotExist")

    def test_parameter_dict_copies_are_independent(self):
        # regression test: get_parameter_dict used to return a shallow copy,
        # so a fit adjusting x_ref corrupted the registry defaults for every
        # later fit
        item = ESFParams._get_model_object("Calendar", "temperature", "Exponential")
        first = item.get_parameter_dict()
        first["x_ref"]["value"] = -999.0
        second = item.get_parameter_dict()
        assert second["x_ref"]["value"] != -999.0

    def test_registered_parameter_names_are_unique(self):
        # regression test: the Cycling/Linear SoC model used to list
        # k_linear_1_soc_cycling twice
        from esf import _mr

        for name, item in _mr.models.items():
            assert len(item.esf_parameters) == len(
                set(item.esf_parameters)
            ), f"duplicate parameter names in {name}: {item.esf_parameters}"


class TestStressModelAccessors:
    def test_stress_model_label(self, prms):
        assert prms.stress_model_label("Calendar", "time") == prms.time_calendar_model
        assert prms.stress_model_label("Cycling", "dod") == prms.dod_cycling_model

    def test_stress_model_function_is_callable(self, prms):
        for regime in ("Calendar", "Cycling"):
            for factor in ("soc", "temperature"):
                assert callable(prms.stress_model_function(regime, factor))

    def test_stress_model_parameter_values_follow_attributes(self, prms):
        values = prms.stress_model_parameter_values("Calendar", "soc")
        assert values == [prms.k_soc_calendar, prms.reference_soc]

    def test_stress_models_covers_active_model_set(self, prms):
        from esf.settings.parameters import MODEL_SETS

        model_set = MODEL_SETS[prms.degradation_model]
        for regime in ("Calendar", "Cycling"):
            models = prms.stress_models(regime)
            assert [factor for factor, *_ in models] == model_set[regime]
            for _, function, values in models:
                assert callable(function)
                assert len(values) > 0

    def test_deprecated_aliases_still_resolve(self, prms):
        with pytest.warns(DeprecationWarning):
            function = prms.soc_calendar_stress_model_function
        assert function is prms.stress_model_function("Calendar", "soc")

        with pytest.warns(DeprecationWarning):
            values = prms.dod_stress_model_parameters_values
        assert values == prms.stress_model_parameter_values("Cycling", "dod")

        with pytest.warns(DeprecationWarning):
            item = prms.temperature_cycling_stress_model
        assert item is not None
        assert item.model_type == "temperature"

    def test_unknown_attribute_raises(self, prms):
        with pytest.raises(AttributeError):
            _ = prms.definitely_not_a_parameter

    def test_copy_still_works_with_getattr_fallback(self, prms):
        clone = prms.copy()
        assert clone.compare(prms)


class TestUnpackParameters:
    def test_stress_model_entries_are_plain_floats(self, prms):
        from esf.simulations.degradation import unpack_parameters

        unpacked = unpack_parameters(prms, use_model_sets=False)
        assert callable(unpacked["dod_stress_model"])
        for values in (
            unpacked["dod_stress_model_parameters"],
            unpacked["soc_calendar_stress_model_parameters"],
            unpacked["time_calendar_stress_model_parameters"],
        ):
            assert all(isinstance(v, np.float64) for v in values)

    @pytest.mark.parametrize(
        "example",
        [get_example_params, get_example_params_from_original_repo],
        ids=["paper", "original-repo"],
    )
    def test_model_sets_unpack_for_examples(self, example):
        from esf.simulations.degradation import unpack_parameters

        unpacked = unpack_parameters(example(), use_model_sets=True)
        for key in ("calendar_models", "cycling_models"):
            assert len(unpacked[key]) >= 3
            for factor_name, function, values in unpacked[key]:
                assert isinstance(factor_name, str)
                assert callable(function)
                assert all(isinstance(v, np.float64) for v in values)


class TestEnums:
    def test_data_type_z_column(self):
        assert DataType.CALENDAR_VS_TEMPERATURE.z == "T"
        assert DataType.CALENDAR_VS_SOC.z == "SoC"
        assert DataType.CYCLE_VS_DOD.z == "DoD"

    def test_data_type_regime(self):
        assert DataType.CALENDAR_VS_TEMPERATURE.regime == Regime.CALENDAR
        assert DataType.CYCLE_VS_DOD.regime == Regime.CYCLING
        assert DataType.OPERATIONAL_DATA.regime == Regime.OPERATIONAL

    def test_data_type_is_nlfd(self):
        assert DataType.CALENDAR_VS_TEMPERATURE.is_nlfd
        assert not DataType.CYCLE_VS_DOD.is_nlfd
