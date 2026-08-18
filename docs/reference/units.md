# Units convention

Internal units, everywhere, **after data ingestion**:

| Quantity | Unit |
|---|---|
| time | seconds |
| temperature | K |
| SoC / DoD / SoH / loss | fraction (0–1) |
| rate | C-rate |

Conversion happens only in two places:

- **on the way in** — `esf.io`: `SampleData.add_data` takes `time_unit=` /
  `temperature_unit=`, and the selectors re-unit to the internal convention;
  `OperationalData.from_field_dataframe` converts a raw field trace.
- **on the way out** — plotting and reporting.

Reference values such as the calendar-time reference
(`x_ref = 86 400 s = 1 day`) are **model parameters**, not hidden unit changes.

All pint quantities must come from the single shared registry:

```python
from esf.settings.units import ureg, Q_
```

!!! warning "Temperature is kelvin, always"

    The exponential temperature stress model is **not** invariant under a unit
    change — its `x_ref / x` factor differs between a degC and a K
    parameterization. Mixing the paper's kelvin-parameterized `k_temperature`
    constants with a degC reference gives wrong stress factors. Keep
    temperatures in kelvin.
