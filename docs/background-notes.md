# Background notes

Reference material moved out of the README: how the rainflow counting feeds the
stress factors, and what the data sets from the original publication contain.

## Provenance

The model is based on (<https://ieeexplore.ieee.org/document/7488267>):

> "Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment" by
> Bolun Xu, Alexandre Oudalov, Andreas Ulbig, Göran Andersson, and Daniel S.
> Kirschen, IEEE Transactions on Smart Grid, vol. 9, no. 2, March 2018.

See also this related code repository:
<https://github.com/DaniCelis25/lithium_ion_battery_degradation_models>.
Modified and adjusted by Jinsong Hua with IFE data. The papers themselves are
not redistributed here -- see the DOI above.

## Rainflow cycle counting

Input: the SoC profile.

Output: the rainflow cycle count —

1. cycle amplitude
2. cycle mean value
3. cycle number (0.5 for a half cycle, 1 for a full cycle)
4. cycle begin time
5. cycle end time

Estimating stress factors from the count:

1. the DoD of the i-th cycle (δ_i) is twice the i-th rainflow cycle amplitude
2. the average SoC of the i-th cycle (σ_i) is the i-th rainflow cycle mean value
3. the average cycle temperature of the i-th cycle (T_c,i) is the mean
   temperature between the start and end times of the i-th rainflow cycle
4. the average profile SoC (σ) is the mean value of the rainflow cycle mean values
5. the average profile temperature (T_c) is the mean value of the temperature profile

The implementation is `esf.models.cycle_counting_algorithm.CycleCounter`
(wrapping the vendored peak-detection and rainflow algorithms in
`esf/external/`); its behaviour on synthetic profiles is pinned in
`tests/test_cycle_counting.py`.

## Dynamic Stress Test (DST) data from the original publication

- Illustrate the battery's performance in mixed-cycle operations.
- For each test, the cell starts at a set SoC level and the DST profile is
  applied repetitively until the set stop level is reached.
- The cell is then recharged back to the starting level at a 1 C-rate to
  finish one test cycle.
- Only the State of Health (SoH) vs. cycle number is provided; the underlying
  profile must be simulated to reproduce the results (see
  `esf.simulations.dst_cycle.DSTCycleDeg`).
- Data sets (in `esf/data/Ageing_Data_Org/DST_cycles/`), with start/stop SoC in
  percent; the test room temperature is assumed 20 °C (293.15 K) based on the
  figure in the publication:

1. `DST_25_100.csv`: 100% to 25% SoC
2. `DST_40_100.csv`: 100% to 40% SoC
3. `DST_50_100.csv`: 100% to 50% SoC
4. `DST_25_85.csv`: 85% to 25% SoC
5. `DST_25_75.csv`: 75% to 25% SoC
6. `DST_45_75.csv`: 75% to 45% SoC
7. `DST_65_75.csv`: 75% to 65% SoC

C-rates (`c_rate`) and time-steps (`delta_t` in seconds) were obtained by
Jinsong Hua from the publication:

```python
delta_t = np.array(
    [18, 28, 12, 8, 16, 24, 12, 8, 16, 24, 12, 8, 16, 36, 8, 24, 8, 32, 8, 42]
)
c_rate = np.array(
    [0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -8, -5, 2, -2, 4, 0]
)
# only the discharge steps; re-charging is performed at constant current of 1C
```

## Other aging data from the original publication

In `esf/data/Ageing_Data_Org/`:

- `calendar_degradation/calend_deg_at_25_deg.csv` — calendar aging at 25 °C
  for several SoC levels
- `calendar_degradation/calend_deg_at_50_SoC.csv` — calendar aging at 50% SoC
  for several temperatures
- `cycling_degradation/cycle_nb_at_80_SoH_{LMO,LFP,NMC}.csv` — cycle life to
  80% SoH vs. DoD per chemistry
