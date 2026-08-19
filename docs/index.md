---
icon: lucide/battery-charging
---

# esf — empirical stress factor degradation model

`esf` models Li-ion battery capacity loss as a product of empirical **stress
factors** (temperature, state of charge, depth of discharge, time) scaling a
nonlinear SEI-driven fade. It is based on Xu et al., *"Modeling of Lithium-Ion
Battery Degradation for Cell Life Assessment"* (IEEE Trans. Smart Grid, 2018),
adapted at IFE.

It does three things:

<div class="grid cards" markdown>

- :material-chart-line: **Fit**

    Extract model parameters from calendar-aging and cycle-life data
    (staged SEI → rates → stress-factor fits).

    [:octicons-arrow-right-24: Fitting](workflows/fitting.md)

- :material-play-circle: **Simulate**

    Predict capacity loss for a drive cycle — rainflow cycle counting →
    stress factors → loss — given a set of parameters.

    [:octicons-arrow-right-24: Predicting degradation](workflows/prediction.md)

- :material-chart-bell-curve: **Quantify uncertainty**

    Propagate the fit covariance through the simulation as quantile bands.

    [:octicons-arrow-right-24: Uncertainty](workflows/uncertainty.md)

</div>

## The model in one paragraph

Capacity loss `L = 1 − SoH` follows a nonlinear "SEI" envelope wrapped around a
*linear* degradation rate `f`:

$$
L(x) = 1 - \alpha\,e^{-x\,\beta f} - (1-\alpha)\,e^{-x f}
$$

where `x` is time (calendar) or cycle number (cycling). Everything
condition-dependent lives in `f`, as a product of independent stress factors:

$$
\text{calendar: } f = S_t(t)\,S_\sigma(\text{SoC})\,S_T(T)
\qquad
\text{cycling: } f = S_\delta(\text{DoD})\,S_\sigma(\text{SoC})\,S_T(T)
$$

This structure dictates a **staged** fitting procedure — you cannot fit
everything from one data set. See the
[fitting architecture](fitting-architecture.md) for the full picture.

## Where to go next

- New here? [Install](install.md) then run the [quickstart](quickstart.md).
- Fitting your own data? [Fitting parameters](workflows/fitting.md).
- Have parameters and want a prediction? [Predicting degradation](workflows/prediction.md).
- Looking for a symbol or function? [Public API](reference/api.md) and the
  [units convention](reference/units.md).

!!! warning "Early release"

    This is a first public version. Expect bugs, incomplete features, and
    possible API changes. Please
    [open an issue](https://github.com/ife-bat/esf/issues) if something
    looks wrong.

!!! note "Status"

    All fitting stages (calendar SEI, rates, SoC/temperature/time/DoD stress
    factors) are implemented and numerically pinned by tests, along with
    uncertainty propagation and an end-to-end reproduction of the paper's DST
    degradation curves. See the project
    [README](https://github.com/ife-bat/esf#readme) and
    `CHANGELOG.md` for the current release.

## Contributors

<div class="contributors">
  <a href="https://github.com/JonathanFagerstrom">
    <img src="https://github.com/JonathanFagerstrom.png?size=100" alt="Jonathan Fagerström" />
    <span>Jonathan Fagerström</span>
  </a>
  <a href="https://github.com/JSHUAIFE">
    <img src="https://github.com/JSHUAIFE.png?size=100" alt="Jinsong Hua" />
    <span>Jinsong Hua</span>
  </a>
  <a href="https://github.com/juliawind">
    <img src="https://github.com/juliawind.png?size=100" alt="Julia Wind" />
    <span>Julia Wind</span>
  </a>
  <a href="https://github.com/jepegit">
    <img src="https://github.com/jepegit.png?size=100" alt="Jan Petter Mæhlen" />
    <span>Jan Petter Mæhlen</span>
  </a>
</div>

## Acknowledgments

This work was supported by Jernbanedirektoratet (the Norwegian Railway
Directorate) through the Europe's Rail project FP4-Rail4EARTH. The work
within the Europe's Rail project FP4-Rail4EARTH is supported by the
Europe's Rail Joint Undertaking and its members.
The project is funded by the European Union. Views and opinion
expressed are however those of the author(s) only and do not
necessarily reflect those of the European Union or the Europe's Rail
Joint Undertaking. Neither the European Union nor the granting
authority can be held responsible for them. 