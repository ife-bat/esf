# Third-party notices

`esf` itself is distributed under the MIT license (see [LICENSE](LICENSE)).
This file records the third-party material it includes or depends on.

## Included in this repository

### `esf/external/peak_det/peak_det.py` — peak detection

Ported from the MATLAB `peakdet` function by **Eli Billauer**
(<http://billauer.co.il/peakdet.html>), released by its author into the
**public domain**: *"Explicitly not copyrighted. This function is released to
the public domain; Any use is allowed."*

## Runtime dependencies

| Package | License |
|---|---|
| [rainflow](https://pypi.org/project/rainflow/) | MIT |
| [numpy](https://numpy.org/) | BSD-3-Clause |
| [pandas](https://pandas.pydata.org/) | BSD-3-Clause |
| [scipy](https://scipy.org/) | BSD-3-Clause |
| [matplotlib](https://matplotlib.org/) | PSF-based (matplotlib license) |
| [seaborn](https://seaborn.pydata.org/) | BSD-3-Clause |
| [lmfit](https://lmfit.github.io/lmfit-py/) | BSD-3-Clause |
| [pint](https://pint.readthedocs.io/) | BSD-3-Clause |
| [uncertainties](https://uncertainties.readthedocs.io/) | BSD-3-Clause |
| [rich](https://rich.readthedocs.io/) | MIT |

All are permissive and compatible with MIT redistribution.

`rainflow` provides the ASTM E1049-85 cycle counting;
`esf/external/rainflow_adapter.py` is the adapter onto it. Earlier, internal
versions of this project vendored a GPL-3.0 rainflow implementation instead,
which is why the adapter exists — it was replaced before the public release so
that the project could be distributed under MIT.

## Scientific provenance

The model is an implementation of the degradation model published in:

> B. Xu, A. Oudalov, A. Ulbig, G. Andersson and D. S. Kirschen, "Modeling of
> Lithium-Ion Battery Degradation for Cell Life Assessment," *IEEE
> Transactions on Smart Grid*, vol. 9, no. 2, pp. 1131–1140, March 2018.
> doi: [10.1109/TSG.2016.2578950](https://doi.org/10.1109/TSG.2016.2578950)

The paper is **not** redistributed here. The aging data under `data/` are
values digitized from the figures of that publication; the reference numbers in
`development/dst-reproduction-reference.md` were likewise read off its figures.
Cite the paper if you use this model.
