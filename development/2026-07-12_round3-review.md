# Round-3 review (2026-07-12)

Review pass at the start of Round 3, after all Round-2 PRs (#11–#17) merged.
Focus: `esf/io/data.py` (the least-touched large module, 1642 loc) and the
decisions deferred from earlier rounds. Labels continue the numbering
(G* = round-3 findings).

## io.data findings

- **G1 — dead plotting code**: `SampleData._plot_data_deprecated` (~165 loc)
  is referenced nowhere. Delete.
- **G2 — `SampleData.get` is broken at the edges**: the cycling branch calls
  `self.cycle_life_vs_soc(...)`, which **does not exist** (latent
  `AttributeError`; unreachable today only because no `DataType` maps to
  cycling+SoC), and unmatched regime/z combinations fall through and silently
  return `None`. Should raise a clear `ValueError`.
- **G3 — selector duplication**: the four `*_life_vs_*` methods differ only
  in a (data_type, filter_key, cols, strict_cols) table. Readable as-is;
  fold into a table-driven `get()` only if/when a fifth selector appears.
- **G4 — plotting inside the data container**: `SampleData.plot_data`
  (~240 loc of matplotlib) belongs next to `fit_plotting`, not in the data
  class. Extract with a thin delegator (same pattern as F6).
- **G5 — dead stubs**: `Data.average_data`, `Data._glob_data`, and the
  no-op base `Data.plot_data` are `pass`-bodies with no callers. Delete.
- **G6 — untested persistence**: `save()` / `load_sample_data()` /
  `append_sample_data()` have no tests and no callers in the repo — the
  pickle/CSV round-trip may or may not work. Pin with a round-trip test
  (found working/broken status decides what happens next).
- **G7 — leftover raw print**: `normalize_cols` prints "normalization
  failed" before re-raising; route through logging.
- **G8 — `OperationalData` placeholder**: docstring-only sketch from the
  Round-1 TODO ("implement an OperationalData class"). Keep as a stub; its
  design belongs with the new-apps work (it is the input format for
  prediction).

## Deferred decisions now due

- **G9 — custom ESFParams round-trip inconsistency** (sharpens the Round-1
  `update()` note): `ESFParams.update()` promotes unknown keys to real
  dataclass fields (via `make_dataclass`) so `to_dict`/`save_json` include
  them — but `load_json` filters keys against `fields(ESFParams)`, so custom
  parameters are **dropped with a warning on load**. Save and load are
  asymmetric. Decide: either custom parameters survive the round trip, or
  `update()` stops accepting unknown keys. (The `make_dataclass` mutation
  itself can stay or go depending on that choice.)

## Feature work — needs owner input (not schedulable as cleanup)

- **DoD fit** (F7): needs modeling decisions (variable orientation,
  `deg_at_eol` rescaling, non-reference normalization) and reference data to
  validate against.
- **Uncertainty propagation**: `ESFParams` can hold `uncertainties` values
  but every pipeline converts to floats; propagating properly is a design
  effort (lmfit errors → parameters → simulation).
- **Reproducing the original paper / HEROES / LFP results**: needs the
  reference numbers and datasets, and acceptance tolerances.
- **New streamlit apps**: per the original brief, to be designed from
  scratch; will drive the final shape of the public API and
  `OperationalData`.

## Round-3 sessions

1. **R3-1 — io.data cleanup** (G1, G2, G4, G5, G6, G7): delete dead code,
   fix `get()`, extract plotting, pin persistence with a round-trip test.
2. **R3-2 — ESFParams custom-parameter round trip** (G9): make save/load
   symmetric (or reject custom keys), then close the `update()` decision.
3. **R3-3+ — feature work**: blocked on owner input listed above; pick up
   when data/decisions are available.
