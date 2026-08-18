# Installation

`esf` requires Python **3.12+**.

## With uv (recommended)

[uv](https://docs.astral.sh/uv/) is what CI uses.

```bash
git clone https://github.com/ife-bat/esf.git
cd esf
uv sync --dev
uv run pytest        # everything should pass
```

`uv sync` installs the runtime dependencies; `--dev` adds the test tooling
(`pytest`, `hypothesis`, `ruff`).

## With pip

```bash
pip install -e .                 # runtime only
pip install -e . --group dev     # with the test dependencies
```

## Optional dependency groups

- **`dev`** — the test suite and the linter (`uv sync --dev`).
- **`docs`** — this documentation site
  (`uv run --group docs zensical serve`).

## Verify

```python
import esf
print(esf.__version__)
prms = esf.get_example_params()   # the paper's parameters
```
