# Engineering Backlog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the live remainder of `waper_refactoring_spec.md` — docstrings, config
serialisation, hemisphere-correct plotting, and parallel identification — so the spec and the
housekeeping backlog can both be retired.

**Architecture:** Four independent workstreams plus a retirement task. Nothing here changes
identification or tracking numerics: the only task that touches the pipeline (parallel
identification) is required to produce byte-identical output to the sequential path, and is
gated on a picklability test before any parallelism is written. Two tasks add public API
surface (`WaperConfig.to_yaml`/`from_yaml`, `Waper.from_config`, a `projection=` argument on
the plot methods); the rest are additive or documentation.

**Tech Stack:** Python >= 3.11, pytest, ruff, mypy, cartopy/matplotlib, pyvista/geovista,
xarray, networkx, `concurrent.futures` (stdlib), PyYAML (new dependency, Task 3 only).

## Global Constraints

- **Python floor is `>= 3.11`.** Do not lower it. Ruff's `target-version` is deliberately
  **inferred** from `requires-python` — do not pin it.
- **Every task ends green on all three:** `ruff check .`, `mypy waper`, and
  `pytest -m "not slow"`. A task is not done until all three have been run and their output
  read. Current baseline: **144 passed, 1 skipped, 4 deselected**.
- **Run pytest directly.** The conda environment is already active; do not wrap commands in
  `conda run`.
- **Real-data tests must be `skipif`-guarded.** `datasets/forecast_bust_hourly.nc` (652 MB) is
  gitignored and absent on CI and on fresh clones. Follow the existing idiom:
  `@pytest.mark.skipif(not DATA_PATH.exists(), reason="...")`.
- **Long-running tests are marked `slow`.** CI runs `pytest -m "not slow"`; the marker is
  already registered in `pyproject.toml`.
- **`mypy` runs with `check_untyped_defs` off.** The package is largely unannotated; do not
  turn it on as a side effect of another task.
- **Commit per task**, with the task's own tests. Do not batch tasks into one commit.
- **Do not commit `example.png` or `datasets/download_era5.ipynb`.** Both are uncommitted in
  the working tree by Joy's choice and are on the out-of-scope register below.

---

## Out of scope — needs Joy's input

Carried verbatim from `2026-08-07-housekeeping-backlog.md`, which this plan retires. **Do not
action any of these as engineering work.** Each changes scientific behaviour or needs external
credentials.

- **Promoting the operating point.** `datasets/visualize.py` hard-codes
  `edge_pruning_threshold=0.02` / `penalty_length_scale_km=4000` / `node_pruning_threshold=20`
  while `WaperConfig` defaults stay at `3e-5` / `2000`. Two configurations are live at once.
  Task 3 makes this *visible* by moving the script's values into a checked-in YAML file; it
  does **not** change either set of numbers.
- **The group-velocity question** — envelope primitive vs trough-to-trough handoff vs dropping
  the expectation. Blocks the feature-track line of work. See `assessments/2026-08-07.md`.
- **Envelope-weighted edge pruning** (`envelope_segmentation_proposal.md`), and whether it adds
  anything on top of the shipped `lat_gate` branch resolution.
- **Re-downloading `datasets/v_winds_300mb_nh_2022_2023.nc`** (still a 239 B stub) — needs ERA5
  credentials.
- **Disposition of two uncommitted files**: `example.png` and the `download_era5.ipynb`
  output-only diff.

Also deliberately cut from this plan:

- **Spec task 7.3 (Hovmöller diagram)** — a new feature, not a fix. Deferred by Joy on
  2026-08-12. Pull it in later if wanted.
- **Spec task 9.3 (`to_dataset()` / `tracks_to_dataframe()`)** — superseded. `waper/io/`
  already ships `extract_rwps()`, `extract_track_nodes()`, `extract_track_edges()` returning
  DataFrames, plus `save_catalogue`/`load_catalogue`. Retire, do not implement.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `waper/interface/api.py` | modify | `Waper`, `WaperConfig`. Gains docstrings (T1), YAML + `from_config` (T3), `projection=` pass-through (T4), `n_jobs` (T6) |
| `waper/interface/explorer.py` | modify | Docstrings (T1); `default_projection` moves out to `projections.py` and is re-imported (T4) |
| `waper/interface/projections.py` | **create** | New. Single responsibility: the display-projection and map-extent defaults shared by `visualization.py` and `explorer.py` |
| `waper/interface/visualization.py` | modify | Display CRS separated from data CRS; hemisphere-aware extent; gridlines (T4) |
| `waper/io/catalogue.py`, `waper/io/extract.py` | modify | Docstrings only (T2) |
| `datasets/configs/visualize_operating_point.yaml` | **create** | The operating point `datasets/visualize.py` already uses, as data rather than a literal (T3) |
| `tests/test_config_yaml.py` | **create** | Round-trip and `from_config` tests (T3) |
| `tests/interface/test_projections.py` | **create** | Hemisphere defaults and display/data CRS separation (T4) |
| `tests/test_benchmark.py` | **create** | `slow`-marked timing baselines (T5) |
| `tests/test_parallel_identify.py` | **create** | Picklability, then sequential-vs-parallel equality (T6) |
| `results/benchmarks.md` | **create** | Recorded baseline timings (T5) |
| `pyproject.toml` | modify | Ruff `D` rules (T1), PyYAML dependency (T3) |

---

### Task 1: Docstrings on the documented API surface, and a ruff gate

**Why:** The published API reference at https://joymonteiro.github.io/waper/ is
signature-only. `quartodoc` renders exactly four objects — `Waper`, `WaperConfig`,
`WaperSingleTimestepData` (from `waper/__init__.py`'s `__all__`) and `RWPExplorer` — and none
carries a docstring. This task covers `api.py` and `explorer.py`: **19 missing** (14 + 5).

**Files:**
- Modify: `waper/interface/api.py` — `WaperConfig`, `WaperSingleTimestepData`, `Waper`,
  `identify_rwps`, `track_rwps`, `plot_clusters`, `plot_association_graph`,
  `plot_pruned_graph`, `plot_rwp_graphs`, `plot_rwp_polygons`, `plot_raster`, `plot_tracks`,
  `plot_track_polygons`, `plot_track_rwps`
- Modify: `waper/interface/explorer.py` — `nodes_layer`, `polygons_layer`, `edges_layer`,
  `field_layer`, `RWPExplorer`
- Modify: `pyproject.toml` — `[tool.ruff.lint]` select + per-file-ignores + pydocstyle
  convention

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable. Later tasks add docstrings to any new public function they
  create, because the ruff gate this task installs will fail them otherwise.

**Style:** Google convention (`Args:` / `Returns:`), matching the existing docstring in
`waper/identification/utils.py::get_vtk_object_from_data_array`. Do not switch the file to
numpydoc.

- [ ] **Step 1: Turn on the gate and watch it fail**

Add to `pyproject.toml` under `[tool.ruff.lint]`, appending to the existing `select` list:

```toml
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "PIE",  # flake8-pie
    "RUF",  # ruff-specific
    "D",    # pydocstyle — scoped to the documented API surface, see per-file-ignores
]
```

Append to the existing `ignore` list:

```toml
    # Module and package docstrings are not the point here; the API reference
    # renders objects, not modules. `__init__` is documented on its class.
    "D100", "D104", "D105", "D107",
```

Append to `[tool.ruff.lint.per-file-ignores]`:

```toml
# `D` is deliberately scoped to the surface quartodoc renders (docs/_quarto.yml)
# plus the io layer users call directly. Everything else is internal: adding
# docstring pressure there would bury the pages that are actually published.
"waper/identification/*" = ["D"]
"waper/tracking/*" = ["D"]
"waper/interface/visualization.py" = ["D"]
"waper/interface/colormaps.py" = ["D"]
"tests/*" = ["D", "E702"]
"scripts/*" = ["D", "E402", "E702"]
"datasets/*" = ["D", "E402", "E702"]
"misc/*" = ["D"]
```

Note the three existing per-file-ignores for `tests/*`, `scripts/*` and `datasets/*` are
**replaced** by the lines above, not duplicated — keep their existing codes.

Add a new section (place it after `[tool.ruff.lint.per-file-ignores]`):

```toml
[tool.ruff.lint.pydocstyle]
convention = "google"
```

- [ ] **Step 2: Run the gate to confirm it fails, and capture the count**

Run: `ruff check . --statistics`
Expected: FAIL. `D101`/`D102`/`D103` violations across `waper/interface/api.py`,
`waper/interface/explorer.py`, `waper/io/catalogue.py`, `waper/io/extract.py`. Record the
number — Task 2 closes the `waper/io/` half.

- [ ] **Step 3: Document `waper/interface/api.py`**

Write one docstring per object listed under **Files** above. Content requirements, not
boilerplate — a reader must learn something the signature does not tell them:

- `WaperConfig`: state that it is **frozen** and that units are explicit —
  `track_pruning_threshold` is **km** (default 8000; the old `0.3` was a bug that emptied the
  graph), `penalty_length_scale_km` and `energy_radius_km` are km, `lat_gate` is degrees.
  Note that `track_weight_threshold=None` **disables** the overlap gate and that this is the
  default because the gate is uncalibrated.
- `WaperSingleTimestepData`: one timestep's intermediate state. Note `energy_raster` is `None`
  until `_identify_rwps` sets it.
- `Waper`: the entry point. Document the call order — `identify_rwps()` then `track_rwps()` —
  and that the plot methods require `identify_rwps()` to have run.
- `identify_rwps` / `track_rwps`: what they populate (`_time_step_data`, `_tracking_graph`,
  `_pruned_tracking_graph`).
- Each `plot_*`: what it draws, that it returns the matplotlib `Axes`, and for the ones that
  take `ax`, that a caller-supplied axes must already carry a cartopy projection.
- `plot_tracks`: document that `threshold=None` falls back to the **configured**
  `track_pruning_threshold`, and that this is not the same as
  `prune_tracking_graph(g, None)`, which means "keep every edge".

Example of the expected shape and depth:

```python
def track_rwps(self, num_time_steps=None):
    """Link identified wave packets across time into a tracking graph.

    Builds the full tracking graph and stores a pruned copy. Pruning drops edges
    whose centroid displacement exceeds ``track_pruning_threshold`` kilometres and,
    if ``track_weight_threshold`` is set, edges whose energy-overlap weight falls
    below it.

    Requires :meth:`identify_rwps` to have run first.

    Args:
        num_time_steps: Number of timesteps to link. ``None`` uses all of them.
    """
```

- [ ] **Step 4: Document `waper/interface/explorer.py`**

Same bar. For the four `*_layer` functions, document that `projection=None` falls back to
`default_projection(hemisphere)` and that the **data** CRS is always `PlateCarree` regardless
of the display projection. For `RWPExplorer`, state that it renders polar-stereographic by
default and that the display projection is overridable per call — the orthographic-over-India
view is the motivating case.

- [ ] **Step 5: Verify the two modules are clean**

Run: `ruff check waper/interface/api.py waper/interface/explorer.py`
Expected: PASS, no `D` violations. `waper/io/` still fails — Task 2 closes it.

- [ ] **Step 6: Verify the docs actually build with the new prose**

Run:
```bash
cd docs && quartodoc build && cd .. && quarto render docs/
```
Expected: succeeds, and `docs/_site/api/Waper.html` now contains the prose. If `quarto` is not
installed locally, skip this step and say so in the task report — CI's `docs.yml` covers it.

- [ ] **Step 7: Full verification**

Run: `ruff check waper/interface/ && mypy waper && pytest -m "not slow" -q`
Expected: no `D` violations in `waper/interface/`, mypy clean, 144 passed / 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml waper/interface/api.py waper/interface/explorer.py
git commit -m "docs(api): document the published API surface and gate it with ruff D"
```

---

### Task 2: Docstrings on the `waper/io/` layer

**Why:** 31 missing (`catalogue.py` 25, `extract.py` 6). `Catalogue` is the object every
analysis script and notebook actually holds, and its 20+ query methods
(`amplitudes`, `zonal_extent`, `implied_wavenumber`, `track_propagation`, `group_velocity`,
`merges`, `splits`, …) are undocumented. Separate task from Task 1 because it is a different
surface with a different reviewer question: Task 1 is "does the published page read well",
this is "are the query semantics right".

**Files:**
- Modify: `waper/io/catalogue.py` — `write_meta`, `read_meta`, `save_catalogue`, `Catalogue`,
  `load_catalogue`, and the `Catalogue` methods `table`, `filter`, `rwps`, `nodes`, `edges`,
  `samples`, `tracks`, `amplitudes`, `zonal_extent`, `implied_wavenumber`, `track_durations`,
  `track_propagation`, `group_velocity`, `merges`, `splits`, `amplitude_pdf`, `duration_pdf`,
  `spatial_frequency`, `cross_stat_correlations`, `rwps_in`
- Modify: `waper/io/extract.py` — `extract_nodes`, `extract_edges`, `extract_rwps`,
  `extract_samples`, `extract_track_nodes`, `extract_track_edges`

**Interfaces:**
- Consumes: the ruff `D` configuration installed by Task 1.
- Produces: nothing importable.

- [ ] **Step 1: Confirm the gate still fails here**

Run: `ruff check waper/io/`
Expected: FAIL with `D101`/`D102`/`D103` on the 31 objects above.

- [ ] **Step 2: Read each method before documenting it**

Do not infer semantics from the method name. Several are non-obvious and the docstring must
state the actual behaviour — in particular, for every method returning a DataFrame, name the
columns and the row granularity (one row per RWP? per track? per timestep?), and state the
**units** of any physical quantity. `group_velocity` and `track_propagation` in particular
must say what sign convention and what units they return, because that is exactly the
ambiguity the open group-velocity question turns on.

- [ ] **Step 3: Document `waper/io/extract.py`**

These six all take a `Waper` and return a `pandas.DataFrame`. Each docstring must list the
returned columns. Example:

```python
def extract_rwps(waper) -> pd.DataFrame:
    """Flatten identified wave packets into one row per RWP per timestep.

    Args:
        waper: A :class:`~waper.interface.api.Waper` after ``identify_rwps()``.

    Returns:
        DataFrame with columns ``time``, ``rwp_id``, ``weighted_lon``,
        ``weighted_lat``, ``peak_value``, ``zonal_extent_deg``. One row per
        identified packet; empty if no packets were found.
    """
```

Verify the column list against the function body before writing it — do not copy the example.

- [ ] **Step 4: Document `waper/io/catalogue.py`**

`Catalogue` itself gets a class docstring explaining that it is the **on-disk, queryable**
form of a `Waper` run, loaded by `load_catalogue`, and that `meta` carries the run's
`hemisphere` (which `waper/interface/explorer.py::_hemisphere` reads to pick a default
projection).

- [ ] **Step 5: Verify**

Run: `ruff check . && mypy waper && pytest -m "not slow" -q`
Expected: **`ruff check .` now passes with zero `D` violations tree-wide** — this is the step
that closes the gate Task 1 opened. mypy clean, 144 passed / 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add waper/io/catalogue.py waper/io/extract.py
git commit -m "docs(io): document the catalogue and extraction API"
```

---

### Task 3: `WaperConfig` YAML round-trip and `Waper.from_config`

**Why:** Two things at once, because neither is useful alone. Serialising a config is only
worth doing if you can then *run* from it, and today you cannot: `Waper.__init__` exposes 18
of `WaperConfig`'s 25 fields. **`hemisphere` is not among them.** Nothing in `tests/`,
`scripts/` or `datasets/` can select the Southern Hemisphere through the public constructor,
even though identification and rasterisation both honour it. Task 4 depends on this being
fixed — it cannot test a Southern Hemisphere plot without a way to build a SH `Waper`.

**Files:**
- Modify: `waper/interface/api.py:26-63` (`WaperConfig`), `:291-338` (`Waper.__init__`)
- Modify: `pyproject.toml` — add `"pyyaml"` to `[project.dependencies]`
- Create: `tests/test_config_yaml.py`
- Create: `datasets/configs/visualize_operating_point.yaml`
- Modify: `datasets/visualize.py:38-51` — load the YAML instead of the literal dict

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `WaperConfig.to_yaml(self, path: str | Path | None = None) -> str` — returns the YAML
    text; also writes it if `path` is given.
  - `WaperConfig.from_yaml(cls, source: str | Path) -> WaperConfig` — accepts a path or a YAML
    string.
  - `Waper.from_config(cls, data_array, config: WaperConfig) -> Waper`.
  - Task 4 calls `Waper.from_config` to construct a Southern Hemisphere run.

**The trap:** `WaperConfig` is `@dataclass(eq=False, frozen=True)`. **`eq=False` means `==`
falls back to identity**, so the obvious round-trip assertion
`WaperConfig.from_yaml(c.to_yaml()) == c` passes vacuously never — it always fails, even on a
perfect round-trip. Compare `dataclasses.asdict()` instead. Do not "fix" this by turning
`eq=True` on: `WaperSingleTimestepData` and the graph code rely on configs being hashable by
identity.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_yaml.py`:

```python
import dataclasses
from pathlib import Path

import pytest
import xarray as xr

from waper.interface.api import Waper, WaperConfig


def _config(**overrides):
    base = dict(
        debug=False,
        scalar_name="v",
        latitude_label="latitude",
        longitude_label="longitude",
        time_label="time",
        clip_value=2,
        extrema_threshold=10,
        max_latitude=80.1,
        min_latitude=20,
        node_pruning_threshold=20,
        edge_pruning_threshold=3e-5,
        max_edge_weight=1,
    )
    base.update(overrides)
    return WaperConfig(**base)


def test_yaml_round_trip_preserves_every_field():
    config = _config(hemisphere="south", lat_gate=12.5, track_weight_threshold=0.4)

    restored = WaperConfig.from_yaml(config.to_yaml())

    # `eq=False` on the dataclass means `==` is identity; compare fields.
    assert dataclasses.asdict(restored) == dataclasses.asdict(config)


def test_to_yaml_writes_the_file_and_returns_the_text(tmp_path):
    config = _config()
    path = tmp_path / "config.yaml"

    text = config.to_yaml(path)

    assert path.read_text() == text
    assert dataclasses.asdict(WaperConfig.from_yaml(path)) == dataclasses.asdict(config)


def test_none_survives_the_round_trip():
    # track_weight_threshold=None disables the overlap gate. YAML must not
    # turn it into the string "None".
    restored = WaperConfig.from_yaml(_config(track_weight_threshold=None).to_yaml())

    assert restored.track_weight_threshold is None


def test_from_yaml_rejects_an_unknown_field(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("scalar_name: v\nnot_a_real_field: 3\n")

    with pytest.raises(TypeError, match="not_a_real_field"):
        WaperConfig.from_yaml(path)


def test_from_config_reaches_fields_the_constructor_cannot(two_timestep_field):
    # `Waper.__init__` has no `hemisphere` parameter, so this is the only way
    # to run the Southern Hemisphere pipeline through the public API.
    config = _config(hemisphere="south", min_latitude=-80, max_latitude=-20)
    ds = xr.Dataset({"v": two_timestep_field})

    waper = Waper.from_config(ds, config)

    assert waper._config.hemisphere == "south"
    assert waper._num_time_steps == 2
    assert waper._time_step_data == []


def test_from_config_runs_the_pipeline(two_timestep_field):
    config = _config()
    waper = Waper.from_config(xr.Dataset({"v": two_timestep_field}), config)

    waper.identify_rwps()

    assert len(waper._time_step_data) == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_config_yaml.py -v`
Expected: FAIL — `AttributeError: type object 'WaperConfig' has no attribute 'from_yaml'`.

- [ ] **Step 3: Add the dependency**

In `pyproject.toml`, add `"pyyaml"` to `[project.dependencies]`, next to the other unpinned
entries. It is a pure-Python wheel on every platform; no version floor is needed.

Run: `python -c "import yaml; print(yaml.__version__)"` to confirm it is already in the local
env (it almost certainly is, as a transitive dependency).

- [ ] **Step 4: Implement the config methods**

Add to `WaperConfig` in `waper/interface/api.py`:

```python
    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialise this configuration to YAML.

        Args:
            path: If given, the YAML is also written to this path.

        Returns:
            The YAML document as a string.
        """
        text = yaml.safe_dump(dataclasses.asdict(self), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> "WaperConfig":
        """Build a configuration from a YAML file or YAML string.

        Args:
            source: A path to a ``.yaml`` file, or the YAML document itself.

        Returns:
            The deserialised configuration.

        Raises:
            TypeError: If the document contains a key that is not a field of
                this class.
        """
        candidate = Path(source)
        try:
            # A YAML document is not a filename. `is_file()` is the cheapest
            # way to tell the two accepted inputs apart; it raises rather than
            # returning False when the string is too long to be a path.
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        text = candidate.read_text() if is_file else str(source)
        return cls(**yaml.safe_load(text))
```

`from_yaml` must accept both a path and a raw YAML string — the tests use both. Do not reach
for `isinstance(source, (str, Path))` here: ruff's `UP` rules flag the tuple form, and the
`Path(source)` round-trip above makes the check unnecessary.

`cls(**...)` raises `TypeError` naming the offending keyword for an unknown field, which is
what the test asserts — no manual validation needed.

Add the imports at the top of `api.py`: `import dataclasses`, `from pathlib import Path`,
`import yaml`. `from dataclasses import dataclass` is already there; keep it and add the
module import alongside, or switch the decorator to `@dataclasses.dataclass` — either is fine,
but do not leave both spellings for the same thing in one file.

- [ ] **Step 5: Implement `Waper.from_config` without duplicating `__init__`**

`__init__` currently builds a `WaperConfig` then does five lines of setup. Extract those five
lines so both constructors share them:

```python
    def _setup(self, data_array, config: WaperConfig) -> None:
        self._config = config
        self.data_array = data_array
        self._num_time_steps = len(data_array[config.time_label])
        self._time_step_data: list = []

        if config.debug:
            logging.basicConfig(level=logging.DEBUG)

    @classmethod
    def from_config(cls, data_array, config: WaperConfig) -> "Waper":
        """Construct from a :class:`WaperConfig`, reaching every field.

        ``__init__`` exposes 18 of the config's 25 fields as keyword arguments;
        ``hemisphere``, ``hull_method``, ``energy_radius_km`` and the clustering
        parameters are not among them. This is the way to set those — and the way
        to run from a config file:

        >>> waper = Waper.from_config(ds, WaperConfig.from_yaml("run.yaml"))

        Args:
            data_array: The input dataset, indexed by the config's ``time_label``.
            config: A fully specified configuration.

        Returns:
            An unrun ``Waper``. Call ``identify_rwps()`` next.
        """
        obj = cls.__new__(cls)
        obj._setup(data_array, config)
        return obj
```

Then replace the tail of `__init__` (from `self.data_array = data_array` to the
`logging.basicConfig` call) with `self._setup(data_array, self._config)`. Keep `__init__`'s
existing signature and behaviour exactly — it has many callers in `tests/` and `scripts/`.

Note `__init__` currently reads `if debug:` on the local parameter; `_setup` reads
`config.debug`. These are the same value because `__init__` passes `debug=debug` into the
config. Verify that is still true after the edit.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_config_yaml.py -v`
Expected: all 6 PASS.

- [ ] **Step 7: Make the second live configuration visible as data**

Create `datasets/configs/visualize_operating_point.yaml` holding exactly the values
`datasets/visualize.py` already uses — no changes to any number:

```yaml
# The operating point used by datasets/visualize.py, extracted from the literal
# dict that used to live in that script. These values are NOT the WaperConfig
# defaults: edge_pruning_threshold is 0.02 here vs 3e-5 in WaperConfig, and
# penalty_length_scale_km is 4000 vs 2000. Whether to promote these to the
# package defaults is Joy's call, not an engineering task — see the
# out-of-scope register in the plan that created this file.
scalar_name: v
latitude_label: latitude
longitude_label: longitude
time_label: time
clip_value: 2
extrema_threshold: 10
min_latitude: 20
max_latitude: 80
node_pruning_threshold: 20
edge_pruning_threshold: 0.02
max_edge_weight: 1
track_pruning_threshold: 8000
penalty_length_scale_km: 4000
debug: false
```

In `datasets/visualize.py`, replace the `WAPER_KWARGS` dict with a load of this file, keeping
the existing call sites working. If those call sites splat `**WAPER_KWARGS` into `Waper(...)`,
the least invasive change is to keep a dict:

```python
WAPER_CONFIG_PATH = Path(__file__).parent / "configs" / "visualize_operating_point.yaml"
WAPER_CONFIG = WaperConfig.from_yaml(WAPER_CONFIG_PATH)
WAPER_KWARGS = {
    k: v for k, v in dataclasses.asdict(WAPER_CONFIG).items()
    if k in inspect.signature(Waper.__init__).parameters
}
```

Read the script's actual call sites first and pick whichever of the two forms keeps them
unchanged. **Verify the resulting kwargs are value-identical to the dict you deleted** — print
both and diff them before committing.

- [ ] **Step 8: Full verification**

Run: `ruff check . && mypy waper && pytest -m "not slow" -q`
Expected: clean, and **150 passed** (144 + 6 new) / 1 skipped.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml waper/interface/api.py tests/test_config_yaml.py \
        datasets/configs/visualize_operating_point.yaml datasets/visualize.py
git commit -m "feat(config): YAML round-trip for WaperConfig and Waper.from_config"
```

---

### Task 4: Hemisphere-aware, caller-overridable plot projections

**Why:** `hemisphere="south"` is honoured by identification (`api.py:233`) and rasterisation
(`api.py:259,268,272`) but **not by any plot**. `visualization.py` hardcodes
`_STEREO_NH = ccrs.Stereographic(central_longitude=0, central_latitude=90)` at 6 call sites
and `set_extent([-180, 180, 20, 90])` at 2. A Southern Hemisphere run plots blank. This is
spec tasks 7.1, 7.2 and 7.4 together — they are one edit to the same 6 call sites.

**The crux (spec 7.4).** `_STEREO_NH` is currently doing **two different jobs**: it is the
display projection (`plt.subplot(projection=...)`) *and* the data CRS for polygon vertices
(`ax.fill(..., transform=_STEREO_NH)` at `visualization.py:316`, `ax.scatter(...,
transform=_STEREO_NH)` at `:348`). Those coincide today only because the display projection
happens to equal the CRS the polygons were built in. **The `transform=` arguments must keep
following the polygon CRS, not the new display projection** — otherwise every polygon lands in
the wrong place the moment someone passes `projection=ccrs.Orthographic(...)`. Separate the
two names; do not collapse them.

**Files:**
- Create: `waper/interface/projections.py`
- Modify: `waper/interface/visualization.py:46-47, 62, 116, 157, 212, 284-287, 355-358`
- Modify: `waper/interface/explorer.py:28-37` — import `default_projection` from the new module
- Modify: `waper/interface/api.py` — plot methods gain `projection=None` pass-through
- Create: `tests/interface/test_projections.py`

**Interfaces:**
- Consumes: `Waper.from_config` from Task 3 (to build a Southern Hemisphere `Waper`).
- Produces:
  - `waper.interface.projections.default_projection(hemisphere: str) -> ccrs.Projection`
  - `waper.interface.projections.default_extent(hemisphere: str) -> list[float]`
  - `waper.interface.projections.POLYGON_CRS` — the fixed stereographic CRS RWP polygons and
    rasters are built in. **Not** the display projection.
  - `Waper.plot_rwp_polygons(time_index, plot_samples=False, ax=None, projection=None)` and the
    same `projection=None` keyword on the other eight plot methods: `plot_clusters`,
    `plot_association_graph`, `plot_pruned_graph`, `plot_rwp_graphs`, `plot_raster`,
    `plot_tracks`, `plot_track_polygons`, `plot_track_rwps`.

**Verified call mapping** (`api.py`), which decides which methods take `hemisphere`:

| Group | Helper | `Waper` methods |
|---|---|---|
| A — stereographic, sets extent | `_plot_polygons`, `_plot_raster` | `plot_rwp_polygons` (`:419`), `plot_raster` (`:432`), `plot_track_polygons` (`:478`) |
| B — whole-globe PlateCarree(180) | `_plot_clusters`, `_plot_graph`, `_plot_rwp_paths` | `plot_clusters` (`:359`), `plot_association_graph` (`:374`), `plot_pruned_graph` (`:381`), `plot_rwp_graphs` (`:392`), `plot_tracks` (`:445`), `plot_track_rwps` (`:506`) |

- [ ] **Step 1: Write the failing tests**

Create `tests/interface/test_projections.py`:

```python
import cartopy.crs as ccrs
import matplotlib
import pytest
import xarray as xr

matplotlib.use("Agg")

from waper.interface.api import Waper, WaperConfig
from waper.interface.projections import POLYGON_CRS, default_extent, default_projection


def test_default_projection_follows_the_hemisphere():
    assert default_projection("north").proj4_params["lat_0"] == 90
    assert default_projection("south").proj4_params["lat_0"] == -90


def test_default_extent_follows_the_hemisphere():
    assert default_extent("north") == [-180, 180, 20, 90]
    assert default_extent("south") == [-180, 180, -90, -20]


def test_polygon_crs_is_northern_stereographic_regardless_of_hemisphere():
    # RWP polygons and rasters are built in a fixed stereographic CRS; the
    # display projection is a separate concern. Changing one must not move
    # the other.
    assert POLYGON_CRS is not default_projection("south")


def _sh_waper(southern_hemisphere_wave_field):
    config = WaperConfig(
        debug=False, scalar_name="v", latitude_label="latitude",
        longitude_label="longitude", time_label="time", clip_value=2,
        extrema_threshold=10, max_latitude=-20, min_latitude=-80,
        node_pruning_threshold=20, edge_pruning_threshold=3e-5,
        max_edge_weight=1, hemisphere="south",
    )
    ds = xr.Dataset({"v": southern_hemisphere_wave_field})
    waper = Waper.from_config(ds, config)
    waper.identify_rwps()
    return waper


def test_southern_hemisphere_run_plots_in_the_southern_hemisphere(
    southern_hemisphere_wave_field,
):
    waper = _sh_waper(southern_hemisphere_wave_field)

    ax = waper.plot_rwp_polygons(0)

    assert ax.projection.proj4_params["lat_0"] == -90
    _, _, lat_lo, lat_hi = ax.get_extent(crs=ccrs.PlateCarree())
    assert lat_lo < -20 and lat_hi <= 0


def test_caller_can_override_the_display_projection(southern_hemisphere_wave_field):
    waper = _sh_waper(southern_hemisphere_wave_field)
    orthographic = ccrs.Orthographic(central_longitude=75, central_latitude=25)

    ax = waper.plot_rwp_polygons(0, projection=orthographic)

    assert ax.projection == orthographic


def test_every_plot_method_accepts_projection():
    # Every plot entry point takes the same keyword, so a caller can switch
    # the whole figure over consistently.
    import inspect

    for name in [
        "plot_clusters", "plot_association_graph", "plot_pruned_graph",
        "plot_rwp_graphs", "plot_rwp_polygons", "plot_raster", "plot_tracks",
        "plot_track_polygons", "plot_track_rwps",
    ]:
        params = inspect.signature(getattr(Waper, name)).parameters
        assert "projection" in params, f"{name} has no projection argument"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/interface/test_projections.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'waper.interface.projections'`.

- [ ] **Step 3: Create the projections module**

```python
"""Display projections and map extents shared by the plotting layers.

Two distinct things live here and must not be conflated:

* ``POLYGON_CRS`` is the coordinate system RWP polygons and rasters are *built*
  in. It is fixed. Vertex coordinates are meaningless in any other CRS, so it is
  what belongs in matplotlib's ``transform=`` argument.
* ``default_projection`` returns the CRS a map is *displayed* in. It is a
  presentation choice and callers override it freely.
"""

import cartopy.crs as ccrs

#: The CRS RWP polygons and rasters are constructed in. Fixed; not a display choice.
POLYGON_CRS = ccrs.Stereographic(central_longitude=0, central_latitude=90)

PLATE_CARREE = ccrs.PlateCarree(central_longitude=0)


def default_projection(hemisphere: str) -> ccrs.Projection:
    """Default *display* projection: polar stereographic for the hemisphere.

    Seam-free, so dateline-crossing packets stay contiguous (Web Mercator tore
    them apart). Override per call for other workflows — western disturbances
    over South Asia read better in
    ``ccrs.Orthographic(central_longitude=75, central_latitude=25)``.

    Args:
        hemisphere: ``"north"`` or ``"south"``.

    Returns:
        The projection to draw into.
    """
    lat0 = -90 if hemisphere == "south" else 90
    return ccrs.Stereographic(central_latitude=lat0, central_longitude=0)


def default_extent(hemisphere: str) -> list[float]:
    """Default map extent in PlateCarree degrees, as ``[lon_lo, lon_hi, lat_lo, lat_hi]``.

    Clips to the mid-to-high latitudes of the hemisphere the run identified in;
    the opposite pole projects to infinity in a polar projection and must be
    excluded.

    Args:
        hemisphere: ``"north"`` or ``"south"``.

    Returns:
        A 4-element extent suitable for ``ax.set_extent(..., crs=PLATE_CARREE)``.
    """
    return [-180, 180, -90, -20] if hemisphere == "south" else [-180, 180, 20, 90]
```

- [ ] **Step 4: Rewire `visualization.py`**

Replace the module-level constants at `:46-47`:

```python
from .projections import PLATE_CARREE as _PLATE_CARREE
from .projections import POLYGON_CRS, default_extent, default_projection
```

**The five `_plot_*` functions do not all get the same treatment.** Two of them draw in the
stereographic polygon view; three draw a whole-globe `PlateCarree(central_longitude=180)`
panel and set no extent at all. Swapping the latter to a polar projection would be a visual
regression, not a fix — their default must not change.

*Group A — `_plot_polygons` (`:284-287`) and `_plot_raster` (`:355-358`).* These currently
default to `_STEREO_NH` and clip to `[-180, 180, 20, 90]`:

1. Add `projection=None` and `hemisphere="north"` parameters.
2. Replace `ax = plt.subplot(projection=_STEREO_NH)` with
   `ax = plt.subplot(projection=projection or default_projection(hemisphere))`.
3. Replace `ax.set_extent([-180, 180, 20, 90], crs=_PLATE_CARREE)` with
   `ax.set_extent(default_extent(hemisphere), crs=_PLATE_CARREE)`.
4. **Leave every `transform=` argument alone in meaning** — but rename the constant:
   `transform=_STEREO_NH` at `:316` and `:348` becomes `transform=POLYGON_CRS`. These describe
   the data, not the display. Check the `imshow(..., extent=(WAPER_X_BOUNDS…))` call in
   `_plot_raster` in the same pass: it passes no `transform` at all, so it is relying on the
   axes projection happening to equal the raster's CRS. That reliance **is** the bug spec 7.4
   names, and it breaks the moment `projection=` is overridden — add `transform=POLYGON_CRS`.
5. `_plot_raster(raster_data)` takes no `ax`. Give it
   `_plot_raster(raster_data, ax=None, projection=None, hemisphere="north")` to match.

*Group B — `_plot_clusters` (`:62`, `:116`), `_plot_graph` (`:157`), `_plot_rwp_paths`
(`:212`).* These default to `ccrs.PlateCarree(central_longitude=180)` and set no extent:

6. Add `projection=None` only — **no `hemisphere` parameter, no `set_extent` call.** A
   whole-globe PlateCarree panel is already hemisphere-agnostic; adding an extent would crop
   plots that are correct today.
7. Replace the hardcoded projection with
   `projection or ccrs.PlateCarree(central_longitude=180)`, keeping the `211`/`212` positional
   arguments in `_plot_clusters` exactly as they are.

*Both groups:*

8. Add gridlines after the existing `coastlines` call in all five functions:
   `ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)`.

- [ ] **Step 5: Fix the `annotate` transform while you are in the file**

`visualization.py:333-340` passes `transform=_PLATE_CARREE` to `ax.annotate`. Matplotlib's
`annotate` does not accept a cartopy transform for its `xy` — the argument applies to the text
artist, so the label anchors in the wrong coordinate system. The cartopy idiom is:

```python
            ax.annotate(
                str(index),
                xy=(lon, lat),
                xycoords=_PLATE_CARREE._as_mpl_transform(ax),
                fontsize=8,
                bbox={'boxstyle': "round", 'fc': "white", 'ec': "b"},
                zorder=1000,
            )
```

- [ ] **Step 6: Thread `projection` through the `Waper` plot methods**

Every `plot_*` method in `api.py` gains `projection=None` and passes it down. Only the methods
whose underlying function is in **Group A** — `plot_rwp_polygons`, `plot_raster`,
`plot_track_polygons` — also pass `hemisphere=self._config.hemisphere`; the Group B functions
have no such parameter and passing one is a `TypeError`. Example:

```python
    def plot_rwp_polygons(self, time_index, plot_samples=False, ax=None, projection=None):
        """..."""   # docstring already written in Task 1 — extend it for `projection`
        time_step_data = self._time_step_data[time_index]
        return _plot_polygons(
            ...,
            ax=ax,
            projection=projection,
            hemisphere=self._config.hemisphere,
        )
```

Update the Task 1 docstrings to describe the new argument: `projection=None` means "the
hemisphere default"; passing `ax` with its own projection takes precedence over both.

- [ ] **Step 7: Run the tests**

Run: `pytest tests/interface/test_projections.py -v`
Expected: all 6 PASS.

- [ ] **Step 8: Confirm no Northern Hemisphere plot changed**

The existing suite covers NH plotting. Run: `pytest -m "not slow" -q`
Expected: **156 passed** (150 + 6) / 1 skipped, with no pre-existing test newly failing. If an
existing plot test fails, the display/data CRS separation was done wrong — do not adjust the
test to match; re-read step 4.

- [ ] **Step 9: Point `explorer.py` at the shared module**

Delete `explorer.py`'s local `default_projection` (`:28-37`) and import it from
`.projections` instead, keeping the name importable from `explorer` for any existing caller:

```python
from .projections import default_projection  # re-exported: callers import it from here
```

Leave `_stereo_proj4` where it is — it builds a proj4 *string* for geopandas, which is a
different consumer.

Run: `pytest tests/interface/ -q`
Expected: PASS, including the datashader-gated explorer tests.

- [ ] **Step 10: Full verification and commit**

Run: `ruff check . && mypy waper && pytest -m "not slow" -q`

```bash
git add waper/interface/projections.py waper/interface/visualization.py \
        waper/interface/explorer.py waper/interface/api.py \
        tests/interface/test_projections.py
git commit -m "fix(viz): make plot projections hemisphere-aware and overridable"
```

---

### Task 5: Benchmark baselines

**Why:** Spec task 6.3. Task 6 claims a speedup; without a recorded baseline that claim is
unfalsifiable. (Spec 6.1 and 6.2 are already closed — absorbed by tasks 5.3 and 2.4.)

**Files:**
- Create: `tests/test_benchmark.py`
- Create: `results/benchmarks.md`

**Interfaces:**
- Consumes: `Waper.from_config` (Task 3) to build runs from an explicit config.
- Produces: `results/benchmarks.md`, which Task 6 appends its parallel numbers to.

**Design note:** Do **not** assert a wall-clock ceiling as the primary check. Shared CI runners
vary by more than the effect sizes here, and a timing assertion that flakes gets deleted rather
than investigated. These tests are marked `slow` (so CI's `-m "not slow"` skips them), assert
only a generous ceiling that catches an order-of-magnitude regression, and print the timing for
the human record.

- [ ] **Step 1: Write the benchmark module**

Create `tests/test_benchmark.py`:

```python
"""Timing baselines for identification and tracking.

Marked `slow`: CI runs `pytest -m "not slow"` and skips these. Run them
deliberately with `pytest -m slow tests/test_benchmark.py -s` and record the
numbers in results/benchmarks.md.
"""

import time

import numpy as np
import pytest
import xarray as xr

from waper.interface.api import Waper, WaperConfig


def _synthetic_field(n_time, n_lat, n_lon):
    """A zonally-wavenumber-6 field with a latitudinally-confined envelope."""
    lats = np.linspace(20, 80, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    times = np.arange(n_time)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    envelope = np.exp(-(((lat_grid - 50) / 12.0) ** 2))
    frames = [
        20 * envelope * np.sin(np.deg2rad(6 * lon_grid + 10 * t))
        for t in times
    ]

    return xr.DataArray(
        np.stack(frames),
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": lats, "longitude": lons},
        name="v",
    )


def _config():
    return WaperConfig(
        debug=False, scalar_name="v", latitude_label="latitude",
        longitude_label="longitude", time_label="time", clip_value=2,
        extrema_threshold=10, max_latitude=80.1, min_latitude=20,
        node_pruning_threshold=20, edge_pruning_threshold=3e-5,
        max_edge_weight=1,
    )


@pytest.mark.slow
def test_identification_benchmark_1p5_degree():
    # 121 x 240 — a 1.5-degree global grid, one timestep.
    field = _synthetic_field(1, 121, 240)
    waper = Waper.from_config(xr.Dataset({"v": field}), _config())

    start = time.perf_counter()
    waper.identify_rwps()
    elapsed = time.perf_counter() - start

    print(f"\nidentification, 1 timestep @ 121x240: {elapsed:.2f}s")
    assert elapsed < 120, "an order of magnitude slower than the recorded baseline"


@pytest.mark.slow
def test_tracking_benchmark_10_timesteps():
    field = _synthetic_field(10, 121, 240)
    waper = Waper.from_config(xr.Dataset({"v": field}), _config())
    waper.identify_rwps()

    start = time.perf_counter()
    waper.track_rwps()
    elapsed = time.perf_counter() - start

    print(f"\ntracking, 10 timesteps @ 121x240: {elapsed:.2f}s")
    assert elapsed < 60, "an order of magnitude slower than the recorded baseline"
```

- [ ] **Step 2: Run them and read the numbers**

Run: `pytest -m slow tests/test_benchmark.py -s -v`
Expected: PASS. Record the two printed timings.

If either exceeds its ceiling on a normal machine, **do not raise the ceiling silently** —
report the number, because that is a finding about the pipeline, not about the test.

- [ ] **Step 3: Confirm they stay out of the default run**

Run: `pytest -m "not slow" -q --collect-only | tail -3`
Expected: `test_benchmark.py` is deselected, not collected.

- [ ] **Step 4: Record the baseline**

Create `results/benchmarks.md` with the date, the machine (`uname -sm` and the CPU model), the
two timings from step 2, and the commit SHA they were measured at. State explicitly that these
are synthetic-field numbers and are not comparable to ERA5 runs.

- [ ] **Step 5: Verify and commit**

Run: `ruff check . && pytest -m "not slow" -q` (expected: still 156 passed / 1 skipped, with 6
deselected now instead of 4)

```bash
git add tests/test_benchmark.py results/benchmarks.md
git commit -m "test(perf): record identification and tracking timing baselines"
```

---

### Task 6: Parallel timestep identification

**Why:** Spec task 9.2. `identify_rwps` loops timesteps sequentially and each timestep is
independent. The task was blocked on VTK objects being unpicklable; Phase 5 removed VTK from
the package, and pyvista 0.48.4's `PolyData` round-trips through `pickle` (verified
2026-08-12). The remaining risk is whether the *whole* `WaperSingleTimestepData` — which holds
two `PolyData`, two `networkx.Graph`, an xarray `DataArray` and two ndarrays — survives the
process boundary. **Step 1 settles that with a test before any parallelism is written.**

**Files:**
- Create: `tests/test_parallel_identify.py`
- Modify: `waper/interface/api.py::Waper.identify_rwps`
- Modify: `results/benchmarks.md` (append the parallel numbers)

**Interfaces:**
- Consumes: `Waper.from_config` (Task 3); `results/benchmarks.md` (Task 5).
- Produces: `Waper.identify_rwps(self, n_jobs: int = 1)`. `n_jobs=1` keeps the current
  sequential path exactly; `n_jobs > 1` uses that many worker processes; `n_jobs=-1` uses
  `os.cpu_count()`.

**Acceptance is not a green suite.** Like Phase 5.2, this changes how the pipeline executes, so
it must be shown to produce **identical** output, not merely working output.

- [ ] **Step 1: Write the picklability test first — it gates the rest of the task**

Create `tests/test_parallel_identify.py`:

```python
import pickle

import numpy as np
import pytest
import xarray as xr

from waper.interface.api import Waper, WaperConfig, _identify_rwps


def _config(**overrides):
    base = dict(
        debug=False, scalar_name="v", latitude_label="latitude",
        longitude_label="longitude", time_label="time", clip_value=2,
        extrema_threshold=10, max_latitude=80.1, min_latitude=20,
        node_pruning_threshold=20, edge_pruning_threshold=3e-5,
        max_edge_weight=1,
    )
    base.update(overrides)
    return WaperConfig(**base)


def test_config_pickles():
    config = _config()
    assert pickle.loads(pickle.dumps(config)).scalar_name == "v"


def test_single_timestep_result_survives_a_process_boundary(two_timestep_field):
    # Everything a worker returns has to pickle: two PolyData, two networkx
    # Graphs, a DataArray and two ndarrays.
    result = _identify_rwps(two_timestep_field[0], _config())

    restored = pickle.loads(pickle.dumps(result))

    assert restored.vtk_data.n_points == result.vtk_data.n_points
    assert len(restored.identified_rwp_paths) == len(result.identified_rwp_paths)
    assert restored.association_graph.number_of_edges() == \
        result.association_graph.number_of_edges()
    np.testing.assert_array_equal(restored.raster_data, result.raster_data)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_parallel_identify.py -v`
Expected: both PASS.

**If `test_single_timestep_result_survives_a_process_boundary` fails, STOP and report.** The
task's design does not survive it: the fix would be to return a reduced payload and rebuild
`vtk_data` in the parent, which is a different design and needs Joy's sign-off, not an
improvised workaround.

- [ ] **Step 3: Write the equality test**

Append to `tests/test_parallel_identify.py`:

```python
def _paths_signature(waper):
    """A comparable summary of what identification produced."""
    return [
        [sorted(map(str, path)) for path in ts.identified_rwp_paths]
        for ts in waper._time_step_data
    ]


def test_parallel_identification_matches_sequential(two_timestep_field):
    ds = xr.Dataset({"v": two_timestep_field})

    sequential = Waper.from_config(ds, _config())
    sequential.identify_rwps()

    parallel = Waper.from_config(ds, _config())
    parallel.identify_rwps(n_jobs=2)

    assert _paths_signature(parallel) == _paths_signature(sequential)
    for par_ts, seq_ts in zip(
        parallel._time_step_data, sequential._time_step_data, strict=True
    ):
        np.testing.assert_array_equal(par_ts.raster_data, seq_ts.raster_data)


def test_timesteps_come_back_in_order(two_timestep_field):
    # A worker pool completes out of order; _time_step_data is indexed by
    # timestep everywhere downstream, so ordering is load-bearing.
    ds = xr.Dataset({"v": two_timestep_field})
    waper = Waper.from_config(ds, _config())

    waper.identify_rwps(n_jobs=2)

    times = [ts.input_data["time"].item() for ts in waper._time_step_data]
    assert times == sorted(times)


def test_n_jobs_1_is_the_sequential_path(two_timestep_field):
    waper = Waper.from_config(xr.Dataset({"v": two_timestep_field}), _config())

    waper.identify_rwps(n_jobs=1)

    assert len(waper._time_step_data) == 2
```

- [ ] **Step 4: Run to verify the new three fail**

Run: `pytest tests/test_parallel_identify.py -v`
Expected: `test_parallel_identification_matches_sequential` and
`test_timesteps_come_back_in_order` FAIL with
`TypeError: identify_rwps() got an unexpected keyword argument 'n_jobs'`.

- [ ] **Step 5: Implement**

In `waper/interface/api.py`:

```python
    def identify_rwps(self, n_jobs: int = 1):
        """Identify wave packets in every timestep.

        Args:
            n_jobs: Worker processes to use. ``1`` (the default) runs
                sequentially in this process. ``-1`` uses every CPU. Results
                are identical either way and always ordered by timestep.
        """
        scalar = self._config.scalar_name
        frames = [self.data_array[scalar][i] for i in range(self._num_time_steps)]

        if n_jobs == 1:
            for frame in tqdm(frames):
                self._time_step_data.append(_identify_rwps(frame, self._config))
            return

        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # `executor.map` yields in submission order, which is what keeps
            # _time_step_data indexed by timestep.
            self._time_step_data.extend(
                tqdm(
                    pool.map(_identify_rwps, frames, repeat(self._config)),
                    total=len(frames),
                )
            )
```

Add the imports: `import os`, `from concurrent.futures import ProcessPoolExecutor`,
`from itertools import repeat`.

Two things to get right:
- `_identify_rwps` is module-level, so it pickles by reference — do not turn it into a closure
  or a lambda.
- Keep the `n_jobs == 1` branch as a genuinely separate code path. It is the reference
  implementation the equality test compares against, and it must not acquire a pool's
  overhead or its start-method quirks.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_parallel_identify.py -v`
Expected: all 5 PASS.

On macOS the default start method is `spawn`, which re-imports the module in each worker;
`waper/__init__.py` installs a warnings filter before the pyvista import chain, so this is
already safe. If workers hang instead of failing, that is the symptom to report — do not
paper over it with a timeout.

- [ ] **Step 7: Verify against real data**

Add to `tests/test_parallel_identify.py`:

```python
from pathlib import Path

DATA_PATH = Path("datasets/forecast_bust_hourly.nc")


@pytest.mark.slow
@pytest.mark.skipif(not DATA_PATH.exists(), reason="652 MB input is gitignored")
def test_parallel_matches_sequential_on_real_data():
    ds = xr.open_dataset(DATA_PATH).isel(time=slice(0, 4))
    config = _config(scalar_name="v", min_latitude=20, max_latitude=80)

    sequential = Waper.from_config(ds, config)
    sequential.identify_rwps()

    parallel = Waper.from_config(ds, config)
    parallel.identify_rwps(n_jobs=-1)

    assert _paths_signature(parallel) == _paths_signature(sequential)
    for par_ts, seq_ts in zip(
        parallel._time_step_data, sequential._time_step_data, strict=True
    ):
        np.testing.assert_array_equal(par_ts.raster_data, seq_ts.raster_data)
        np.testing.assert_array_equal(par_ts.energy_raster, seq_ts.energy_raster)
```

Run: `pytest -m slow tests/test_parallel_identify.py -v -s`
Expected: PASS if the data file is present; SKIP otherwise. **Check the variable names in
`ds` before running** — confirm the scalar is `v` and the coordinate labels match the config,
the way `tests/test_method_comparison.py` already does.

- [ ] **Step 8: Measure and record the speedup**

Time `identify_rwps()` vs `identify_rwps(n_jobs=-1)` on the 10-timestep synthetic field from
Task 5, and append both numbers plus the core count to `results/benchmarks.md`.

Report the honest number. Per-timestep identification is already ~1.5–2× faster since the
scipy Dijkstra swap, and process startup plus pickling a `PolyData` per timestep is real
overhead — a modest or even negative speedup on small inputs is a legitimate result and should
be written down as one.

- [ ] **Step 9: Full verification and commit**

Run: `ruff check . && mypy waper && pytest -m "not slow" -q`
Expected: **161 passed** (156 + the 5 non-slow tests of this task) / 1 skipped, and the
real-data test either skipped or, with the file present, deselected as `slow`.

```bash
git add waper/interface/api.py tests/test_parallel_identify.py results/benchmarks.md
git commit -m "perf(identification): optional parallel timestep processing"
```

---

### Task 7: Retire the superseded plans

**Why:** With Tasks 1–6 done, `waper_refactoring_spec.md` has no live remainder, and
`2026-08-07-housekeeping-backlog.md` has none either — its "needs Joy's input" register is
reproduced at the top of this plan. Leaving three overlapping documents in
`plans/implementation/` is how the next session re-derives all of this.

**Files:**
- Move: `superpowers/plans/implementation/waper_refactoring_spec.md` →
  `superpowers/plans/archive/implementation/`
- Move: `superpowers/plans/implementation/2026-08-07-housekeeping-backlog.md` →
  `superpowers/plans/archive/implementation/`
- Modify: `superpowers/plans/archive/README.md`

**Interfaces:**
- Consumes: Tasks 1–6 being complete. **Do not run this task early.**

- [ ] **Step 1: Confirm there is nothing left to close**

Re-read the spec's Phases 6–9 and confirm each task is either done by this plan, closed
earlier, or on the out-of-scope register. Expected disposition:

| Spec task | Disposition |
|---|---|
| 6.1, 6.2 | Closed earlier — absorbed by tasks 5.3 and 2.4 |
| 6.3 | Done — Task 5 |
| 7.1, 7.2, 7.4 | Done — Task 4 |
| 7.3 (Hovmöller) | Deferred by Joy, 2026-08-12 |
| 8.1, 8.3 | Closed earlier — Quarto site and `docs/algorithm.md` |
| 8.2 | Done — Tasks 1 and 2 |
| 9.1 | Closed 2026-08-07 — quadtree deleted |
| 9.2 | Done — Task 6 |
| 9.3 | Retired as superseded by `waper/io/` |
| 9.4 | Done — Task 3 |

If any spec task does not fit a row above, **stop and report it** rather than archiving a
document with live work in it.

- [ ] **Step 2: Move the files with `git mv`**

```bash
git mv superpowers/plans/implementation/waper_refactoring_spec.md \
       superpowers/plans/archive/implementation/
git mv superpowers/plans/implementation/2026-08-07-housekeeping-backlog.md \
       superpowers/plans/archive/implementation/
```

- [ ] **Step 3: Update the archive index**

In `superpowers/plans/archive/README.md`, delete the two bullets for these files under **"What
deliberately stayed active"** and add two rows to the **"Retired because the work shipped"**
table, following the existing column format (`Plan | Delivered | Evidence in tree`). The
evidence column must name real artifacts — `waper/interface/projections.py`,
`tests/test_parallel_identify.py`, `results/benchmarks.md`, the ruff `D` gate in
`pyproject.toml` — not "see the plan".

Add one sentence recording where the register went: the housekeeping backlog's "needs Joy's
input" items live on in
`superpowers/plans/implementation/2026-08-12-engineering-backlog.md`.

- [ ] **Step 4: Confirm what is left active**

Run: `ls superpowers/plans/implementation/ superpowers/plans/design/`
Expected in `implementation/`: exactly this plan and `phase0_implementation_plan.md` (the
science gate, which is not engineering work and stays). `design/` is unchanged.

- [ ] **Step 5: Commit**

```bash
git add -A superpowers/plans/
git commit -m "docs(plans): retire the refactoring spec and housekeeping backlog"
```

---

## Verification summary

Expected suite growth, task by task:

| After task | Command | Expected |
|---|---|---|
| baseline | `pytest -m "not slow"` | 144 passed, 1 skipped, 4 deselected |
| 1, 2 | `ruff check .` | zero `D` violations tree-wide |
| 3 | `pytest -m "not slow"` | 150 passed |
| 4 | `pytest -m "not slow"` | 156 passed |
| 5 | `pytest -m slow tests/test_benchmark.py` | 2 passed, timings recorded |
| 6 | `pytest -m "not slow"` | 161 passed |
| 6 | `pytest -m slow tests/test_parallel_identify.py` | passes, or skips without the data file |

Counts are expectations, not contracts — if a task lands a different number of tests, say so
in the report rather than padding to hit the table.
