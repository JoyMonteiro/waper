# Benchmark baselines

Timings produced by `tests/test_benchmark.py`. Reproduce with:

```bash
pytest -m slow tests/test_benchmark.py -s -v
```

**These are synthetic-field numbers and are not comparable to ERA5 runs.** The input is an
analytic wavenumber-6 field with a Gaussian latitudinal envelope — smooth, noise-free, and far
sparser in extrema than reanalysis. It exists to catch order-of-magnitude regressions in the
pipeline, not to predict how long a real run takes.

## 2026-08-13 — initial baseline

| Quantity | Value |
|---|---|
| Machine | Apple M3 Pro, 11 cores, Darwin arm64 |
| Python | 3.12 (conda env `waper`) |
| Commit | `0753026` (`test(viz): cover the polygon data CRS; drop the NH-only POLYGON_CRS alias`) |
| Grid | 121 × 240 (1.5°, 20–80°N) |

| Benchmark | Time |
|---|---|
| Identification, 1 timestep | **0.52 s** |
| Tracking, 10 timesteps (identification excluded) | **0.07 s** |

Ceilings in the tests are 120 s and 60 s — deliberately ~100× the measured value. They catch a
pipeline that has gone quadratic; they are not a performance contract, and they are loose
enough not to flake on a shared CI runner. The tests are marked `slow`, so `pytest -m "not
slow"` skips them.

### Note on the field amplitude

The field is generated at amplitude 30 against a `extrema_threshold` of 10. At amplitude 20 —
the value first tried — the envelope-modulated peaks sit close enough to the threshold that a
10° phase advance drops every packet after the first timestep: identification returned 3 paths
at *t*=0 and 0 thereafter, and the resulting tracking graph had 3 nodes and **no edges**. The
tracking benchmark then reported 0.00 s, which is the honest cost of tracking nothing.

Both tests now assert that packets (and, for tracking, edges) actually exist, so the benchmark
cannot silently degenerate back into measuring an empty graph.

### What is not measured here

- Rasterisation and energy-raster construction are inside the identification number, not broken
  out separately.

## 2026-08-13 — parallel identification (`identify_rwps(n_jobs=...)`)

Same machine: Apple M3 Pro, macOS. **Its 11 cores are not interchangeable — 5 are performance
cores and 6 are efficiency cores** (`sysctl hw.perflevel0/1.physicalcpu`). `os.cpu_count()`
reports 11 and does not distinguish them, so `n_jobs=-1` spawns 11 workers and puts 6 of them
on the slow cores. That matters below.

**Run-to-run variance on this machine is ±15%** (thermal state, page cache). Rows measured in
different runs are not directly comparable; the two tables below are each internally consistent.

### Is parallelism worth it at all?

| Input | Sequential | `n_jobs=-1` | Speedup |
|---|---|---|---|
| Synthetic, 10 steps @ 121×240 | 5.08 s | 12.56 s | **0.40×** (2.5× slower) |
| ERA5 `forecast_bust_hourly`, 12 steps @ 90×360 | 8.07 s | 14.47 s | **0.56×** (1.8× slower) |
| ERA5 `forecast_bust_hourly`, 48 steps @ 90×360 | 29.21 s | 17.88 s | **1.63×** |

**Parallelism is a loss on short runs and a win on long ones.** Fitting the two ERA5 rows gives
a fixed pool cost of roughly **14–15 s** and a per-timestep cost of ~0.6 s, putting break-even
at about **25 timesteps**. Below that, `n_jobs=1` — the default — is faster, and it is the
default for exactly this reason.

The fixed cost is process startup, not pickling: macOS spawns rather than forks, so every
worker re-imports the whole `pyvista` / `cartopy` / `waper` stack before doing any work. It is
paid once per pool regardless of how many timesteps follow, which is why the speedup climbs
with run length and would keep climbing past 48 steps. On a fork-based Linux runner the
crossover should sit much lower; that has not been measured.

### How many workers? `n_jobs=-1` is not the best choice here

48 ERA5 timesteps, all three measured in one run:

| Setting | Time | Speedup |
|---|---|---|
| `n_jobs=1` (sequential) | 24.47 s | 1.00× |
| `n_jobs=5` (performance cores only) | **11.23 s** | **2.18×** |
| `n_jobs=-1` → 11 (all cores) | 12.94 s | 1.89× |

**Matching `n_jobs` to the performance-core count beats `n_jobs=-1`.** Six extra workers on
efficiency cores do not add throughput; they add six more spawns of the import stack, and
because `executor.map` hands out work in even chunks, the E-core workers finish last and the
pool waits on them. `n_jobs=8` measured 11.96 s in a separate run — indistinguishable from
`n_jobs=5`, consistent with the E-cores contributing nothing useful.

So on heterogeneous ARM machines, prefer an explicit `n_jobs` equal to the P-core count over
`n_jobs=-1`. `-1` remains a reasonable default on homogeneous x86.

Even the best case, 2.18×, is far short of 5× — the run is only ~25 s long, so ~14 s of pool
startup still dominates. The speedup is a function of run length, not of core count.

### Correctness, not just speed

`tests/test_parallel_identify.py::test_parallel_matches_sequential_on_real_data` (marked
`slow`) asserts that `n_jobs=-1` reproduces the sequential `raster_data` and `energy_raster`
**exactly**, via `assert_array_equal`, on 4 timesteps of ERA5. It passes. Parallelism here is a
scheduling change only; it does not perturb the numerics.
