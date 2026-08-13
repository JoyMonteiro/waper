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
- Parallel identification (`n_jobs > 1`) is not covered — those numbers are appended by the
  parallel-identification task.
