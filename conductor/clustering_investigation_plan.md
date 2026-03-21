# Clustering Algorithm Investigation: OPTICS Parameter Tuning

> **Context:** This investigation should be picked up after the current refactoring work (Phases 4+) is complete and the pipeline can be tested against real data.

## Background

The current implementation uses OPTICS (`max_eps_km=1500`, `min_samples=1`, `xi=0.05`) to cluster extrema within connected components of the level set. This replaced DBSCAN (which itself replaced Affinity Propagation) in Phase 3. OPTICS was chosen because it discovers clusters at varying density scales without committing to a single fixed radius, which better handles the multi-scale structure of RWPs.

The default parameters need validation and tuning on real data. The key questions are whether `min_samples=1` provides sufficient density-based separation, and whether the `xi` parameter needs adjustment.

## Investigation Steps

### Step 1: Identify failure cases with current DBSCAN

Run the full pipeline on a multi-day real dataset (ideally spanning different synoptic regimes — a blocking event, a short-wave train, and a transitional period). Visually inspect the clustering output and identify timesteps where:

- (a) Distinct wave components are incorrectly merged (eps too large)
- (b) A single wave component is fragmented into multiple clusters (eps too small)
- (c) Both (a) and (b) occur simultaneously in different parts of the domain

Save 3-5 representative failure cases as test fixtures for reproducible comparison.

### Step 2: Characterize the spatial scale variability

For the failure cases, measure:

- The inter-extremum distances within correctly-identified clusters
- The inter-extremum distances between clusters that were incorrectly merged
- The dominant zonal wavelength (from FFT of meridional wind along a representative latitude)

This establishes whether the problem is a single "wrong eps" or genuinely multi-scale structure in the same timestep.

### Step 3: Test OPTICS with min_samples=1

Using the failure-case fixtures, run OPTICS (`min_samples=1`, `max_eps=1500`, `xi=0.05`) with the same penalty-augmented distance matrix currently fed to DBSCAN.

Examine:
- The reachability plot — does it show clear valleys between the clusters that DBSCAN incorrectly merged?
- The extracted clusters — are they better than DBSCAN's output?
- Vary `xi` from 0.01 to 0.1 and check sensitivity.

**Key question to answer:** Does the reachability ordering capture useful structure even with `min_samples=1`, or is the plot essentially flat?

### Step 4: Test OPTICS with min_samples=2

If Step 3 shows promise but the reachability plot is too flat, try `min_samples=2`. This requires that single-extremum components have at least one nearby companion to survive.

Two sub-experiments:
- (a) Simply set `min_samples=2` and accept that isolated single-extremum components become noise. Check how many real features are lost.
- (b) Lower `extrema_threshold` slightly (e.g., from 5 to 4 m/s) to admit near-peak points as genuine extrema, giving small components more points. Then run with `min_samples=2`.

### Step 5: Test adaptive eps for DBSCAN (fallback)

If OPTICS doesn't improve results, test an adaptive eps scheme:

- For each connected component with >= 5 extrema: set `eps` to 1.5x the median pairwise inter-extremum distance within the component.
- For components with < 5 extrema: use the global default (500km) or a spectral estimate.

Compare against the fixed-eps results on the failure cases.

### Step 6: Decision

Choose based on the failure-case comparison:

| Outcome | Action |
|---------|--------|
| OPTICS with `min_samples=1` resolves failure cases | Switch to OPTICS, expose `xi` and `max_eps` in `WaperConfig` |
| OPTICS needs `min_samples=2` + lower threshold | Switch to OPTICS, adjust threshold, document the tradeoff |
| Adaptive eps matches OPTICS quality | Keep DBSCAN with per-component adaptive eps |
| Nothing helps significantly | Keep fixed eps, document the limitation |

## Design constraints

- Whatever the outcome, the clustering interface should remain `cluster_extrema(...)` returning labels — the downstream pipeline (centroids, association graph, pruning) should not need changes.
- The penalty-augmented distance matrix is the input to any algorithm. That stays.
- Any new parameters should be added to `WaperConfig` with sensible defaults that reproduce current behavior.

## Key references from our discussion

- OPTICS reachability plots capture inter-cluster gaps even with `min_samples=1`, but density gradients within clusters require higher `min_samples`.
- The companion-point idea (adding an adjacent grid point per extremum to guarantee min_samples=2) was considered but risks artificial density inflation, centroid bias, and micro-clustering artifacts. Lowering the extrema threshold is a cleaner way to increase point density.
- Adaptive eps effectively reimplements what OPTICS does natively — worth trying only if OPTICS itself doesn't help.
