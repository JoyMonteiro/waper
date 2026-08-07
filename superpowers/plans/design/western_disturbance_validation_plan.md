# WAPER for Western Disturbances — Validation & Beyond-Hunt Plan

> **Context:** A domain-specific application of WAPER to western disturbances (WDs), benchmarked against Hunt, Turner & Shaffrey (2018, *QJRMS* 144:278–290). This sits alongside the layers in `superpowers/plans/design/validation_strategy_plan.md` — treat it as **Layer 5 (applied extension + novel science)**. It depends on Phase 0 passing and on the subtropical-waveguide threshold tuning discussed below.
>
> **Framing (important):** the aim is **not to validate WAPER against Hunt's results — it is to update them.** WAPER's own correctness is established globally in Layer 2 (against the Souders climatology). Here we apply that validated tool, on **modern global ERA5** (not ERA-Interim), to produce an improved and richer WD climatology than Hunt could with a cyclone-only tracker. Hunt's numbers are the benchmark we expect to *recover-then-exceed*, not a ground truth we must match.

---

## 1. What Hunt et al. (2018) do — and the gap WAPER fills

### Their method (verified from the paper)
- Track in the **450–300 hPa layer-mean relative vorticity** ξ, spectrally truncated to **T63** (~200 km).
- Locate local maxima of ξ separated by a radius **δ = 850 km**; integrate positive ξ around each to get a centroid.
- Link centroids frame-to-frame by nearest neighbour within **Δ = 1000 km (6 h)⁻¹**, with a **background-wind advection bias** on the search; hold tracks across 1 gap timestep.
- Filter tracks: reject stubs < 2 days; **reject tracks that don't pass through the north-India box (20–36.5°N, 60–80°E)**; reject tracks whose genesis is east of lysis (must move eastward).
- Data: 37 yr ERA-Interim (0.7°), 6-hourly → catalogue of **3090 WDs, ~6–7 per month**.
- Composite framework: vertical structure (NW tilt with height, warm-over-cold, dry-over-moist, PV max at tropopause — classic baroclinic), peak WD meridional wind ~10 m/s, jet ~40 m/s at 200 hPa.
- **k-means classification** (k=4) on dynamical fields → 4 types by intensity and **wavelength** (type 3a ~2300 km, type 3b ~1600 km); separately k-means on TRMM precipitation → non-precipitating (N) vs precipitating (P) types.
- **§3.5 jet interaction:** define p(in jet) = probability a 200 hPa point has u>0 and |u|>30 m/s. High-WD winters have the jet shifted south and **"substantially more coherent, reaching as far west as North America."**

### What Hunt's tracker cannot do (WAPER's structural advantage)
1. **It tracks single cyclonic vorticity maxima — troughs only.** It has no representation of the accompanying ridges, hence no wave packet. WAPER detects both v-maxima and v-minima and **explicitly links troughs to crests** (max–min edges).
2. **A WD is an isolated feature in Hunt's catalogue.** There is no notion that a WD is one node within a larger coherent Rossby wave packet. The connection to upstream/downstream wave activity exists only as a hemispheric composite statistic (§3.5), never per-event.
3. **No packet-level diagnostics:** no group velocity vs phase speed, no packet extent, no downstream-development energetics, no explicit provenance (where upstream did the packet come from).
4. The §3.5 jet-coherence result is the smoking gun: the thing that distinguishes high-WD winters is *Rossby-wave coherence reaching back to North America* — precisely what WAPER measures directly and Hunt can only infer.

### The conceptual alignment (the key idea)
A WD is a cyclonic vortex embedded in the subtropical westerly jet. In the 300 hPa meridional wind, that vortex sits at the **v zero-crossing between a southerly lobe (v-max, to its east/ahead) and a northerly lobe (v-min, to its west/behind)** — consistent with Hunt's Fig. 4 (northerly lobe aloft, southerly below/ahead). Therefore:

> **A Hunt WD center corresponds to the midpoint of a WAPER max–min edge in the subtropical waveguide.** WAPER does not merely re-detect the WD — it natively expresses the WD as a wave-phase pair and embeds it in a tracked packet.

This is also why the subtropical waveguide is the right place to look: **Chang & Yu (1999) found their *maximum* wave-packet coherence over southern Asia** — i.e. the WD waveguide is one of the most coherent RWP channels in the NH, so WAPER should detect it well.

---

## 2. Part A — Event-level cross-check against Hunt's catalogue (deferred)

> **Priority:** deferred. Hunt's catalogue is expected to be available but **not within ~1 month** (per project timeline). So this part is **not on the critical path** — do Parts B and C first (they only need Hunt's *published* numbers, not his raw tracks). Treat Part A as a later cross-check that locates *where WAPER and Hunt agree and diverge*, not as a gate.

**Goal:** quantify, event by event, how WAPER's packet-based WDs relate to Hunt's cyclone-based catalogue — including the WDs WAPER sees that Hunt misses (non-cyclonic / weak / wrong-phase), which are scientifically interesting in their own right.

**Data dependency:** Hunt's WD track catalogue (a feature-tracked database derived from reanalysis). Obtain when available; do **not** re-implement his tracker just to unblock — Parts B/C proceed without it.

**Matching protocol (per 6-hourly timestep):**
- Run WAPER on **ERA5** 300 hPa v over the region/period, subtropical-tuned thresholds from §4.
- For each Hunt WD center `(lon, lat, t)`, mark it **captured** if it falls inside a WAPER RWP polygon at `t`, or within a small radius (~δ = 850 km) of a WAPER max–min edge midpoint.
- Empirically confirm the WD→edge phase mapping on a handful of cases first (which is the v-max, which the v-min relative to the cyclonic center).

**Metrics:**
- **Capture rate (POD-analogue):** fraction of Hunt WDs that lie within a WAPER RWP footprint. Restrict to WDs inside the north-India box and the WD season (Dec–Apr).
- **Reverse rate:** fraction of WAPER RWPs intersecting the box that contain a Hunt WD (the complement flags WAPER packets Hunt misses — candidates for "non-cyclonic" or weak WDs, scientifically interesting).
- **Centroid offset distribution:** distance between Hunt WD center and the nearest WAPER edge midpoint; expect a peak within a few hundred km.
- **Track-level agreement:** for Hunt tracks ≥ 2 days, does a WAPER tracked RWP shadow it (temporal overlap, co-propagation)? Report duration and eastward-propagation agreement.

**Acceptance:** high capture rate (target > 0.7–0.8 for box-passing WDs in season) with a tight centroid-offset distribution. Disagreements are catalogued, not hidden — they drive the Part C science.

---

## 3. Part B — Climatological / structural validation against Hunt's published numbers

Reproduce Hunt's bulk statistics with WAPER restricted to the subtropical waveguide / north-India box, as distribution-shape and tolerance-band checks (same philosophy as `validation_strategy_plan.md`):

| Quantity | Hunt target | How WAPER reproduces it | Source |
|---|---|---|---|
| Event frequency | ~6–7 WDs/month; 3090 over 37 yr | WAPER edges/RWPs intersecting the box per month | §2.2, §3 |
| Seasonal cycle | peak ~early Feb, min ~early Aug; season Dec–Apr; strongest third almost absent in JJA | monthly WAPER RWP activity in the box | Fig. 3 |
| Wavelength | type 3a ~2300 km, 3b ~1600 km; weak wavelike structure ~7000 km | WAPER edge length / node spacing → wavelength | §3.2, §3.4 |
| WD meridional wind | peak composite ~10 m/s | WAPER node `scalar` (|v|) at WD-edge nodes in the subtropics | §3.2.1.2 |
| Genesis distribution | Arctic→Caribbean, mostly Arabia/N Africa/Med, subset N America | genesis of WAPER RWPs whose downstream node enters the box | §3, §2.2 |
| Eastward propagation | required by construction; substantial zonal extent | WAPER track propagation (deg lon) for box-entering packets | §2.2 |

**Acceptance:** WAPER's seasonal cycle and wavelength distributions match Hunt's shapes; counts land in the right band. A correct seasonal cycle here is the strongest single check, because it is relative (bias-immune).

---

## 4. Subtropical-waveguide tuning (a prerequisite)

WDs live on the STWJ at ~25–35°N — lower latitude and **lower amplitude** than the midlatitude storm track WAPER is tuned for. So:
- The default `node_pruning_threshold = 20` m/s (ST) may be too high for subtropical packets — sweep downward (e.g. 12–20 m/s) and check capture rate against Hunt (Part A) as the objective function. This is a targeted reuse of Phase 0 Step 0.3.
- Confirm `min_latitude` reaches low enough (Hunt's box goes to 20°N; WAPER default `min_latitude=20` is fine).
- Keep `edge_pruning_threshold` in the 0.0–0.08 (m/s)/km range; subtropical edges are shorter/weaker, so verify they survive pruning.
- Record the chosen WD-regime parameter set separately from the midlatitude default.

---

## 5. Part C — Going beyond Hunt (the novel science WAPER enables)

These are only possible because WAPER computes explicit trough–crest links and tracks packets globally. Each is a publishable extension of Hunt.

### 5.1 Explicit RWP embedding of every WD
For each captured WD, identify the **full parent RWP**: number of nodes upstream/downstream of the WD-edge, total packet zonal extent, packet peak amplitude. Define an **"embeddedness" index** — is the WD an isolated vortex or one node in a long coherent packet?
- **Hypothesis (testable):** Hunt's high-impact WDs (type P, the long-wavelength type 3a, the heavy-rain events) are disproportionately the ones embedded in long, coherent, upstream-connected RWPs. Hunt could only classify by local dynamical structure; WAPER classifies by *packet context*.

### 5.2 Upstream provenance — linking WDs to the global RWP climatology
This is the centerpiece, and the direct answer to "linking WDs to global RWP distribution."
- Trace each WD's parent RWP backward through the tracking graph (including merges) to its **genesis location**.
- Build the distribution of WD-parent-packet genesis longitudes and compare to **Souders et al. (2014) global RWP genesis hotspots** (N Pacific >70% of extreme RWPs; W Atlantic intensification). Question: *do WD-producing packets preferentially originate at the same hotspots that dominate the global extreme-RWP climatology?*
- This makes Hunt's §3.5 result **explicit and per-event**: instead of "high-WD winters have a jet coherent to North America," WAPER states "X% of WDs belong to packets traceable to Atlantic/North-American genesis," and correlates that fraction with WD intensity/rainfall.
- It is a genuine bridge between two previously separate climatologies — the global RWP climatology (Souders) and the regional WD climatology (Hunt) — enabled precisely by WAPER's explicit trough–crest chain.

### 5.3 Downstream-development energetics
WAPER yields group velocity vs phase speed and packet growth (Chang & Yu physics).
- Test whether WDs intensify via **downstream development** — energy fluxing from the upstream ridge into the WD trough — by checking whether WD amplification coincides with the arrival/strengthening of the upstream packet node.
- Hunt observes that WDs "intensify suddenly over India" and attributes it to orography; WAPER can disentangle the orographic trigger from the **upstream wave-energy supply**, which Hunt's tracker cannot see.

### 5.4 Predictability / precursor lead time
Hunt's conclusion is that WD statistics may be predictable from jet position. WAPER sharpens it:
- If a WD is a node in an RWP that formed over North America/the Atlantic days earlier, the explicit packet provides a **physically-based precursor with a measurable lead time**.
- Quantify the lead time between upstream packet genesis and WD entry into the north-India box. This is a concrete, mechanism-based predictability statement beyond Hunt's correlation with jet latitude.

### 5.5 Direct, packet-based reclassification
Hunt's k-means types 3a/3b are essentially a **wavelength** distinction inferred from composite fields. WAPER measures wavelength directly (node spacing) and adds packet context.
- Replace the k-means wavelength proxy with a direct packet wavenumber, and add an orthogonal axis: *isolated vs embedded*, *locally-forced vs upstream-connected*. Then re-examine the WD→precipitation relationship (Hunt's type P/N) under this richer, physically-grounded classification.

### 5.6 Both wave phases → the ridges Hunt discards
WAPER captures the ridge ahead of (downstream-development seed) and behind (upstream energy source) each WD.
- Relate the **downstream ridge** to where the *next* WD forms (serial WD outbreaks), and the **upstream ridge** to the energy source. This phase information is structurally absent from Hunt's cyclone-only catalogue.

---

## 6. Honest caveats & dependencies

- **Field mismatch is minor.** Hunt tracks 450–300 hPa relative vorticity (T63); WAPER uses 300 hPa v. In the strongly **zonal** subtropical jet, relative vorticity is dominated by `∂v/∂x` (the cross-stream gradient of the meridional wind), so v carries essentially the same WD signal — the WD→edge mapping (§1) is sound. Still worth a quick **sanity confirmation** on a few cases, but this is not a serious obstacle and should not slow Parts B/C.
- **Single-signed vs alternating.** Vorticity is naturally single-signed for cyclonic WDs, which is why Hunt sees only troughs; WAPER's alternating-extrema graph is built for v, which is the right field for packets. Keep WAPER on **v** — do not force it onto vorticity.
- **Amplitude regime (must plan for).** Subtropical WD packets are **weaker and smaller-amplitude than the midlatitude RWPs** WAPER's defaults target. Without the §4 tuning, WAPER will under-detect WDs. **Tune the subtropical thresholds first** — this is a required, scheduled step, not an afterthought.
- **Data.** Use **ERA5** throughout (global, 0.25°), at the resolution Phase 0 recommends. We are deliberately *not* using ERA-Interim: the goal is to update Hunt with better, modern data, not to reproduce his exact period. Differences from Hunt's numbers are a feature (the update), interpreted as such.
- **Domain of tracking.** Hunt filters to box-passing tracks; WAPER detects globally. For Part B/A restrict to packets intersecting the box; for Part C (provenance) you *want* the global tracks — that's the whole point.

## 7. Suggested order

1. **§4 tuning** on a WD season (subtropical thresholds, ERA5) — prerequisite; reuses Phase 0 Step 0.3.
2. **Part B** — recover Hunt's bulk statistics with ERA5 (no catalogue needed); confirm WAPER reproduces the seasonal cycle/wavelength, then report the *updated* numbers.
3. **Part C** — novel analyses. Start with §5.2 (provenance → link to the Souders global climatology), the highest-impact result, then §5.1 embeddedness and §5.4 predictability.
4. **Part A** — event-level cross-check against Hunt's catalogue, **once it is available** (expected > 1 month out). Not on the critical path.

## 8. Relationship to other plans
- **`validation_strategy_plan.md`:** this is Layer 5. §4 tuning reuses Phase 0 Step 0.3; Part C.2 builds directly on Layer 2 (the Souders global climatology) and Layer 4 (tracking quality / POD-FAR, which underwrites the provenance tracing).
- **`regime_rwp_structure_plan.md`:** sibling Layer 5 analysis over the same NW-South-Asia region. The WD (trough node) and the Shah & Monteiro heatwave regime (the hotspot's phase relative to that node) are two faces of the same packet; the two plans share the subtropical tuning and the §5.1–5.2 embeddedness/provenance machinery.
- **`clustering_investigation_plan.md`:** subtropical packets stress the clustering/connection step at low amplitude — a good additional test case for that investigation.
