# Regimes → RWP Structure: a WAPER analysis of Shah & Monteiro (2025)

> **Context:** A novel-science application of WAPER, sibling to the western-disturbance plan (`superpowers/plans/design/western_disturbance_validation_plan.md`) — same region, same data, same subtropical tuning. Depends on Phase 0 passing and on the §4 subtropical threshold tuning in the WD plan. Treat as part of **Layer 5**.

## 1. What Shah & Monteiro (2025) did

Shah & Monteiro (2025, *Weather Clim. Dynam.* 6:1699–1721, "The role of synoptic circulations in lower-tropospheric dry static energy variability over a South Asian heatwave hotspot"):

- **Region:** NW South Asian heatwave hotspot, **25–31°N, 68–78°E** (this sits *inside* Hunt's WD box, 20–36.5°N / 60–80°E).
- **Season/data:** March–April, **1980–2022, ERA5 (0.25°)**, daily means.
- **Quantity:** lower-tropospheric (600–900 hPa) **dry static energy (DSE)**; daily changes `δS` are dominated by horizontal/vertical **advection** by synoptic eddies and strongly track near-surface temperature (heatwave onset/decay).
- **Decomposition (Reynolds):** they split advection into **quasilinear (mean–eddy)** terms — `v′S̄_y`, `ū S′_x`, `w′S̄_z` — and **nonlinear (eddy–eddy)** terms — `v′S′_y + w′S′_z + u′S′_x`.
- **Decision-tree regimes:** the sign-combination of the two largest quasilinear terms (`ū S′_x`, `w′S̄_z`) defines a discrete set of **regimes** for daily DSE change (their Table 1):
  - **Negative** (cooling): `ū S′_x ≤ 0`, `w′S̄_z ≤ 0`
  - **Positive** (warming): `ū S′_x > 0`, `w′S̄_z > 0`
  - **Neutral**: the out-of-phase combinations (two sub-types, "Neutral1" / "Neutral2")
- **Nonlinear tails:** the eddy–eddy terms govern the **extreme** deciles of `δS`; the specific nonlinear component that matters depends on the phase of growth/decay and the sign of the pre-existing DSE anomaly. These are the "energetically distinct configurations" of the abstract.
- **The schematic (their Fig. 3) is a Rossby wave.** A sinusoidally displaced geopotential contour with **cyclones at the troughs and anticyclones at the ridges** (C–W–C). The regime a given day falls into is set by **where the hotspot sits in the wave phase** — at the head of the cold anomaly (case 1) vs the head of the warm anomaly (case 2). The eddy fields are **barotropic and strengthen with height**, and `w′` tracks QG-omega — i.e. **upper-tropospheric wave dynamics drive the surface DSE response.**
- Their preprocessing/diagnostic code is released (Hardik, 2025).

## 2. The analogy and the goal

In the Pandey et al. (2020) re-examination of Monteiro & Caballero (2019), a closer look at the **RWP structure** revealed that the *same* extreme outcome (humid-heat extremes) averaged over **at least two distinct mechanisms**, depending on the **phase of the wave packet** over the region — so a naive composite hides the real diversity.

> **Goal:** do the same for Shah & Monteiro. Map each DSE regime to its characteristic **RWP structure**, and test whether the *same* outcome (e.g. extreme positive `δS`) is produced by **different packet structures**. Replace the implicit, term-sign definition of a regime with an **explicit packet-structure characterization**.

## 3. Why WAPER is the right tool

Shah & Monteiro define regimes by **local advection-term signs**, which encode the eddy phase relative to the hotspot only *implicitly*. WAPER makes that phase **explicit**: it identifies the actual trough/crest nodes (cyclones/anticyclones) and the region's position within the packet. So WAPER turns "the region is at the head of the warm anomaly" into "the hotspot lies one quarter-wavelength downstream of trough node *k* in a 5-node packet of amplitude *A* that formed over the eastern Mediterranean three days ago." That is exactly the structural detail their term-based regimes cannot express — and it is WAPER's native output (explicit trough–crest links + tracked packets).

## 4. Connection to the western-disturbance plan

The hotspot is inside the WD box, and the **cyclone in the C–W–C schematic is a western disturbance** (a trough node in WAPER). So WAPER unifies the two South-Asia analyses:
- **WD** = the trough node passing the region (WD plan, Part A/B).
- **Heatwave regime** = the hotspot's **phase position relative to that node** (this plan).

A Positive (warming) regime should correspond to the region sitting **ahead of a ridge / behind a trough** (southerly, warm advection); a Negative regime to the region **ahead of a trough** (northerly, cold advection). WAPER can test these correspondences directly and per-event.

## 5. Analysis steps

### 5.1 Obtain regime labels per day
Use Hardik (2025)'s released code to label each March–April day 1980–2022 as Negative / Neutral / Positive (and flag the extreme-decile, nonlinear-tail days). The three quasilinear terms and the nonlinear terms are computable directly from ERA5, so this is reproducible independently if needed. Output: a daily table `date → regime, δS, decile, dominant_term`.

### 5.2 Run WAPER over the same days and characterize packet structure
On **daily-mean ERA5 300 hPa v** (the upper-level wave driver — justified by their barotropic Fig. S4), subtropical-tuned per the WD plan §4, for each labelled day compute, for the packet over/upstream of the hotspot:
- **Phase over the region:** nearest node type (trough/crest) and the region's fractional position between adjacent nodes (the explicit version of "head of warm/cold anomaly").
- **Amplitude** (peak |v| of the governing nodes), **number of nodes**, **zonal extent**, **implied wavelength/wavenumber**.
- **Embeddedness** (isolated eddy vs node in a long coherent packet) and **upstream provenance** (genesis location via the tracking graph) — reuse WD plan §5.1–5.2.
- **Group velocity** of the parent packet.

> **Eddy-definition note:** Shah define eddies as anomalies from a 10-day running mean. At 300 hPa the climatological-mean `v` is small (zonal flow), so raw `v` ≈ `v′` and WAPER on raw `v` is fine — but test feeding WAPER the `v` anomaly (`v − 10-day running mean`) for strict consistency with their eddy definition, and report whether it changes the packet identification.

### 5.3 Map regime ↔ packet structure
Composite/condition the §5.2 packet metrics by regime. Does each regime have a characteristic packet phase and structure? Produce the regime→structure table and the phase distributions per regime. **Expectation:** Positive vs Negative regimes separate cleanly by region-phase (downstream-of-ridge vs downstream-of-trough); Neutral by weaker / out-of-phase / out-of-region packets.

### 5.4 Multi-pathway test (the Monteiro & Caballero analogue)
Within a *single* outcome class — e.g. the extreme positive-`δS` decile — cluster the WAPER packet structures. Are there **distinct structural pathways to the same outcome** (e.g. region just downstream of a single deep isolated trough vs region embedded in a long coherent packet arriving from upstream)? If so, this is the explicit, packet-level demonstration of the multi-mechanism insight that Pandey/Monteiro–Caballero showed qualitatively — now systematic and quantified.

### 5.5 Quasilinear vs nonlinear ↔ packet structure
Their nonlinear (eddy–eddy) terms drive the extreme tails. Hypothesis: the **nonlinear-dominated extreme days correspond to large-amplitude, tightly-packed packets** (multiple strong nodes overlapping the region, where eddy–eddy interaction is strongest), whereas quasilinear days correspond to a single dominant node on the mean gradient. Test by relating node density/amplitude over the region to the quasilinear-vs-nonlinear partition.

### 5.6 Close the loop — mechanism validation
WAPER's explicit region-phase predicts the **sign of `v′S̄_y`** (region downstream of trough → northerly `v′` → cold advection of the climatological gradient). Check that WAPER's phase diagnosis recovers the advection sign / regime label day-by-day. Agreement validates both the WD→phase mapping and the physical interpretation; systematic disagreement flags either tuning issues or genuinely non-wavelike days (themselves interesting).

## 6. Caveats & dependencies
- **Vertical level (decided).** WAPER uses **300 hPa v** as the driver; Shah's `δS` is a 600–900 hPa response. The link is their barotropic, height-strengthening eddy structure (Fig. S4) and QG-omega-controlled `w′`. **Use 300 hPa for the regime verification — no mid-tropospheric variant.** State the assumption in the writeup, but do not branch on it.
- **Subtropical tuning is a prerequisite** (WD plan §4) — heatwave-regime packets are subtropical and weaker than midlatitude defaults expect.
- **Day alignment.** Regime labels (§5.1) and WAPER runs (§5.2) must be on identical ERA5 days; daily means, March–April, 1980–2022 (subset to a few seasons to start).
- **Eddy definition** (§5.2 note) — reconcile raw v vs v-anomaly.
- This is an *application* of a WAPER already validated globally (Layer 2); it is not itself a WAPER-correctness test.

## 7. Order
1. Subtropical tuning (WD plan §4) — shared prerequisite.
2. §5.1 regime labels (reproduce from Hardik 2025; cheap, no WAPER).
3. §5.2–5.3 packet characterization and the regime→structure map (the core result).
4. §5.4 multi-pathway test (the headline analogue of Monteiro & Caballero).
5. §5.5–5.6 nonlinear-tail structure and mechanism validation.

## 8. Relationship to other plans
- **`western_disturbance_validation_plan.md`:** shares region, ERA5 data, and the §4 subtropical tuning; the WD (trough node) and the heatwave regime (region phase relative to that node) are two faces of the same packet. Provenance/embeddedness reuse WD plan §5.1–5.2.
- **`validation_strategy_plan.md`:** Layer 5 application; depends on Phase 0 and builds on the Layer 2 global climatology (for provenance) and Layer 4 tracking quality.
