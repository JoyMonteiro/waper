# Identification & tracking assessments

Joy's running judgement on how well WAPER's **identification** and **tracking**
actually perform, read off the figure outputs and metrics.

This is the *judgement* track. `datasets/experiments/` is the *quantitative* track —
"we swept GT from 0.015 to 0.03, here are the tables." The two are complementary and
cross-link, but they are deliberately separate: sweep entries follow a strict
"numbers > prose" rule, and the reasoning captured here is exactly the prose that
rule discourages.

**Why this exists:** these assessments were being made in conversation and lost at
every restart, which meant re-proposing approaches that had already been examined and
rejected. The **Do-not-retry register** below is the point of the whole directory.

## Format

One file per working day: `YYYY-MM-DD.md`. Sections:

- **Verdict** — one line. Where identification/tracking stands today.
- **Wins** — what demonstrably improved. Cite the figure or metric and its path.
- **Open problems** — what is still wrong. Cite evidence; note whether it is a
  *tuning* problem (a threshold can fix it) or a *structural* one (the algorithm
  cannot express the answer). That distinction has already cost us once.
- **Ruled out** — approaches examined and rejected today, each with the reason.
  Every item here must also be promoted to the register below.
- **Next** — what to look at next.

Only record an assessment that was actually made. If a day's work produced no
judgement on identification or tracking quality, write no entry — an empty day is
information too. Where an entry carries forward a conclusion from an earlier session
rather than a fresh look, say so and cite the source.

## Do-not-retry register

Approaches that have been examined and rejected, with the reason. **Read this before
proposing a fix to tracking continuity or RWP over-extension.** Promote every "Ruled
out" item here so nobody has to read the full daily history to avoid a known dead end.

| Approach | Rejected | Why |
|---|---|---|
| Tuning `footprint_fraction` / `min_split_iou` / `max_recover_steps` to make a feature track follow a packet eastward | 2026-06-19 | **Structural, not tuning.** A feature track follows *phase*; the eastward march is the envelope moving at *group* velocity, carried by a succession of distinct troughs with no footprint lineage between them. No threshold can bridge that. See [2026-08-07](2026-08-07.md) §Ruled out. |
| Approach 2 of the envelope-segmentation proposal (split the RWP polygon post-hoc on a 1-D zonal energy profile) | 2026-06 | Architecturally backward — the polygon is a *derived* output of `get_ranked_paths`. Splitting it means reverse-engineering which nodes belong to which fragment and severing edges anyway, i.e. undoing work already done. Segment at the graph level instead. See `envelope_segmentation_proposal.md`. |
| Full graph-watershed segmentation (find envelope maxima per component, locate valleys, cut) | 2026-06 | Correct in principle but needs a post-hoc walk over assembled paths. Greedy merging happens edge-by-edge, and an edge-level pruning step already exists, so the simpler variant belongs there. See `envelope_segmentation_proposal.md`. |
| Thresholding the Zimin envelope in 2-D to define RWP extent | 2026-06 | The 1-D Hilbert transform fractures meridionally tilted packets across rigid latitude bands. The envelope is usable as a *node-sampled weight*, not as a spatial boundary. |

## Index

- [2026-08-07](2026-08-07.md) — baseline carry-forward: what we believe entering this log
