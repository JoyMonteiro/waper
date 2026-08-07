"""Phase vs group velocity check: is eastward motion carried by ONE trough
(phase) or by a SUCCESSION of distinct troughs appearing downstream (group)?"""
import pickle

with open("/tmp/diag_fb.pkl", "rb") as fh:
    fb, thr = pickle.load(fh)

top5 = sorted(fb[0], key=lambda f: abs(f.scalar), reverse=True)[:5]
print("Top-5 strongest features @ t0:")
for i, f in enumerate(top5):
    print(f"  #{i+1} {f.node_type} lon={f.lon:6.1f} lat={f.lat:5.1f} scalar={f.scalar:6.1f}")

purple = top5[2]
print(f"\nPurple = #{3}: {purple.node_type} @ lon={purple.lon:.1f}\n")

# For the first 18 h, list strong troughs (min) in the 30-55N band, sorted by lon.
# If group velocity is eastward, the *set* of trough longitudes marches east
# even though each individual trough is near-stationary.
print("Strong MIN (trough) longitudes in 30-55N band, per hour:")
for t in range(18):
    mins = [f for f in fb[t]
            if f.node_type == "min" and f.strength == "strong" and 30 <= f.lat <= 55]
    lons = sorted(round(f.lon) for f in mins)
    print(f"  t={t:2d}: {lons}")

# How many DISTINCT strong troughs does any single eastward-propagating
# packet envelope pass through? Track the easternmost trough each step.
print("\nEasternmost strong trough (30-55N) per hour, lon & lat:")
for t in range(30):
    mins = [f for f in fb[t]
            if f.node_type == "min" and f.strength == "strong" and 30 <= f.lat <= 55]
    if not mins:
        print(f"  t={t:2d}: none"); continue
    e = max(mins, key=lambda f: f.lon if f.lon < 180 else f.lon - 360)
    print(f"  t={t:2d}: lon={e.lon:6.1f} lat={e.lat:5.1f} scalar={e.scalar:6.1f}")
