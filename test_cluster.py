"""
test_pipeline.py — Full pipeline diagnostic
clustering → route_compat → bin_packing

Shows what happened at each stage, why, and final savings.
"""

import pandas as pd
from collections import defaultdict

from engine.clustering  import get_valid_groups
from engine.geocoder    import cache_stats
from engine.route_compat import apply_route_filter, group_route_stats
from engine.bin_packing  import pack_groups, vehicle_mix, savings_vs_solo

# ── Load ──────────────────────────────────────────────────────────────
df        = pd.read_csv("data/shipments.csv")
shipments = df.to_dict("records")

print(f"\n{'═'*65}")
print(f"  🚛  LORRI FULL PIPELINE DIAGNOSTIC")
print(f"{'═'*65}")
print(f"  Input: {len(shipments)} shipments from data/shipments.csv")

# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"  STAGE 1 — CLUSTERING  (same origin + destination + time window)")
print(f"{'─'*65}")

clustered_groups = get_valid_groups(shipments)

# Geocoder health
stats = cache_stats()
print(f"\n  📍 Geocoder cache    : {stats['total']} entries "
      f"({stats['populated']} resolved, {stats['empty']} failed)")

# Lane distribution
lane_counts: dict = defaultdict(int)
for g in clustered_groups:
    lane_counts[g[0].get("_lane", "unknown")] += len(g)

multi_lanes  = sum(1 for v in lane_counts.values() if v > 1)
single_lanes = len(lane_counts) - multi_lanes

print(f"  🗺️  Lanes            : {len(lane_counts)} total  "
      f"({multi_lanes} multi-shipment, {single_lanes} single)")
print(f"\n  Top lanes:")
for lane, count in sorted(lane_counts.items(), key=lambda x: -x[1])[:8]:
    print(f"    {count:>3}  {'█' * count:<15}  {lane}")

# Grouping results
c_singletons   = [g for g in clustered_groups if len(g) == 1]
c_batched      = [g for g in clustered_groups if len(g) > 1]
c_exact        = [g for g in c_batched if not any(s.get("_multi_drop") for s in g)]
c_multidrop    = [g for g in c_batched if any(s.get("_multi_drop") for s in g)]
c_total_batched = sum(len(g) for g in c_batched)
c_trip_red     = (1 - len(clustered_groups) / len(shipments)) * 100

print(f"\n  📦 Groups            : {len(clustered_groups)} total")
print(f"     Batched           : {len(c_batched)}  ({c_total_batched} shipments)")
print(f"       • exact lane    : {len(c_exact)}")
print(f"       • multi-drop    : {len(c_multidrop)}")
print(f"     Singletons        : {len(c_singletons)}")
print(f"  📉 Trip reduction    : {c_trip_red:.1f}%  "
      f"({len(shipments)} → {len(clustered_groups)} groups)")

# Degraded coords
degraded_ids = list(dict.fromkeys(
    s.get("shipment_id", "?")
    for g in clustered_groups
    for s in g
    if s.get("_hub_degraded")
))
if degraded_ids:
    print(f"\n  ⚠️  Degraded coords  : {len(degraded_ids)} shipments → fix generate_data.py")
else:
    print(f"\n  ✅ All coordinates resolved cleanly")
