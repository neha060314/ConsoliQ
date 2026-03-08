import pandas as pd
from collections import defaultdict
from datetime import datetime

df = pd.read_csv("data/shipments.csv")
shipments = df.to_dict("records")

# Load the cache to get hub names without re-querying
import json, math, os

cache_file = ".geocache/nominatim.json"
with open(cache_file) as f:
    cache = json.load(f)

def key(lat, lng):
    return f"{round(lat, 3)},{round(lng, 3)}"

def hub_name(lat, lng):
    addr = cache.get(key(lat, lng), {})
    for field in ["industrial", "suburb", "neighbourhood", "city_district",
                  "town", "city", "district", "state_district", "county"]:
        val = addr.get(field, "").strip()
        if val:
            return val
    return f"{round(lat/0.2)*0.2:.1f}_{round(lng/0.2)*0.2:.1f}"

def parse_dt(s):
    return datetime.fromisoformat(s)

# Assign lanes
for s in shipments:
    s["_pickup_hub"]   = hub_name(s["pickup_lat"],   s["pickup_lng"])
    s["_delivery_hub"] = hub_name(s["delivery_lat"], s["delivery_lng"])
    s["_lane"]         = f"{s['_pickup_hub']} → {s['_delivery_hub']}"

# Group by lane
lanes = defaultdict(list)
for s in shipments:
    lanes[s["_lane"]].append(s)

print("=" * 70)
print("LANE DISTRIBUTION")
print("=" * 70)
by_size = sorted(lanes.items(), key=lambda x: -len(x[1]))
single_lane_count = sum(1 for _, v in by_size if len(v) == 1)
print(f"Total lanes      : {len(lanes)}")
print(f"Single-shipment  : {single_lane_count} ({100*single_lane_count/len(lanes):.0f}%)")
print(f"Multi-shipment   : {len(lanes) - single_lane_count}")
print()

print("Multi-shipment lanes:")
for lane, ships in by_size:
    if len(ships) < 2:
        break
    tws = [(parse_dt(s["earliest_delivery"]), parse_dt(s["latest_delivery"]),
            s["shipment_id"], s["weight_kg"]) for s in ships]
    tws.sort()
    print(f"\n  {lane}  ({len(ships)} shipments)")
    for e, l, sid, w in tws:
        print(f"    {sid}  {e.strftime('%H:%M')}–{l.strftime('%H:%M')}  {w:.0f}kg")
    # Check all pairs for overlap
    for i in range(len(tws)):
        for j in range(i+1, len(tws)):
            e1, l1, sid1, _ = tws[i]
            e2, l2, sid2, _ = tws[j]
            overlap = (min(l1,l2) - max(e1,e2)).total_seconds()
            print(f"    overlap {sid1}↔{sid2}: {overlap/3600:.1f}h {'✅' if overlap > 0 else '❌ NO OVERLAP'}")

print()
print("=" * 70)
print("PICKUP HUB DISTRIBUTION (corridor grouping potential)")
print("=" * 70)
hub_buckets = defaultdict(list)
for s in shipments:
    hub_buckets[s["_pickup_hub"]].append(s)

for hub, ships in sorted(hub_buckets.items(), key=lambda x: -len(x[1])):
    if len(ships) < 3:
        continue
    destinations = [s["_delivery_hub"] for s in ships]
    print(f"  {hub:<30} {len(ships):>3} shipments → {len(set(destinations))} unique destinations")
    for s in sorted(ships, key=lambda x: x["earliest_delivery"]):
        e = parse_dt(s["earliest_delivery"])
        l = parse_dt(s["latest_delivery"])
        print(f"    {s['shipment_id']}  {e.strftime('%H:%M')}–{l.strftime('%H:%M')}  "
              f"{s['weight_kg']:.0f}kg  → {s['_delivery_hub']}")