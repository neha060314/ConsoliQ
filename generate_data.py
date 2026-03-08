"""
generate_data.py  —  Realistic LoRRI shipment data generator
-------------------------------------------------------------
LoRRI is India's freight intelligence platform connecting manufacturers
with transporters for INTER-CITY industrial freight (>150km routes).

Key realism fixes:
- Dispatch waves clustered into morning/afternoon per hub (not random spread)
- Lane concentration: high-volume corridors dominate (Bhiwandi->Pune etc.)
- 80% of shipments on known high-volume lanes = natural consolidation
- Same-hub batching: multiple shipments share pickup zones naturally
"""

import argparse
import os
import random
from collections import defaultdict
import pandas as pd

# ── Pickup hubs ───────────────────────────────────────────────────────
PICKUP_HUBS = [
    (19.296, 73.066, "Bhiwandi Logistics Park",  22),
    (18.903, 73.111, "Taloja MIDC",              16),
    (18.658, 73.773, "Chakan Auto Cluster",      16),
    (18.534, 73.854, "Pimpri-Chinchwad MIDC",    14),
    (19.218, 72.978, "Thane Industrial Estate",  10),  # was 8
    (20.011, 73.765, "Nashik MIDC",               6),
    (21.170, 72.831, "Surat Textile Hub",         6),
    (22.307, 73.181, "Vadodara GIDC",             5),
    (22.980, 72.499, "Ahmedabad Naroda GIDC",     5),
    (12.971, 77.594, "Bangalore Peenya",          2),
]

# ── High-volume lane pairs (80% of shipments go on these) ─────────────
LANE_PAIRS = [
    ("Bhiwandi Logistics Park",  (18.520, 73.856, "Pune Distribution Centre"),    18),
    ("Bhiwandi Logistics Park",  (21.195, 72.830, "Surat Wholesale Market"),      10),
    ("Bhiwandi Logistics Park",  (23.022, 72.571, "Ahmedabad CFS"),                8),
    ("Bhiwandi Logistics Park",  (21.146, 79.088, "Nagpur Butibori"),              6),
    ("Taloja MIDC",              (18.520, 73.856, "Pune Distribution Centre"),    12),
    ("Taloja MIDC",              (21.195, 72.830, "Surat Wholesale Market"),       8),
    ("Taloja MIDC",              (17.385, 78.487, "Hyderabad"),                    6),
    ("Chakan Auto Cluster",      (21.195, 72.830, "Surat Wholesale Market"),      10),
    ("Chakan Auto Cluster",      (23.022, 72.571, "Ahmedabad CFS"),                8),
    ("Chakan Auto Cluster",      (17.385, 78.487, "Hyderabad"),                    6),
    ("Chakan Auto Cluster",      (13.067, 80.237, "Chennai"),                      4),  # fixed
    ("Pimpri-Chinchwad MIDC",    (21.195, 72.830, "Surat Wholesale Market"),       8),
    ("Pimpri-Chinchwad MIDC",    (23.022, 72.571, "Ahmedabad CFS"),                8),
    ("Pimpri-Chinchwad MIDC",    (17.385, 78.487, "Hyderabad"),                    5),
    ("Nashik MIDC",              (18.520, 73.856, "Pune Distribution Centre"),     6),
    ("Nashik MIDC",              (19.076, 72.877, "Mumbai BKC Warehouse"),         5),
    ("Surat Textile Hub",        (19.076, 72.877, "Mumbai BKC Warehouse"),         8),
    ("Surat Textile Hub",        (26.449, 80.331, "Kanpur"),                       4),
    ("Vadodara GIDC",            (19.076, 72.877, "Mumbai BKC Warehouse"),         5),
    ("Vadodara GIDC",            (18.520, 73.856, "Pune Distribution Centre"),     4),
    ("Ahmedabad Naroda GIDC",    (19.076, 72.877, "Mumbai BKC Warehouse"),         6),
    ("Ahmedabad Naroda GIDC",    (28.613, 77.209, "Delhi NCR Mundka"),             4),
    ("Thane Industrial Estate",  (18.520, 73.856, "Pune Distribution Centre"),     6),
    ("Thane Industrial Estate",  (21.195, 72.830, "Surat Wholesale Market"),       4),
    ("Bangalore Peenya",         (13.067, 80.237, "Chennai"),                      4),  # fixed
    ("Bangalore Peenya",         (17.385, 78.487, "Hyderabad"),                    3),
]

# Build per-hub lane lookup
LANES_BY_HUB = defaultdict(list)
for hub_name, dest, weight in LANE_PAIRS:
    LANES_BY_HUB[hub_name].append((dest, weight))

# ── Goods types ───────────────────────────────────────────────────────
GOODS_TYPES = [
    ("automotive_parts",      20),
    ("fmcg_packaged_goods",   18),
    ("construction_material", 15),
    ("textiles_apparel",      12),
    ("chemicals_industrial",  10),
    ("pharma_medical",         8),
    ("electronics",            8),
    ("steel_metal_parts",      9),
]

TRUCK_PREFERENCE = {
    "automotive_parts":      "heavy",
    "fmcg_packaged_goods":   "medium",
    "construction_material": "heavy",
    "textiles_apparel":      "medium",
    "chemicals_industrial":  "heavy",
    "pharma_medical":        "light",
    "electronics":           "light",
    "steel_metal_parts":     "heavy",
}

# ── Dispatch waves per hub ────────────────────────────────────────────
HUB_WAVE_WEIGHTS = {
    "Bhiwandi Logistics Park":  [0.65, 0.30, 0.05],
    "Taloja MIDC":              [0.60, 0.35, 0.05],
    "Chakan Auto Cluster":      [0.55, 0.40, 0.05],
    "Pimpri-Chinchwad MIDC":    [0.55, 0.40, 0.05],
    "Thane Industrial Estate":  [0.50, 0.45, 0.05],
    "Nashik MIDC":              [0.55, 0.40, 0.05],
    "Surat Textile Hub":        [0.35, 0.55, 0.10],
    "Vadodara GIDC":            [0.40, 0.50, 0.10],
    "Ahmedabad Naroda GIDC":    [0.60, 0.35, 0.05],
    "Bangalore Peenya":         [0.50, 0.40, 0.10],
    "_default":                 [0.50, 0.40, 0.10],
}

WAVES = {
    "morning":   (6  * 60,  11 * 60),
    "afternoon": (12 * 60,  17 * 60),
    "evening":   (17 * 60,  21 * 60),
}


def _weighted_choice(options):
    values  = [o[0] for o in options]
    weights = [o[-1] for o in options]
    return random.choices(values, weights=weights, k=1)[0]


def _hub_location(hub, radius=0.06):
    lat = round(hub[0] + random.uniform(-radius, radius), 4)
    lng = round(hub[1] + random.uniform(-radius, radius), 4)
    return lat, lng


def _weight(goods_type):
    """
    Realistic LTL/PTL/FTL weight distribution for Indian inter-city industrial freight.
    Previous version averaged 11t/shipment (near-FTL) leaving zero consolidation room.
    
    Target distribution:
      ~40% LTL  (<5,000 kg)    → 2-4 can share a 20ft/24ft truck
      ~35% PTL  (5,000-12,000) → 2 can share a 32ft/Volvo truck  
      ~25% FTL  (>12,000 kg)   → single truck, consolidation saves fixed costs only
    Expected avg: ~7,000 kg/shipment, total ~700t/100 shipments
    """
    tier = TRUCK_PREFERENCE.get(goods_type, "medium")
    roll = random.random()
    if tier == "light":
        # pharma, electronics — small parcels and instruments
        if roll < 0.60: return round(random.uniform(300,  3000), 1)
        if roll < 0.90: return round(random.uniform(3000, 7000), 1)
        return          round(random.uniform(7000, 12000), 1)
    elif tier == "medium":
        # FMCG, textiles, chemicals — cartons and drums
        if roll < 0.45: return round(random.uniform(800,  4500), 1)
        if roll < 0.80: return round(random.uniform(4500, 10000), 1)
        return          round(random.uniform(10000, 18000), 1)
    else:
        # heavy: automotive, construction, steel — parts and materials
        if roll < 0.30: return round(random.uniform(1500, 5000), 1)
        if roll < 0.65: return round(random.uniform(5000, 12000), 1)
        return          round(random.uniform(12000, 22000), 1)


def _dimensions(goods_type):
    """
    Cargo dimensions in cm (L × W × H) per shipment/pallet group.
    HARD CONSTRAINT: all dims must allow placement in at least one truck.
    Binding limit is truck interior WIDTH = 245cm and HEIGHT = 240cm.
    We use ≤220cm for both to leave 25cm handling clearance.
    Length is capped at 700cm (fits in any 32ft truck).

    Previous values had construction_material up to 800×400×300 = 96m³
    which is larger than the Volvo FH hold (79m³), making it physically
    impossible to bin-pack, forcing every shipment into its own truck.
    """
    if goods_type == "pharma_medical":
        # Boxes/cartons — compact and stackable
        return (random.randint(60,  200), random.randint(50,  150), random.randint(40,  120))
    elif goods_type == "steel_metal_parts":
        # Coils, sheets, billets — dense but bounded
        return (random.randint(120, 500), random.randint(80,  210), random.randint(60,  180))
    elif goods_type == "construction_material":
        # Pipes, panels, bags — long but must fit width
        return (random.randint(150, 600), random.randint(80,  210), random.randint(60,  200))
    elif goods_type == "fmcg_packaged_goods":
        # Cartons, pallets — stackable
        return (random.randint(80,  350), random.randint(60,  200), random.randint(60,  180))
    elif goods_type == "automotive_parts":
        # Engines, axles, body panels
        return (random.randint(100, 500), random.randint(80,  210), random.randint(60,  190))
    elif goods_type == "textiles_apparel":
        # Bales, rolls — light and bulky
        return (random.randint(100, 400), random.randint(80,  210), random.randint(60,  200))
    elif goods_type == "electronics":
        # Packed equipment — compact
        return (random.randint(60,  300), random.randint(50,  180), random.randint(40,  150))
    else:
        # chemicals_industrial: drums, IBC tanks
        return (random.randint(80,  350), random.randint(80,  200), random.randint(80,  180))


def _time_window(hub_name: str):
    weights = HUB_WAVE_WEIGHTS.get(hub_name, HUB_WAVE_WEIGHTS["_default"])
    wave    = random.choices(["morning", "afternoon", "evening"], weights=weights, k=1)[0]
    w_start, w_end = WAVES[wave]

    start = w_start + random.randint(0, int((w_end - w_start) * 0.6))
    end   = min(start + random.randint(120, 240), w_end)
    end   = max(end, start + 120)

    date = "2024-01-15"
    return (
        f"{date}T{start // 60:02d}:{start % 60:02d}:00",
        f"{date}T{end   // 60:02d}:{end   % 60:02d}:00",
    )


def _pick_destination(hub_name):
    lanes = LANES_BY_HUB.get(hub_name)
    if lanes and random.random() < 0.80:
        dests   = [l[0] for l in lanes]
        weights = [l[1] for l in lanes]
        return random.choices(dests, weights=weights, k=1)[0]
    else:
        fallbacks = [
            (28.613, 77.209, "Delhi NCR Mundka"),
            (22.794, 88.328, "Kolkata Dankuni"),   # fixed: was 22.572, 88.363
            (13.067, 80.237, "Chennai"),            # fixed: was 13.082, 80.270
            (26.449, 80.331, "Kanpur"),
            (15.317, 75.135, "Hubli"),
            (16.704, 74.243, "Kolhapur"),
        ]
        return random.choice(fallbacks)


def generate_shipments(n: int = 100, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    hub_pool = [h for h in PICKUP_HUBS for _ in range(h[3])]

    rows = []
    for i in range(n):
        p_hub = random.choice(hub_pool)
        pickup_lat, pickup_lng = _hub_location(p_hub)

        d_hub        = _pick_destination(p_hub[2])
        delivery_lat = round(d_hub[0] + random.uniform(-0.07, 0.07), 4)
        delivery_lng = round(d_hub[1] + random.uniform(-0.07, 0.07), 4)

        goods_type       = _weighted_choice(GOODS_TYPES)
        weight           = _weight(goods_type)
        l, w, h          = _dimensions(goods_type)

        # ── Density sanity guard ────────────────────────────────────────
        # Prevent unrealistically low-density cargo (lighter than foam).
        # Floor: 100 kg/m³ (compressed polystyrene / lightweight FMCG pallets).
        # Reduces the longest dimension in 5 cm steps until density is met.
        _MIN_DENSITY_KG_M3 = 100
        _vol_m3 = (l * w * h) / 1_000_000  # cm³ → m³
        while _vol_m3 > 0 and (weight / _vol_m3) < _MIN_DENSITY_KG_M3:
            # Shrink the largest dimension by 5 cm
            if l >= w and l >= h:
                l = max(30, l - 5)
            elif w >= h:
                w = max(30, w - 5)
            else:
                h = max(30, h - 5)
            _vol_m3 = (l * w * h) / 1_000_000
        earliest, latest = _time_window(p_hub[2])

        rows.append({
            "shipment_id":       f"SH{i+1:03d}",
            "pickup_lat":        pickup_lat,
            "pickup_lng":        pickup_lng,
            "delivery_lat":      delivery_lat,
            "delivery_lng":      delivery_lng,
            "weight_kg":         weight,
            "length_cm":         l,
            "width_cm":          w,
            "height_cm":         h,
            "earliest_delivery": earliest,
            "latest_delivery":   latest,
            "goods_type":        goods_type,
            "pickup_hub":        p_hub[2],
            "delivery_hub":      d_hub[2],
        })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/shipments.csv", index=False)

    print(f"\n{'═'*55}")
    print(f"  LoRRI Shipment Generator  —  {n} shipments")
    print(f"{'═'*55}")
    print(f"  Saved to : data/shipments.csv  |  Seed: {seed}")

    print(f"\n  Weight:")
    print(f"    avg  {df['weight_kg'].mean():,.0f} kg")
    print(f"    min  {df['weight_kg'].min():,.0f} kg")
    print(f"    max  {df['weight_kg'].max():,.0f} kg")
    ltl = (df['weight_kg'] < 8000).sum()
    print(f"    LTL (<8t, consolidation candidates): {ltl} ({ltl*100//n}%)")

    print(f"\n  Top lanes (pickup → delivery):")
    lane_col = df["pickup_hub"].str.split().str[0] + " → " + df["delivery_hub"].str.split().str[0]
    for lane, cnt in lane_col.value_counts().head(8).items():
        print(f"    {lane:<35} {cnt:>3}  {'█' * cnt}")

    print(f"\n  Dispatch waves:")
    hour_col  = df["earliest_delivery"].str[11:13].astype(int)
    morning   = ((hour_col >= 6)  & (hour_col < 12)).sum()
    afternoon = ((hour_col >= 12) & (hour_col < 17)).sum()
    evening   = (hour_col >= 17).sum()
    print(f"    Morning   06–11  {morning:>3} shipments")
    print(f"    Afternoon 12–17  {afternoon:>3} shipments")
    print(f"    Evening   17–21  {evening:>3} shipments")

    print(f"\n  Goods types:")
    for g, c in df["goods_type"].value_counts().items():
        print(f"    {g:<28} {c:>3}")
    print(f"{'═'*55}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate realistic LoRRI inter-city freight data"
    )
    parser.add_argument("--n",    type=int, default=100, help="Number of shipments")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed")
    args = parser.parse_args()
    generate_shipments(n=args.n, seed=args.seed)