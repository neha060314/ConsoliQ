"""
fleet.py  —  LoRRI truck fleet configuration (COMPLETE for bin_packing.py)
---------------------------------------------
Based on LoRRI's 80+ truck types on their platform.
For inter-city industrial freight (routes >150km).

ALL REQUIRED FIELDS for bin_packing.py:
✓ name                    — vehicle model  
✓ vehicle_no              — fleet registration numbers (VehiclePool)
✓ max_kg                  — weight capacity
✓ l, w, h                 — internal dimensions (cm)
✓ fixed_cost_per_trip     — INR base cost
✓ cost_per_km             — INR/km variable cost  
✓ co2_per_km_kg           — baseline emissions
✓ co2_per_ton_km          — loaded efficiency
✓ truck_class             — LCV/ICV/HCV grouping
✓ volume_m3               — derived (used in metrics)
✓ axles                   — axle balance reference
✓ fuel_efficiency_kmpl    — carbon calcs
"""

FLEET = [
    # ── LCV (Light Commercial Vehicles) ──────────────────────────────────────
    {
        "name":                 "Tata Ace (Mini Truck)",
        "vehicle_no":           "MH-04-AB-1001",
        "max_kg":               750,
        "cost_per_km":          18,
        "l":                    180,
        "w":                    140, 
        "h":                    120,
        "volume_m3":            3.02,
        "axles":                2,
        "truck_class":          "LCV",
        "fixed_cost_per_trip":  500,
        "fuel_efficiency_kmpl": 18.0,
        "co2_per_km_kg":        0.18,
        "co2_per_ton_km":       0.062,
    },
    {
        "name":                 "Tata 407 (LCV)",
        "vehicle_no":           "MH-04-AB-1002", 
        "max_kg":               2000,
        "cost_per_km":          28,
        "l":                    290,
        "w":                    175,
        "h":                    150,
        "volume_m3":            7.61,
        "axles":                2,
        "truck_class":          "LCV", 
        "fixed_cost_per_trip":  700,
        "fuel_efficiency_kmpl": 14.0,
        "co2_per_km_kg":        0.22,
        "co2_per_ton_km":       0.058,
    },
    {
        "name":                 "Ashok Leyland Dost",
        "vehicle_no":           "MH-04-AB-1003",
        "max_kg":               1500,
        "cost_per_km":          25,
        "l":                    250,
        "w":                    160,
        "h":                    140,
        "volume_m3":            5.60,
        "axles":                2,
        "truck_class":          "LCV",
        "fixed_cost_per_trip":  650,
        "fuel_efficiency_kmpl": 15.5,
        "co2_per_km_kg":        0.20,
        "co2_per_ton_km":       0.060,
    },

    # ── ICV (Intermediate Commercial Vehicles) ───────────────────────────────
    {
        "name":                 "Eicher 20ft",
        "vehicle_no":           "MH-04-AB-1004",
        "max_kg":               7500,
        "cost_per_km":          45,
        "l":                    600,
        "w":                    230,
        "h":                    220,
        "volume_m3":            30.36,
        "axles":                3,
        "truck_class":          "ICV",
        "fixed_cost_per_trip":  1200,
        "fuel_efficiency_kmpl": 9.0,
        "co2_per_km_kg":        0.45,
        "co2_per_ton_km":       0.048,
    },
    {
        "name":                 "Mahindra Bolero Pickup",
        "vehicle_no":           "MH-04-AB-1005",
        "max_kg":               1200,
        "cost_per_km":          22,
        "l":                    320,
        "w":                    180,
        "h":                    160,
        "volume_m3":            9.22,
        "axles":                2,
        "truck_class":          "ICV", 
        "fixed_cost_per_trip":  800,
        "fuel_efficiency_kmpl": 13.0,
        "co2_per_km_kg":        0.25,
        "co2_per_ton_km":       0.055,
    },
    {
        "name":                 "Eicher 19ft",
        "vehicle_no":           "MH-04-AB-1006",
        "max_kg":               9000,
        "cost_per_km":          48,
        "l":                    580,
        "w":                    210,
        "h":                    210,
        "volume_m3":            25.60,
        "axles":                3,
        "truck_class":          "ICV",
        "fixed_cost_per_trip":  1300,
        "fuel_efficiency_kmpl": 8.5,
        "co2_per_km_kg":        0.48,
        "co2_per_ton_km":       0.046,
    },

    # ── HCV (Heavy Commercial Vehicles) ─────────────────────────────────────
    {
        "name":                 "Ashok Leyland 32ft",
        "vehicle_no":           "MH-04-AB-1234",
        "max_kg":               15000,
        "cost_per_km":          65,
        "l":                    975,
        "w":                    245,
        "h":                    240,
        "volume_m3":            57.29,
        "axles":                4,
        "truck_class":          "HCV",
        "fixed_cost_per_trip":  1800,
        "fuel_efficiency_kmpl": 6.5,
        "co2_per_km_kg":        0.68,
        "co2_per_ton_km":       0.038,
    },
    {
        "name":                 "Volvo FH 40ft (FTL)",
        "vehicle_no":           "MH-04-AB-5678",
        "max_kg":               22000,
        "cost_per_km":          95,
        "l":                    1200,
        "w":                    245,
        "h":                    270,
        "volume_m3":            79.38,
        "axles":                5,
        "truck_class":          "HCV",
        "fixed_cost_per_trip":  2500,
        "fuel_efficiency_kmpl": 4.5,
        "co2_per_km_kg":        0.95,
        "co2_per_ton_km":       0.032,
    },
    {
        "name":                 "BharatBenz 32ft MXL", 
        "vehicle_no":           "MH-04-AB-9999",
        "max_kg":               16000,
        "cost_per_km":          68,
        "l":                    980,
        "w":                    250,
        "h":                    245,
        "volume_m3":            60.00,
        "axles":                4,
        "truck_class":          "HCV",
        "fixed_cost_per_trip":  1900,
        "fuel_efficiency_kmpl": 6.2,
        "co2_per_km_kg":        0.70,
        "co2_per_ton_km":       0.036,
    },

    # ── CONTAINERS (SXL/MXL variants) ───────────────────────────────────────
    {
        "name":                 "32ft Single Axle (SXL)",
        "vehicle_no":           "MH-04-AB-2001",
        "max_kg":               9500,
        "cost_per_km":          52,
        "l":                    975,
        "w":                    240,
        "h":                    235,
        "volume_m3":            54.93,
        "axles":                3,
        "truck_class":          "HCV",
        "fixed_cost_per_trip":  1500,
        "fuel_efficiency_kmpl": 7.8,
        "co2_per_km_kg":        0.55,
        "co2_per_ton_km":       0.042,
    },
    {
        "name":                 "32ft Multi Axle (MXL)", 
        "vehicle_no":           "MH-04-AB-2002",
        "max_kg":               16000,
        "cost_per_km":          70,
        "l":                    975,
        "w":                    245,
        "h":                    240,
        "volume_m3":            57.29,
        "axles":                4,
        "truck_class":          "HCV",
        "fixed_cost_per_trip":  2000,
        "fuel_efficiency_kmpl": 6.0,
        "co2_per_km_kg":        0.72,
        "co2_per_ton_km":       0.035,
    },
    {
        "name":                 "24ft Multi Axle",
        "vehicle_no":           "MH-04-AB-2003",
        "max_kg":               12000,
        "cost_per_km":          58,
        "l":                    730,
        "w":                    240,
        "h":                    235,
        "volume_m3":            41.13,
        "axles":                3,
        "truck_class":          "ICV",
        "fixed_cost_per_trip":  1600,
        "fuel_efficiency_kmpl": 7.2,
        "co2_per_km_kg":        0.60,
        "co2_per_ton_km":       0.040,
    },
]

# ── VALIDATION ────────────────────────────────────────────────────────────
# All required fields present ✓
REQUIRED_FIELDS = {"name", "vehicle_no", "max_kg", "l", "w", "h", 
                  "fixed_cost_per_trip", "cost_per_km", "co2_per_km_kg", 
                  "co2_per_ton_km", "truck_class"}

for i, truck in enumerate(FLEET):
    missing = REQUIRED_FIELDS - set(truck.keys())
    if missing:
        raise ValueError(f"FLEET[{i}]: missing fields {missing}")
    
print(f"✅ FLEET loaded: {len(FLEET)} vehicles across {len(set(t['truck_class'] for t in FLEET))} classes")