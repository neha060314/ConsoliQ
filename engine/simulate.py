"""
simulate.py  —  Scenario simulation engine for ConsoliQ
---------------------------------------------------------
Key fix vs previous version:
  - route_compat params passed DIRECTLY as function args (not global patching)
    because apply_route_filter() already accepts max_bearing_diff + max_detour
  - clustering params still patched as globals (get_valid_groups reads them
    from module scope directly, no function-arg equivalent)
  - MAX_DETOUR_RATIO in clustering = 0.20 (offset: 20% extra km)
    MAX_DETOUR_RATIO in route_compat = 1.25 (multiplier: 1.25x direct)
    These are DIFFERENT scales — handled correctly here
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


PRESET_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id":    "baseline",
        "label": "Baseline (Solo Dispatch)",
        "description": "Every shipment dispatched individually. No consolidation.",
        "color": "#ef4444",
        "params": {
            "pickup_radius_km":    1.0,
            "delivery_cluster_km": 1.0,
            "min_overlap_minutes": 480,
            "clustering_detour":   0.02,
            "route_detour":        1.02,
            "max_bearing_diff":    10.0,
        },
    },
    {
        "id":    "conservative",
        "label": "Conservative",
        "description": "Tight windows, same-lane only. Moderate savings, low risk.",
        "color": "#f97316",
        "params": {
            "pickup_radius_km":    20.0,
            "delivery_cluster_km": 15.0,
            "min_overlap_minutes": 120,
            "clustering_detour":   0.10,
            "route_detour":        1.10,
            "max_bearing_diff":    30.0,
        },
    },
    {
        "id":    "balanced",
        "label": "Balanced (Recommended)",
        "description": "Default ConsoliQ settings. Maximises consolidation with route logic.",
        "color": "#22c55e",
        "params": {
            "pickup_radius_km":    50.0,
            "delivery_cluster_km": 50.0,
            "min_overlap_minutes": 30,
            "clustering_detour":   0.20,
            "route_detour":        1.25,
            "max_bearing_diff":    45.0,
        },
    },
    {
        "id":    "aggressive",
        "label": "Aggressive",
        "description": "Wider zones, larger detour tolerance. Maximum trips saved.",
        "color": "#3b82f6",
        "params": {
            "pickup_radius_km":    80.0,
            "delivery_cluster_km": 80.0,
            "min_overlap_minutes": 15,
            "clustering_detour":   0.35,
            "route_detour":        1.40,
            "max_bearing_diff":    60.0,
        },
    },
    {
        "id":    "ultra",
        "label": "Ultra Aggressive",
        "description": "Maximum consolidation. Best for bulk/non-urgent freight.",
        "color": "#8b5cf6",
        "params": {
            "pickup_radius_km":    120.0,
            "delivery_cluster_km": 100.0,
            "min_overlap_minutes": 0,
            "clustering_detour":   0.50,
            "route_detour":        1.60,
            "max_bearing_diff":    75.0,
        },
    },
]


def run_scenario(
    shipments: List[dict],
    params:    Dict[str, Any],
    label:     str = "Custom",
) -> Dict[str, Any]:
    """
    Run the full ConsoliQ pipeline with custom parameters.

    clustering params are patched as module globals (only mechanism available).
    route_compat params are passed directly as function arguments.
    """
    import engine.clustering   as clustering_mod
    from engine.clustering   import get_valid_groups
    from engine.route_compat import apply_route_filter
    from engine.bin_packing  import (
        pack_groups, avg_load_factor, avg_weight_utilization,
        avg_volume_utilization, avg_spatial_utilization,
        total_cost_inr, total_co2_kg, vehicle_mix, savings_vs_solo,
    )

    t0 = time.time()

    # Deep-copy shipments so _route_distance_km and other mutations set
    # by pack_groups/route_compat don't bleed between scenario runs.
    import copy as _copy
    shipments = [_copy.copy(s) for s in shipments]

    # Save originals
    orig_pickup   = clustering_mod.PICKUP_RADIUS_KM
    orig_delivery = clustering_mod.DELIVERY_CLUSTER_KM
    orig_overlap  = clustering_mod.MIN_OVERLAP_SECONDS
    orig_detour   = clustering_mod.MAX_DETOUR_RATIO

    try:
        # Patch clustering module globals
        clustering_mod.PICKUP_RADIUS_KM    = float(params.get("pickup_radius_km",    50.0))
        clustering_mod.DELIVERY_CLUSTER_KM = float(params.get("delivery_cluster_km", 50.0))
        clustering_mod.MIN_OVERLAP_SECONDS = int(params.get("min_overlap_minutes",   30)) * 60
        clustering_mod.MAX_DETOUR_RATIO    = float(params.get("clustering_detour",   0.20))

        # Step 1 — cluster with patched globals
        groups = get_valid_groups(shipments)

        # Step 2 — route filter: pass args DIRECTLY (not globals)
        # route_detour is a multiplier (1.25 = 25% extra km allowed)
        routed = apply_route_filter(
            groups,
            max_bearing_diff = float(params.get("max_bearing_diff", 45.0)),
            max_detour       = float(params.get("route_detour",      1.25)),
        )

        # Step 3 — pack
        packed  = pack_groups(routed)
        savings = savings_vs_solo(packed)

    finally:
        # Always restore originals even if pipeline throws
        clustering_mod.PICKUP_RADIUS_KM    = orig_pickup
        clustering_mod.DELIVERY_CLUSTER_KM = orig_delivery
        clustering_mod.MIN_OVERLAP_SECONDS = orig_overlap
        clustering_mod.MAX_DETOUR_RATIO    = orig_detour

    elapsed   = round(time.time() - t0, 2)
    n_ships   = len(shipments)
    n_trucks  = len(packed)
    singleton = sum(1 for g in packed if len(g) == 1)
    batched   = n_trucks - singleton
    batched_s = sum(len(g) for g in packed if len(g) > 1)

    return {
        "label":               label,
        "params":              params,
        "elapsed_s":           elapsed,
        "n_shipments":         n_ships,
        "n_trucks":            n_trucks,
        "n_singleton":         singleton,
        "n_batched_groups":    batched,
        "n_batched_shipments": batched_s,
        "trips_saved":         savings.get("trips_saved", 0),
        "trip_reduction_pct":  round((1 - n_trucks / n_ships) * 100, 1) if n_ships else 0,
        "avg_load_factor":     round(avg_load_factor(packed)         * 100, 1),
        "avg_weight_util":     round(avg_weight_utilization(packed)  * 100, 1),
        "avg_volume_util":     round(avg_volume_utilization(packed)  * 100, 1),
        "avg_spatial_util":    round(avg_spatial_utilization(packed) * 100, 1),
        "total_cost_inr":      total_cost_inr(packed),
        "cost_saved_inr":      savings.get("cost_saved_inr", 0),
        "cost_saving_pct":     savings.get("cost_saving_pct", 0),
        "total_co2_kg":        total_co2_kg(packed),
        "co2_saved_kg":        savings.get("co2_saved_kg", 0),
        "co2_saving_pct":      savings.get("co2_saving_pct", 0),
        "vehicle_mix":         vehicle_mix(packed),
        "_packed_groups":      packed,
    }


def compare_scenarios(
    shipments: List[dict],
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if scenarios is None:
        scenarios = PRESET_SCENARIOS

    results = []
    for sc in scenarios:
        logger.info(f"simulate: running '{sc['label']}'...")
        try:
            r = run_scenario(shipments, sc["params"], label=sc["label"])
            r["color"] = sc.get("color", "#6b7280")
            r["id"]    = sc.get("id", sc["label"])
            results.append(r)
        except Exception as e:
            logger.error(f"Scenario '{sc['label']}' failed: {e}")
            results.append({
                "label": sc["label"], "id": sc.get("id", sc["label"]),
                "color": sc.get("color", "#6b7280"), "error": str(e),
                "trip_reduction_pct": 0, "avg_load_factor": 0,
                "cost_saving_pct": 0, "co2_saving_pct": 0,
            })

    valid        = [r for r in results if "error" not in r]
    non_baseline = [r for r in valid   if r.get("id") != "baseline"]

    def composite(r):
        return (r.get("trip_reduction_pct", 0) * 0.35
                + r.get("avg_load_factor",   0) * 0.35
                + r.get("cost_saving_pct",   0) * 0.30)

    recommended = max(non_baseline, key=composite) if non_baseline else (valid[0] if valid else {})
    return {"scenarios": results, "recommended": recommended}


def build_custom_scenario(
    pickup_radius_km:    float = 50.0,
    delivery_cluster_km: float = 50.0,
    min_overlap_minutes: int   = 30,
    max_detour_ratio:    float = 1.25,
    max_bearing_diff:    float = 45.0,
    label:               str   = "Custom",
) -> Dict[str, Any]:
    """
    Build a custom scenario dict from user-facing slider values.
    max_detour_ratio is the route_compat multiplier (1.25 = 25% extra km).
    clustering_detour is derived as multiplier - 1.0 (offset form).
    """
    return {
        "label": label, "id": "custom", "color": "#06b6d4",
        "params": {
            "pickup_radius_km":    pickup_radius_km,
            "delivery_cluster_km": delivery_cluster_km,
            "min_overlap_minutes": min_overlap_minutes,
            "clustering_detour":   round(max(max_detour_ratio - 1.0, 0.01), 3),
            "route_detour":        max_detour_ratio,
            "max_bearing_diff":    max_bearing_diff,
        },
    }


def scenarios_to_dataframe(results: List[Dict[str, Any]]):
    import pandas as pd
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"Scenario": r["label"], "Error": r["error"]})
            continue
        rows.append({
            "Scenario":         r["label"],
            "Trucks":           r.get("n_trucks",           0),
            "Trip Reduction %": r.get("trip_reduction_pct", 0),
            "Load Factor %":    r.get("avg_load_factor",    0),
            "Weight Util %":    r.get("avg_weight_util",    0),
            "Spatial Util %":   r.get("avg_spatial_util",   0),
            "Cost Saved ₹":     r.get("cost_saved_inr",     0),
            "Cost Saving %":    r.get("cost_saving_pct",    0),
            "CO₂ Saved kg":     r.get("co2_saved_kg",       0),
            "CO₂ Saving %":     r.get("co2_saving_pct",     0),
            "Batched Groups":   r.get("n_batched_groups",   0),
            "Singletons":       r.get("n_singleton",        0),
        })
    return pd.DataFrame(rows)