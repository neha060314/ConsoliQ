"""
metrics.py  —  Consolidated metrics and analytics for ConsoliQ
--------------------------------------------------------------
Single place for all KPI computation used by dashboard + reports.

Replaces scattered metric helpers across bin_packing.py.
Provides both per-run metrics and cross-run trend analysis.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
#  CORE UTILIZATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def avg_load_factor(groups: List[List[dict]]) -> float:
    vals = [g[0].get("_load_factor", 0.0) for g in groups if g]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def avg_weight_utilization(groups: List[List[dict]]) -> float:
    vals = [g[0].get("_weight_utilization", 0.0) for g in groups if g]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def avg_volume_utilization(groups: List[List[dict]]) -> float:
    vals = [g[0].get("_volume_utilization", 0.0) for g in groups if g]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def avg_spatial_utilization(groups: List[List[dict]]) -> float:
    vals = [g[0].get("_spatial_utilization", 0.0) for g in groups if g]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def total_cost_inr(groups: List[List[dict]]) -> float:
    return round(sum(g[0].get("_cost_estimate_inr", 0.0) for g in groups if g), 0)


def total_co2_kg(groups: List[List[dict]]) -> float:
    return round(sum(g[0].get("_co2_kg_estimate", 0.0) for g in groups if g), 2)


def vehicle_mix(groups: List[List[dict]]) -> Dict[str, int]:
    mix: Dict[str, int] = {}
    for g in groups:
        if g:
            name = g[0].get("_assigned_vehicle", "unknown")
            mix[name] = mix.get(name, 0) + 1
    return dict(sorted(mix.items(), key=lambda x: -x[1]))


# ═══════════════════════════════════════════════════════════════════════
#  TRIP + CONSOLIDATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def consolidation_summary(
    shipments: List[dict],
    groups:    List[List[dict]],
) -> Dict:
    """High-level consolidation summary for the top metrics row."""
    n_ships   = len(shipments)
    n_trucks  = len(groups)
    singletons = sum(1 for g in groups if len(g) == 1)
    batched   = n_trucks - singletons
    batched_s = sum(len(g) for g in groups if len(g) > 1)

    trip_reduction = round((1 - n_trucks / n_ships) * 100, 1) if n_ships else 0.0

    exact_lane = sum(
        1 for g in groups
        if len(g) > 1 and not any(s.get("_multi_drop") for s in g)
    )
    corridor = sum(
        1 for g in groups
        if any(s.get("_multi_drop") for s in g)
    )

    return {
        "n_shipments":         n_ships,
        "n_trucks":            n_trucks,
        "n_singletons":        singletons,
        "n_batched_groups":    batched,
        "n_batched_shipments": batched_s,
        "exact_lane_groups":   exact_lane,
        "corridor_groups":     corridor,
        "trip_reduction_pct":  trip_reduction,
        "trips_saved":         n_ships - n_trucks,
    }


# ═══════════════════════════════════════════════════════════════════════
#  FINANCIAL + CARBON SAVINGS vs SOLO BASELINE
# ═══════════════════════════════════════════════════════════════════════

def pipeline_savings(
    packed_groups:  List[List[dict]],
    solo_cost_inr:  float,
    solo_co2_kg:    float,
    solo_trips:     int,
) -> Dict:
    """
    Compute savings vs a pre-computed solo baseline.
    Use this when you already have solo figures from savings_vs_solo().
    """
    cons_cost  = total_cost_inr(packed_groups)
    cons_co2   = total_co2_kg(packed_groups)
    cons_trips = len(packed_groups)

    cost_saved = solo_cost_inr - cons_cost
    co2_saved  = solo_co2_kg   - cons_co2

    return {
        "solo_trips":         solo_trips,
        "consolidated_trips": cons_trips,
        "trips_saved":        solo_trips - cons_trips,
        "trip_reduction_pct": round((solo_trips - cons_trips) / solo_trips * 100, 1) if solo_trips else 0,

        "solo_cost_inr":      round(solo_cost_inr, 0),
        "cons_cost_inr":      round(cons_cost, 0),
        "cost_saved_inr":     round(cost_saved, 0),
        "cost_saving_pct":    round(cost_saved / solo_cost_inr * 100, 1) if solo_cost_inr > 0 else 0,

        "solo_co2_kg":        round(solo_co2_kg, 1),
        "cons_co2_kg":        round(cons_co2, 1),
        "co2_saved_kg":       round(co2_saved, 1),
        "co2_saving_pct":     round(co2_saved / solo_co2_kg * 100, 1) if solo_co2_kg > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  LANE-LEVEL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

def lane_efficiency(groups: List[List[dict]]) -> List[Dict]:
    """
    Per-lane breakdown: consolidation rate, avg load factor, total weight.
    Used by the dashboard's lane heatmap.
    """
    lane_data: Dict[str, dict] = defaultdict(lambda: {
        "shipments": 0, "trucks": 0, "total_weight_kg": 0,
        "load_factors": [], "cost_inr": 0, "co2_kg": 0,
    })

    for group in groups:
        lane = group[0].get("_group_lane") or group[0].get("_lane", "Unknown")
        d    = lane_data[lane]
        d["shipments"]    += len(group)
        d["trucks"]       += 1
        d["total_weight_kg"] += group[0].get("_group_weight_kg", 0)
        d["load_factors"].append(group[0].get("_load_factor", 0))
        d["cost_inr"]     += group[0].get("_cost_estimate_inr", 0)
        d["co2_kg"]       += group[0].get("_co2_kg_estimate", 0)

    result = []
    for lane, d in sorted(lane_data.items(), key=lambda x: -x[1]["shipments"]):
        lfs = d["load_factors"]
        result.append({
            "lane":           lane,
            "shipments":      d["shipments"],
            "trucks":         d["trucks"],
            "consolidation_rate": round((1 - d["trucks"] / d["shipments"]) * 100, 1)
                                  if d["shipments"] > 0 else 0,
            "avg_load_factor": round(sum(lfs) / len(lfs) * 100, 1) if lfs else 0,
            "total_weight_kg": round(d["total_weight_kg"], 0),
            "cost_inr":       round(d["cost_inr"], 0),
            "co2_kg":         round(d["co2_kg"], 1),
        })
    return result


# ═══════════════════════════════════════════════════════════════════════
#  UTILIZATION DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════

def utilization_distribution(groups: List[List[dict]]) -> Dict[str, int]:
    """
    Histogram of load factors: <40%, 40-60%, 60-80%, 80%+
    Used for the utilization breakdown chart.
    """
    buckets = {"<40%": 0, "40-60%": 0, "60-80%": 0, "80%+": 0}
    for g in groups:
        lf = g[0].get("_load_factor", 0) * 100
        if lf < 40:
            buckets["<40%"] += 1
        elif lf < 60:
            buckets["40-60%"] += 1
        elif lf < 80:
            buckets["60-80%"] += 1
        else:
            buckets["80%+"] += 1
    return buckets


# ═══════════════════════════════════════════════════════════════════════
#  GOODS TYPE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def goods_type_summary(groups: List[List[dict]]) -> List[Dict]:
    """Per goods-type count and avg weight. Used by composition chart."""
    goods_data: Dict[str, dict] = defaultdict(lambda: {"count": 0, "total_kg": 0})
    for group in groups:
        for s in group:
            gt = s.get("goods_type", "unknown")
            goods_data[gt]["count"]    += 1
            goods_data[gt]["total_kg"] += s.get("weight_kg", 0)

    return [
        {
            "goods_type": gt,
            "shipments":  d["count"],
            "total_kg":   round(d["total_kg"], 0),
            "avg_kg":     round(d["total_kg"] / d["count"], 0) if d["count"] else 0,
        }
        for gt, d in sorted(goods_data.items(), key=lambda x: -x[1]["count"])
    ]


# ═══════════════════════════════════════════════════════════════════════
#  FULL RUN REPORT  (used by main.py and dashboard summary)
# ═══════════════════════════════════════════════════════════════════════

def full_report(
    shipments:     List[dict],
    clustered:     List[List[dict]],
    routed:        List[List[dict]],
    packed:        List[List[dict]],
    savings:       Dict,
) -> Dict:
    """
    Comprehensive report dict for the full pipeline run.
    Pass output of savings_vs_solo() as `savings`.
    """
    return {
        "pipeline": {
            "input_shipments":  len(shipments),
            "after_clustering": len(clustered),
            "after_routing":    len(routed),
            "trucks_dispatched": len(packed),
        },
        "consolidation":  consolidation_summary(shipments, packed),
        "utilization": {
            "avg_load_factor":    round(avg_load_factor(packed) * 100, 1),
            "avg_weight_util":    round(avg_weight_utilization(packed) * 100, 1),
            "avg_volume_util":    round(avg_volume_utilization(packed) * 100, 1),
            "avg_spatial_util":   round(avg_spatial_utilization(packed) * 100, 1),
            "distribution":       utilization_distribution(packed),
        },
        "savings": pipeline_savings(
            packed,
            solo_cost_inr=savings.get("solo_cost_inr",
                savings.get("cost_saved_inr", 0) + total_cost_inr(packed)),
            solo_co2_kg=savings.get("solo_co2_kg",
                savings.get("co2_saved_kg", 0) + total_co2_kg(packed)),
            solo_trips=savings.get("solo_trips", len(shipments)),
        ),
        "fleet":          vehicle_mix(packed),
        "lanes":          lane_efficiency(packed),
        "goods":          goods_type_summary(packed),
    }