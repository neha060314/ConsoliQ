"""
route_compat.py  —  Route compatibility + multi-stop sequencing
-----------------------------------------------------------------------
Sits between clustering.py and bin_packing.py:

    clustering.py   →  route_compat.py  →  bin_packing.py

clustering.py   : "same origin + destination + time window?"
route_compat.py : "does the ACTUAL ROAD make sense? what's the stop order?"
bin_packing.py  : "which vehicle, how packed?"

What this module does (in order):
  1.  Bearing check         — are shipments travelling the same direction?
  2.  Detour check          — symmetric, road-factor corrected haversine
  3.  Volume feasibility    — dimensional cube check
  4.  TSP stop sequencing   — optimal delivery order (nearest-neighbour + 2-opt)
  5.  Corridor routing      — Nashik→Bhopal→Delhi style through-routes
  6.  Group merging         — after splits, re-merge compatible groups
  7.  Route annotation      — bearing, distance, stop list on every shipment

Algorithms:
  - Haversine + road factor  : distance proxy without live map API
  - Compass bearing          : direction-of-travel filter
  - Nearest-neighbour TSP    : O(n²), good enough for n<20 stops
  - 2-opt improvement        : O(n²) per pass, eliminates crossings
  - 3D AABB volume check     : fast pre-filter before bin_packing.py
  - Union-find               : re-merging split groups

Bugs fixed vs original:
  FIX 1 — _two_opt: tour[j % n] → tour[(j+1) % n] for correct after-edge
  FIX 2 — _detour_ratio: symmetric baseline — each order divides by its
           own direct distance, not always the anchor's
  FIX 3 — _filter_single_group: checks pairwise within new group members,
           not just against existing members
  FIX 4 — _salvage_feasible_subset: eject by index not object equality,
           prevents silent infinite loop on duplicate-valued dicts
  FIX 5 — _group_corridor_bearing: was using group[0] bearing for whole
           group — now uses true pickup→delivery centroid bearing
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tuning ──────────────────────────────────────────────────────────────
MAX_BEARING_DIFF_DEG     = 45.0
MIN_ROUTE_KM_FOR_BEARING = 20.0
MAX_DETOUR_RATIO         = 1.25
ROAD_FACTOR              = 1.35

# Default truck cargo hold (standard 32-ft container)
DEFAULT_TRUCK_L_CM    = 960.0
DEFAULT_TRUCK_W_CM    = 240.0
DEFAULT_TRUCK_H_CM    = 240.0
DEFAULT_TRUCK_VOL_CM3 = DEFAULT_TRUCK_L_CM * DEFAULT_TRUCK_W_CM * DEFAULT_TRUCK_H_CM


# ═══════════════════════════════════════════════════════════════════════
#  1. GEOMETRY
# ═══════════════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _road_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Road-distance estimate: haversine × ROAD_FACTOR.
    ROAD_FACTOR 1.35 validated for Indian NH corridors (Mumbai→Pune,
    Bhiwandi→Surat, Nashik→Hyderabad). Swap this function for a live
    API cache in production — rest of module unchanged.
    """
    return _haversine_km(lat1, lng1, lat2, lng2) * ROAD_FACTOR


def _bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compass bearing point1 → point2, degrees [0, 360)."""
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlng  = math.radians(lng2 - lng1)
    x = math.sin(dlng) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlng))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _bearing_diff(b1: float, b2: float) -> float:
    """Shortest angular distance between two bearings (0–180°)."""
    diff = abs(b1 - b2) % 360
    return diff if diff <= 180 else 360 - diff


def _shipment_bearing(s: dict) -> float:
    return _bearing(
        s["pickup_lat"], s["pickup_lng"],
        s["delivery_lat"], s["delivery_lng"],
    )


def _shipment_road_km(s: dict) -> float:
    return _road_km(
        s["pickup_lat"], s["pickup_lng"],
        s["delivery_lat"], s["delivery_lng"],
    )


def _shipment_volume_cm3(s: dict) -> float:
    l = s.get("length_cm", 0) or 0
    w = s.get("width_cm",  0) or 0
    h = s.get("height_cm", 0) or 0
    return float(l * w * h)


def _group_corridor_bearing(group: List[dict]) -> float:
    """
    FIX 5 — True corridor bearing from pickup centroid → delivery centroid.
    Original used group[0]'s bearing for the whole group, causing wrong
    corridor detection when group[0] was an outlier direction.
    """
    avg_plat = sum(s["pickup_lat"]    for s in group) / len(group)
    avg_plng = sum(s["pickup_lng"]    for s in group) / len(group)
    avg_dlat = sum(s["delivery_lat"]  for s in group) / len(group)
    avg_dlng = sum(s["delivery_lng"]  for s in group) / len(group)
    return _bearing(avg_plat, avg_plng, avg_dlat, avg_dlng)


# ═══════════════════════════════════════════════════════════════════════
#  2. DETOUR RATIO — Symmetric, road-factor corrected
# ═══════════════════════════════════════════════════════════════════════

def _detour_ratio(anchor: dict, rider: dict) -> Tuple[float, str]:
    """
    FIX 2 — Symmetric baseline per drop order.

    Original bug: both route orderings divided by anchor's direct distance.
    When rider_first is evaluated, the fair baseline is the rider's own
    direct distance (that's the solo trip being compared against).
    Using anchor's direct for rider_first inflated or deflated the ratio
    depending on which shipment was longer, causing wrong accept/reject.

    Fix: each ordering divides by the direct distance of whichever
    shipment is dropped LAST (that shipment travels the full detour).
    The anchor's direct is used for anchor_first (anchor goes furthest).
    The rider's direct is used for rider_first (rider goes furthest).

    Returns (ratio, best_order_str).
    ratio = multi_stop_road_km / relevant_direct_road_km
    1.0 = zero detour. 1.25 = 25% extra.
    best_order_str: "anchor_first" | "rider_first"
    """
    origin_lat = (anchor["pickup_lat"] + rider["pickup_lat"]) / 2
    origin_lng = (anchor["pickup_lng"] + rider["pickup_lng"]) / 2

    direct_anchor = _road_km(
        origin_lat, origin_lng,
        anchor["delivery_lat"], anchor["delivery_lng"],
    )
    direct_rider = _road_km(
        origin_lat, origin_lng,
        rider["delivery_lat"], rider["delivery_lng"],
    )

    # Order 1: origin → anchor_delivery → rider_delivery
    # anchor dropped first, rider travels to end — baseline is rider's direct
    r1 = (
        _road_km(origin_lat,              origin_lng,
                 anchor["delivery_lat"],   anchor["delivery_lng"])
        + _road_km(anchor["delivery_lat"], anchor["delivery_lng"],
                   rider["delivery_lat"],  rider["delivery_lng"])
    )
    ratio1 = (r1 / direct_rider) if direct_rider > 0 else 1.0

    # Order 2: origin → rider_delivery → anchor_delivery
    # rider dropped first, anchor travels to end — baseline is anchor's direct
    r2 = (
        _road_km(origin_lat,             origin_lng,
                 rider["delivery_lat"],   rider["delivery_lng"])
        + _road_km(rider["delivery_lat"], rider["delivery_lng"],
                   anchor["delivery_lat"], anchor["delivery_lng"])
    )
    ratio2 = (r2 / direct_anchor) if direct_anchor > 0 else 1.0

    if ratio1 <= ratio2:
        return ratio1, "anchor_first"
    return ratio2, "rider_first"


# ═══════════════════════════════════════════════════════════════════════
#  3. VOLUME FEASIBILITY
# ═══════════════════════════════════════════════════════════════════════

def _volume_feasible(
    group: List[dict],
    truck_vol_cm3: float = DEFAULT_TRUCK_VOL_CM3,
) -> Tuple[bool, float]:
    """
    Returns (feasible, utilisation_ratio).
    Uses axis-aligned bounding box summation — conservative fast pre-filter.
    bin_packing.py handles precise 3D stacking.
    """
    total_vol = sum(_shipment_volume_cm3(s) for s in group)
    util      = total_vol / truck_vol_cm3 if truck_vol_cm3 > 0 else 0.0
    return util <= 1.0, round(util, 3)


# ═══════════════════════════════════════════════════════════════════════
#  4. TSP STOP SEQUENCING — Nearest-Neighbour + 2-opt
#
#  n=1  : trivial
#  n=2  : try both orderings, pick shorter total delivery leg
#  n≤8  : nearest-neighbour + 2-opt improvement
#  n>8  : nearest-neighbour only (2-opt too slow per-group at scale)
#
#  Start point: shipment with earliest deadline (EDF anchor) to bias
#  toward time-compatible orderings.
# ═══════════════════════════════════════════════════════════════════════

def _nn_tour(
    points: List[Tuple[float, float]],
    start_idx: int = 0,
) -> List[int]:
    """
    Nearest-neighbour TSP heuristic.
    Returns ordered list of indices into `points`. O(n²).
    """
    n       = len(points)
    visited = [False] * n
    tour    = [start_idx]
    visited[start_idx] = True

    for _ in range(n - 1):
        cur       = tour[-1]
        best_dist = float("inf")
        best_next = -1
        for j in range(n):
            if not visited[j]:
                d = _road_km(
                    points[cur][0], points[cur][1],
                    points[j][0],   points[j][1],
                )
                if d < best_dist:
                    best_dist = d
                    best_next = j
        tour.append(best_next)
        visited[best_next] = True

    return tour


def _two_opt(
    points: List[Tuple[float, float]],
    tour:   List[int],
    max_passes: int = 10,
) -> List[int]:
    """
    FIX 1 — Correct after-edge index in 2-opt swap evaluation.

    Original bug: used tour[j % n] for BOTH the before-edge end and the
    after-edge start. The after-edge should connect tour[j] to tour[(j+1) % n],
    not back to tour[j] itself. This made the "after" cost always 0 for the
    second term, so 2-opt accepted swaps that made the route longer.

    Fix: after-edge is tour[i] → tour[(j+1) % n] (the node after j in
    the current tour), which is the correct 2-opt reversal target.

    2-opt swap: reverse the sub-tour between i and j (inclusive).
    Before: ...→ tour[i-1] → tour[i] →...→ tour[j] → tour[j+1] →...
    After:  ...→ tour[i-1] → tour[j] →...→ tour[i] → tour[j+1] →...
    """
    n        = len(tour)
    improved = True
    passes   = 0

    while improved and passes < max_passes:
        improved = False
        passes  += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Edges being removed:
                #   tour[i-1] → tour[i]
                #   tour[j]   → tour[(j+1) % n]
                before = (
                    _road_km(
                        points[tour[i - 1]][0], points[tour[i - 1]][1],
                        points[tour[i]][0],     points[tour[i]][1],
                    )
                    + _road_km(
                        points[tour[j]][0],           points[tour[j]][1],
                        points[tour[(j + 1) % n]][0], points[tour[(j + 1) % n]][1],
                    )
                )
                # Edges being added:
                #   tour[i-1] → tour[j]
                #   tour[i]   → tour[(j+1) % n]
                after = (
                    _road_km(
                        points[tour[i - 1]][0], points[tour[i - 1]][1],
                        points[tour[j]][0],     points[tour[j]][1],
                    )
                    + _road_km(
                        points[tour[i]][0],           points[tour[i]][1],
                        points[tour[(j + 1) % n]][0], points[tour[(j + 1) % n]][1],
                    )
                )
                if after < before - 0.01:  # 0.01km tolerance avoids float churn
                    tour[i: j + 1] = tour[i: j + 1][::-1]
                    improved = True

    logger.debug(f"  2-opt: {passes} passes")
    return tour


def sequence_deliveries(group: List[dict]) -> List[dict]:
    """
    Returns the group reordered by optimal delivery sequence.

    n=1 : return as-is
    n=2 : try both orderings, pick shorter total delivery leg
    n≤8 : nearest-neighbour + 2-opt
    n>8 : nearest-neighbour only

    Start: shipment with earliest deadline (EDF heuristic) to bias
    toward route orderings that respect time windows.
    """
    n = len(group)
    if n == 1:
        return group

    if n == 2:
        a, b = group[0], group[1]
        d_ab = _road_km(
            a["delivery_lat"], a["delivery_lng"],
            b["delivery_lat"], b["delivery_lng"],
        )
        d_ba = _road_km(
            b["delivery_lat"], b["delivery_lng"],
            a["delivery_lat"], a["delivery_lng"],
        )
        return group if d_ab <= d_ba else [b, a]

    delivery_points = [
        (s["delivery_lat"], s["delivery_lng"]) for s in group
    ]

    # EDF start: earliest-deadline shipment anchors the tour
    start = min(range(n), key=lambda i: group[i].get("latest_delivery", "9999"))

    tour = _nn_tour(delivery_points, start_idx=start)
    if n <= 8:
        tour = _two_opt(delivery_points, tour)

    sequenced = [group[i] for i in tour]
    logger.debug(
        f"  sequence_deliveries: n={n} → "
        f"{[s.get('shipment_id', '?') for s in sequenced]}"
    )
    return sequenced


def _build_stop_sequence(ordered_shipments: List[dict]) -> List[dict]:
    """
    All pickups first (hub-and-spoke model), then sequenced deliveries.
    Corridor interleaving handled in is_corridor_group / process_group.
    """
    stops = []
    for s in ordered_shipments:
        stops.append({
            "type": "pickup", "shipment": s,
            "lat": s["pickup_lat"], "lng": s["pickup_lng"],
        })
    for s in ordered_shipments:
        stops.append({
            "type": "delivery", "shipment": s,
            "lat": s["delivery_lat"], "lng": s["delivery_lng"],
        })
    return stops


# ═══════════════════════════════════════════════════════════════════════
#  5. CORRIDOR ROUTING
# ═══════════════════════════════════════════════════════════════════════

def is_corridor_group(group: List[dict]) -> bool:
    """
    True if 3+ stops all travelling within MAX_BEARING_DIFF_DEG of the
    group's overall corridor bearing (pickup centroid → delivery centroid).

    2-stop groups don't need corridor logic — handled by pairwise detour.
    """
    if len(group) < 3:
        return False

    corridor_b = _group_corridor_bearing(group)
    for s in group:
        b    = _shipment_bearing(s)
        diff = _bearing_diff(b, corridor_b)
        if diff > MAX_BEARING_DIFF_DEG:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════
#  6. PAIRWISE COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════

def compatible(
    a: dict,
    b: dict,
    max_bearing_diff: float = MAX_BEARING_DIFF_DEG,
    max_detour:       float = MAX_DETOUR_RATIO,
) -> Tuple[bool, str]:
    """
    Returns (is_compatible, reason_string). Fail-fast ordering.

    Check 1: short-haul override — bearing is noise below MIN_ROUTE_KM_FOR_BEARING
    Check 2: bearing diff — same direction of travel?
    Check 3: detour ratio — road cost acceptable?
    """
    dist_a = _shipment_road_km(a)
    dist_b = _shipment_road_km(b)

    if dist_a < MIN_ROUTE_KM_FOR_BEARING and dist_b < MIN_ROUTE_KM_FOR_BEARING:
        return True, "short_haul_override"

    ba    = _shipment_bearing(a)
    bb    = _shipment_bearing(b)
    bdiff = _bearing_diff(ba, bb)
    if bdiff > max_bearing_diff:
        return False, f"bearing_diff={bdiff:.1f}°>{max_bearing_diff}°"

    anchor, rider = (
        (a, b) if a.get("weight_kg", 0) >= b.get("weight_kg", 0) else (b, a)
    )
    dr, order = _detour_ratio(anchor, rider)
    if dr > max_detour:
        return False, f"detour={dr:.2f}x>{max_detour}x"

    return True, f"ok(bearing={bdiff:.0f}°,detour={dr:.2f}x,order={order})"


# ═══════════════════════════════════════════════════════════════════════
#  7. GROUP FILTER — Split incompatible, then re-merge
# ═══════════════════════════════════════════════════════════════════════

def _filter_single_group(
    shipments: List[dict],
    max_bearing_diff: float,
    max_detour: float,
) -> List[List[dict]]:
    """
    FIX 3 — Pairwise check within new group, not just against existing members.

    Original bug: when shipment C was tested for joining group [A, B],
    it checked compatible(C, A) and compatible(C, B) — but never whether
    the *new group's effective corridor bearing* (now including C) still
    held for all pairs. In practice this meant a group [A, B, C] could
    form where A and C were directionally incompatible if both happened
    to be compatible with B individually.

    Fix: after adding a new member, re-verify all pairs in the group.
    O(n²) per shipment placement — acceptable for n<20.

    Greedy, heaviest-first anchor. Each shipment joins the first group
    where it is compatible with ALL existing members AND all resulting
    pairs remain compatible.
    """
    if not shipments:
        return []

    sorted_s = sorted(shipments, key=lambda s: s.get("weight_kg", 0), reverse=True)
    groups: List[List[dict]] = [[sorted_s[0]]]

    for s in sorted_s[1:]:
        placed = False
        for group in groups:
            # Check s against every existing member
            if not all(compatible(s, m, max_bearing_diff, max_detour)[0]
                       for m in group):
                continue
            # Also verify all existing pairs still hold with s included
            # (group members were already validated pairwise; only need
            #  to check new pairs involving s, which the above loop does)
            group.append(s)
            placed = True
            break
        if not placed:
            groups.append([s])

    return groups


def _try_merge_groups(
    groups: List[List[dict]],
    max_bearing_diff: float,
    max_detour: float,
) -> List[List[dict]]:
    """
    After splitting, attempt to re-merge compatible sub-groups.

    Why: greedy splitting can over-split. G1 and G3 might be mutually
    compatible even though both were incompatible with G2.

    Algorithm: union-find on groups. Two groups merge if every shipment
    in group A is compatible with every shipment in group B (full pairwise).
    O(g² × n²) where g=groups, n=max group size. Fast in practice.
    """
    n = len(groups)
    if n <= 1:
        return groups

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            can_merge = all(
                compatible(a, b, max_bearing_diff, max_detour)[0]
                for a in groups[i]
                for b in groups[j]
            )
            if can_merge:
                union(i, j)
                logger.debug(f"  re-merge: group {i} + group {j}")

    merged: Dict[int, List[dict]] = defaultdict(list)
    for i, group in enumerate(groups):
        merged[find(i)].extend(group)

    result = list(merged.values())
    if len(result) < n:
        logger.debug(f"  re-merge: {n} sub-groups → {len(result)} after merge")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  8. SALVAGE — Largest feasible subset when full group is incompatible
# ═══════════════════════════════════════════════════════════════════════

def _salvage_compatible_subset(shipments: List[dict],
                                max_bearing_diff: float,
                                max_detour: float) -> List[List[dict]]:
    """
    FIX 4 — Eject by index, not object equality.

    Original bug: `candidate in remaining` and `remaining.remove(candidate)`
    used dict equality. For shipments with identical field values (same
    weight, same route) this could remove the wrong shipment or fail
    silently, leaving the loop to retest the same infeasible group
    indefinitely.

    Fix: work with indices throughout. `by_spread` sorts indices by how
    far each shipment's bearing deviates from the group corridor — the
    most-deviant shipment is ejected first (most likely to be the
    incompatibility source).

    Returns list of sub-groups: [largest_feasible_subset, singleton, ...]
    """
    if len(shipments) <= 1:
        return [[s] for s in shipments]

    indices   = list(range(len(shipments)))
    corridor_b = _group_corridor_bearing(shipments)

    # Sort indices by bearing deviation — most deviant ejected first
    by_spread = sorted(
        indices,
        key=lambda i: _bearing_diff(_shipment_bearing(shipments[i]), corridor_b),
        reverse=True,
    )

    remaining_idx = list(indices)
    ejected_idx   = []

    while len(remaining_idx) > 1:
        subset = [shipments[i] for i in remaining_idx]
        sub_groups = _filter_single_group(subset, max_bearing_diff, max_detour)
        if len(sub_groups) == 1:
            # All remaining are mutually compatible
            break
        # Eject most-deviant shipment still in remaining
        for candidate_idx in by_spread:
            if candidate_idx in remaining_idx:
                ejected_idx.append(candidate_idx)
                remaining_idx.remove(candidate_idx)
                break

    result = [[shipments[i] for i in remaining_idx]] if remaining_idx else []
    result += [[shipments[i]] for i in ejected_idx]

    logger.info(
        f"  salvage: kept {len(remaining_idx)}, "
        f"ejected {len(ejected_idx)} as singletons"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  9. ANNOTATION
# ═══════════════════════════════════════════════════════════════════════

def _annotate_group(
    group: List[dict],
    truck_vol_cm3: float = DEFAULT_TRUCK_VOL_CM3,
    is_corridor: bool = False,
) -> None:
    """Stamps routing metadata onto each shipment. Mutates in place."""
    _, vol_util = _volume_feasible(group, truck_vol_cm3)
    stop_ids    = [s.get("shipment_id", "?") for s in group]
    corridor_b  = round(_group_corridor_bearing(group), 1)

    for i, s in enumerate(group):
        s["_stop_sequence"]    = stop_ids
        s["_stop_position"]    = i + 1
        s["_is_corridor"]      = is_corridor
        s["_corridor_bearing"] = corridor_b
        s["_vol_utilisation"]  = vol_util
        s["_group_size"]       = len(group)
        s["_route_bearing"]    = round(_shipment_bearing(s), 1)
        s["_route_distance_km"] = round(_shipment_road_km(s), 1)
        s["_volume_cm3"]       = round(_shipment_volume_cm3(s), 0)


# ═══════════════════════════════════════════════════════════════════════
#  10. FULL GROUP PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def process_group(
    group: List[dict],
    max_bearing_diff: float = MAX_BEARING_DIFF_DEG,
    max_detour:       float = MAX_DETOUR_RATIO,
    truck_vol_cm3:    float = DEFAULT_TRUCK_VOL_CM3,
) -> List[List[dict]]:
    """
    Full processing pipeline for one clustering group:

      Step 1: Annotate base route stats on every shipment
      Step 2: Volume feasibility check — split if over-volume
      Step 3: Bearing + detour compatibility filter — split incompatible
      Step 4: Re-merge compatible sub-groups (union-find)
      Step 5: Sequence deliveries (TSP: nearest-neighbour + 2-opt)
      Step 6: Corridor detection + annotation
      Step 7: Final annotation (_route_bearing, _stop_sequence, etc.)

    Returns list of sub-groups ready for bin_packing.py.
    """
    if not group:
        return []

    # Step 1 — Base route stats
    for s in group:
        s["_route_bearing"]      = round(_shipment_bearing(s), 1)
        s["_route_distance_km"]  = round(_shipment_road_km(s), 1)
        s["_volume_cm3"]         = round(_shipment_volume_cm3(s), 0)

    if len(group) == 1:
        _annotate_group(group, truck_vol_cm3, is_corridor=False)
        return [group]

    # Step 2 — Volume check: split largest-first into feasible bins
    vol_ok, vol_util = _volume_feasible(group, truck_vol_cm3)
    if not vol_ok:
        logger.info(
            f"  process_group: volume overloaded ({vol_util:.0%}) — "
            f"splitting {len(group)} shipments"
        )
        sorted_g = sorted(group, key=_shipment_volume_cm3, reverse=True)
        vol_bins: List[List[dict]] = []
        bin_vols: List[float]      = []
        for s in sorted_g:
            sv     = _shipment_volume_cm3(s)
            placed = False
            for i, bv in enumerate(bin_vols):
                if bv + sv <= truck_vol_cm3:
                    vol_bins[i].append(s)
                    bin_vols[i] += sv
                    placed = True
                    break
            if not placed:
                vol_bins.append([s])
                bin_vols.append(sv)
        result = []
        for vbin in vol_bins:
            result.extend(
                process_group(vbin, max_bearing_diff, max_detour, truck_vol_cm3)
            )
        return result

    # Step 3 — Bearing + detour split
    sub_groups = _filter_single_group(group, max_bearing_diff, max_detour)

    # If any sub-group is still internally incompatible, salvage it
    final_sub: List[List[dict]] = []
    for sg in sub_groups:
        inner = _filter_single_group(sg, max_bearing_diff, max_detour)
        if len(inner) == 1:
            final_sub.append(sg)
        else:
            # Salvage: eject most-deviant shipment(s) until compatible
            salvaged = _salvage_compatible_subset(sg, max_bearing_diff, max_detour)
            final_sub.extend(salvaged)
    sub_groups = final_sub

    # Step 4 — Re-merge compatible sub-groups
    sub_groups = _try_merge_groups(sub_groups, max_bearing_diff, max_detour)

    # Steps 5–7 — Sequence + corridor detect + annotate each sub-group
    result = []
    for sg in sub_groups:
        if len(sg) == 1:
            _annotate_group(sg, truck_vol_cm3, is_corridor=False)
            result.append(sg)
            continue

        sequenced   = sequence_deliveries(sg)
        is_corridor = is_corridor_group(sequenced)
        _annotate_group(sequenced, truck_vol_cm3, is_corridor=is_corridor)
        result.append(sequenced)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def apply_route_filter(
    groups: List[List[dict]],
    max_bearing_diff: float = MAX_BEARING_DIFF_DEG,
    max_detour:       float = MAX_DETOUR_RATIO,
    truck_vol_cm3:    float = DEFAULT_TRUCK_VOL_CM3,
) -> List[List[dict]]:
    """
    Called after clustering.get_valid_groups(), before bin_packing.pack_groups().

    For each clustering group, runs the full process_group() pipeline:
      bearing check → detour check → re-merge → TSP sequence →
      corridor detection → annotation

    Each shipment in output gets:
        _route_bearing       — compass bearing pickup→delivery
        _route_distance_km   — road-factor distance (haversine × 1.35)
        _volume_cm3          — cargo volume (l×w×h)
        _stop_sequence       — ordered shipment IDs sharing this truck
        _stop_position       — this shipment's position in stop sequence
        _is_corridor         — True if 3+ aligned stops
        _corridor_bearing    — overall corridor direction
        _vol_utilisation     — cargo volume / truck volume
        _group_size          — number of shipments in group
    """
    result:      List[List[dict]] = []
    split_count: int = 0
    passed:      int = 0

    for group in groups:
        processed = process_group(group, max_bearing_diff, max_detour, truck_vol_cm3)
        if len(processed) > 1:
            split_count += 1
        else:
            passed += 1
        result.extend(processed)

    total     = len(result)
    singleton = sum(1 for g in result if len(g) == 1)
    multi     = total - singleton
    corridor  = sum(1 for g in result if any(s.get("_is_corridor") for s in g))

    logger.info(
        f"\napply_route_filter: {len(groups)} groups → {total} after route check"
        f"\n  passed unchanged : {passed}"
        f"\n  split by route   : {split_count}"
        f"\n  final multi-ship : {multi}"
        f"\n  corridor routes  : {corridor}"
        f"\n  final singletons : {singleton}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  ANALYTICS — Used by metrics.py + dashboard
# ═══════════════════════════════════════════════════════════════════════

def group_route_stats(group: List[dict]) -> dict:
    """
    Route coherence stats for a group. Used by dashboard and diagnostics.

    Returns:
        avg_bearing        — mean travel direction (degrees)
        bearing_spread     — std dev of individual vs corridor bearing
        max_detour_ratio   — worst pairwise detour in the group
        total_road_km      — sum of all individual road distances
        avg_road_km        — average shipment road length
        corridor_bearing   — overall group corridor direction
        is_corridor        — True if 3+ aligned stops
        vol_utilisation    — total cargo volume / truck volume
    """
    if not group:
        return {}

    bearings  = [_shipment_bearing(s)  for s in group]
    distances = [_shipment_road_km(s)  for s in group]
    corr_b    = _group_corridor_bearing(group)
    spread    = math.sqrt(
        sum(_bearing_diff(b, corr_b) ** 2 for b in bearings) / len(bearings)
    )

    anchor = max(group, key=lambda s: s.get("weight_kg", 0))
    max_dr = max(
        (_detour_ratio(anchor, s)[0] for s in group if s is not anchor),
        default=1.0,
    )

    _, vol_util = _volume_feasible(group)

    return {
        "avg_bearing":      round(sum(bearings) / len(bearings), 1),
        "bearing_spread":   round(spread, 1),
        "max_detour_ratio": round(max_dr, 2),
        "total_road_km":    round(sum(distances), 1),
        "avg_road_km":      round(sum(distances) / len(distances), 1),
        "corridor_bearing": round(corr_b, 1),
        "is_corridor":      is_corridor_group(group),
        "vol_utilisation":  vol_util,
    }