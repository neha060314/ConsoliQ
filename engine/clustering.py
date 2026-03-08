"""
clustering.py — Shipment consolidation grouping for inter-city freight
-----------------------------------------------------------------------
Single responsibility: "which shipments CAN share a truck on the same route?"

Strategy — OSM names + coordinate math, combined:

  OSM (geocoder.py) gives us CITY-LEVEL names like "Pune", "Surat", "Hyderabad".
  These are stable, human-meaningful, and perfect for defining a LANE:
      lane = "Bhiwandi → Pune"

  Coordinate math handles what OSM cannot:
    - Two pickups both named "Bhiwandi" but 8km apart → same cluster (haversine)
    - Delivery named "Pimpri-Chinchwad" vs "Pune" → 12km apart → same dest cluster
    - Multi-drop: Mumbai→Pune + Mumbai→Nashik → Pune is 70km from Nashik → corridor

  Together: OSM names define the primary lane key for grouping.
  Coordinate distance handles fuzziness within and between lanes.

Three rules — ALL must hold:

  Rule 1 — Same origin cluster
    Pickups within PICKUP_RADIUS_KM (50km) of each other.
    OSM name used as primary bucket, union-find merges nearby OSM variants.
    "Bhiwandi Logistics Park" + "Thane" (15km) → same cluster.
    "Bhiwandi" + "Pune" (120km) → different clusters. Hard stop.

  Rule 2 — Route compatibility
    OSM delivery names match (exact lane) → direct consolidation.
    OR delivery coords within DELIVERY_CLUSTER_KM (50km) → same dest cluster.
    OR detour ratio < MAX_DETOUR_RATIO — symmetric baseline.

  Rule 3 — Time window overlap
    Shipments must share >= MIN_OVERLAP_SECONDS of delivery window.
    Hard stop — zero overlap → never group.

Physical constraint (not optimisation):
    Combined weight > largest vehicle → split. bin_packing.py owns everything below.

Bugs fixed vs original:
  FIX 1 — Union-find with live centroid recomputation (_build_origin_clusters)
           Merged cluster centroid recalculated from actual member shipments,
           not stale pre-merge per-bucket centroid.
  FIX 2 — Same live-centroid fix in _merge_delivery_buckets.
  FIX 3 — Symmetric detour baseline in _detour_ratio (was asymmetric).
  FIX 4 — Phase B corridor loop: mark i as absorbed when it merges,
           preventing a grown group being re-absorbed into another singleton.
  FIX 5 — Per-shipment _lane annotation in annotate() instead of group[0]._lane.
  FIX 6 — HITL hooks for borderline corridor and dest-cluster decisions.
"""

import copy
import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from engine.geocoder import get_hub_name, warm_cache, cache_stats
from fleet import FLEET

logger = logging.getLogger(__name__)

if not FLEET:
    raise ValueError("fleet.py: FLEET is empty — add at least one vehicle.")

_SORTED_FLEET = sorted(FLEET, key=lambda v: v["max_kg"])
_MAX_CAPACITY = _SORTED_FLEET[-1]["max_kg"]

# ── Tuning knobs ────────────────────────────────────────────────────────
PICKUP_RADIUS_KM     = 50.0
DELIVERY_CLUSTER_KM  = 50.0
MAX_DETOUR_RATIO     = 0.20
MIN_OVERLAP_SECONDS  = 30 * 60

# ── HITL thresholds ─────────────────────────────────────────────────────
HITL_CORRIDOR_DETOUR_HIGH = 0.10
HITL_CORRIDOR_DETOUR_LOW  = 0.20
HITL_DEST_CLUSTER_HIGH_KM = 25.0
HITL_DEST_CLUSTER_LOW_KM  = 50.0
HITL_WEIGHT_UTIL_MIN      = 0.50


# ═══════════════════════════════════════════════════════════════════════
#  HITL callback type
# ═══════════════════════════════════════════════════════════════════════

HITLReviewer = Callable[[List[dict], List[dict], str, dict], bool]


def _default_hitl_reviewer(
    group_a: List[dict],
    group_b: List[dict],
    reason: str,
    context: dict,
) -> bool:
    ids_a = [s.get("shipment_id", "?") for s in group_a]
    ids_b = [s.get("shipment_id", "?") for s in group_b]
    logger.info(
        f"[HITL] Review requested — reason={reason}\n"
        f"  Group A: {ids_a}\n"
        f"  Group B: {ids_b}\n"
        f"  Context: {context}\n"
        f"  Decision: AUTO-APPROVED (replace _default_hitl_reviewer for manual review)"
    )
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Geometry
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


def _centroid(shipments: List[dict], field: str) -> Tuple[float, float]:
    lats = [s[f"{field}_lat"] for s in shipments]
    lngs = [s[f"{field}_lng"] for s in shipments]
    return sum(lats) / len(lats), sum(lngs) / len(lngs)


def _detour_ratio(a: dict, b: dict) -> float:
    """
    FIX 3 — Symmetric baseline.
    Uses the midpoint of the two pickups as shared origin.
    Baseline = longer of the two direct solo trips.
    Returns ratio: 0.10 = 10% extra km. Lower is better.
    """
    origin_lat = (a["pickup_lat"] + b["pickup_lat"]) / 2
    origin_lng = (a["pickup_lng"] + b["pickup_lng"]) / 2

    direct_a = _haversine_km(origin_lat, origin_lng,
                              a["delivery_lat"], a["delivery_lng"])
    direct_b = _haversine_km(origin_lat, origin_lng,
                              b["delivery_lat"], b["delivery_lng"])

    # Symmetric baseline: longer solo trip is the reference
    direct = max(direct_a, direct_b)

    # Best sequential multi-drop order
    route_ab = (
        _haversine_km(origin_lat, origin_lng,
                      a["delivery_lat"], a["delivery_lng"])
        + _haversine_km(a["delivery_lat"], a["delivery_lng"],
                        b["delivery_lat"], b["delivery_lng"])
    )
    route_ba = (
        _haversine_km(origin_lat, origin_lng,
                      b["delivery_lat"], b["delivery_lng"])
        + _haversine_km(b["delivery_lat"], b["delivery_lng"],
                        a["delivery_lat"], a["delivery_lng"])
    )
    best_multi = min(route_ab, route_ba)

    if direct <= 0:
        return 0.0

    # Extra km beyond the solo baseline, as a ratio
    extra = best_multi - direct
    return max(extra / direct, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Step 1 — Validate
# ═══════════════════════════════════════════════════════════════════════

def validate(shipments: List[dict]) -> List[dict]:
    valid = []
    for s in shipments:
        sid = s.get("shipment_id", "?")
        try:
            e = datetime.fromisoformat(s["earliest_delivery"])
            l = datetime.fromisoformat(s["latest_delivery"])
            if e >= l:
                logger.warning(f"Dropped {sid}: earliest >= latest")
                continue
        except (KeyError, ValueError) as err:
            logger.warning(f"Dropped {sid}: bad time window — {err}")
            continue
        valid.append(s)

    dropped = len(shipments) - len(valid)
    if dropped:
        logger.info(f"validate: dropped {dropped}, {len(valid)} remain")
    return valid


# ═══════════════════════════════════════════════════════════════════════
#  Step 2 — Assign hub names via OSM + coordinate fallback
# ═══════════════════════════════════════════════════════════════════════

def assign_hubs(shipments: List[dict]) -> List[dict]:
    unique_coords = list({
        (s["pickup_lat"],   s["pickup_lng"])   for s in shipments
    } | {
        (s["delivery_lat"], s["delivery_lng"]) for s in shipments
    })

    warm_cache(unique_coords)

    geo = cache_stats()
    if geo["empty"] > 0:
        logger.warning(
            f"assign_hubs: {geo['empty']} coords unresolved by OSM "
            f"— will use coordinate buckets for those lanes"
        )

    result = []
    for s in shipments:
        s2 = copy.copy(s)
        s2["_pickup_hub"]   = get_hub_name(s["pickup_lat"],   s["pickup_lng"])
        s2["_delivery_hub"] = get_hub_name(s["delivery_lat"], s["delivery_lng"])
        s2["_lane"]         = f"{s2['_pickup_hub']} → {s2['_delivery_hub']}"
        result.append(s2)

    lane_counts: Dict[str, int] = defaultdict(int)
    for s in result:
        lane_counts[s["_lane"]] += 1
    top = sorted(lane_counts.items(), key=lambda x: -x[1])[:10]
    logger.info(
        f"assign_hubs: {len(result)} shipments across {len(lane_counts)} lanes\n"
        + "\n".join(f"    {c:>3}  {lane}" for lane, c in top)
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Step 3 — Time window helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _window_overlap_seconds(group: List[dict], candidate: dict) -> float:
    all_s        = group + [candidate]
    latest_start = max(_parse_dt(s["earliest_delivery"]) for s in all_s)
    earliest_end = min(_parse_dt(s["latest_delivery"])   for s in all_s)
    return (earliest_end - latest_start).total_seconds()


def _groups_overlap(a: List[dict], b: List[dict]) -> bool:
    combined     = a + b
    latest_start = max(_parse_dt(s["earliest_delivery"]) for s in combined)
    earliest_end = min(_parse_dt(s["latest_delivery"])   for s in combined)
    return (earliest_end - latest_start).total_seconds() >= MIN_OVERLAP_SECONDS


# ═══════════════════════════════════════════════════════════════════════
#  Union-find primitives
# ═══════════════════════════════════════════════════════════════════════

def _find(parent: Dict[str, str], x: str) -> str:
    """Path-compressed find."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: Dict[str, str], a: str, b: str) -> str:
    """Merge b's set into a's set. Returns the new root."""
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra
    return _find(parent, a)


# ═══════════════════════════════════════════════════════════════════════
#  Step 4 — Origin cluster assignment
#  FIX 1: live centroid recomputation after every union.
# ═══════════════════════════════════════════════════════════════════════

def _build_origin_clusters(shipments: List[dict]) -> Dict[str, List[dict]]:
    """
    FIX 1 — Live centroid recomputation.

    Original bug: centroids dict was built once before any merges.
    After union(A, B), _find(parent, X) might return A, but centroids[A]
    was computed only from A's original shipments, not A+B combined.
    Subsequent proximity checks used the wrong centroid, causing missed
    merges and phantom splits.

    Fix: maintain a members dict that accumulates shipments as unions happen.
    Every proximity check recomputes centroid from current members[root],
    so it always reflects the true merged cluster.
    """
    # Pass 1 — OSM name buckets
    osm_buckets: Dict[str, List[dict]] = defaultdict(list)
    for s in shipments:
        osm_buckets[s["_pickup_hub"]].append(s)

    bucket_names = list(osm_buckets.keys())
    parent: Dict[str, str] = {name: name for name in bucket_names}

    # Live member tracking — updated on every union
    members: Dict[str, List[dict]] = {
        name: list(ships) for name, ships in osm_buckets.items()
    }

    for i, a in enumerate(bucket_names):
        for b in bucket_names[i + 1:]:
            ra = _find(parent, a)
            rb = _find(parent, b)
            if ra == rb:
                continue
            # Use LIVE members of current roots for centroid
            ca = _centroid(members[ra], "pickup")
            cb = _centroid(members[rb], "pickup")
            dist = _haversine_km(ca[0], ca[1], cb[0], cb[1])
            if dist <= PICKUP_RADIUS_KM:
                new_root = _union(parent, ra, rb)
                old_root = rb if new_root == ra else ra
                # Merge member lists into the surviving root
                members[new_root] = members[ra] + members[rb]
                # Clean up the absorbed root to avoid stale lookups
                if old_root in members:
                    del members[old_root]
                logger.debug(
                    f"origin clusters: merged '{old_root}' into '{new_root}' "
                    f"({dist:.1f}km apart)"
                )

    # Build final clusters using resolved roots
    clusters: Dict[str, List[dict]] = defaultdict(list)
    for name, ships in osm_buckets.items():
        clusters[_find(parent, name)].extend(ships)

    logger.info(
        f"origin clusters: {len(bucket_names)} OSM names → "
        f"{len(clusters)} origin clusters"
    )
    return dict(clusters)


# ═══════════════════════════════════════════════════════════════════════
#  Step 5 — Route compatibility check (Rule 2)
# ═══════════════════════════════════════════════════════════════════════

def _route_compatible(a: dict, b: dict) -> Tuple[bool, str, dict]:
    """
    Returns (compatible, reason, context).

    Confidence tiers:
        exact_lane       — OSM names match, always auto-approve
        dest_cluster_Xkm — coords within DELIVERY_CLUSTER_KM, medium confidence
        corridor_detour  — detour ratio check, lower confidence
    """
    # Exact lane match — always auto-approve
    if a["_delivery_hub"] == b["_delivery_hub"]:
        return True, "exact_lane", {}

    dist = _haversine_km(
        a["delivery_lat"], a["delivery_lng"],
        b["delivery_lat"], b["delivery_lng"],
    )

    if dist <= DELIVERY_CLUSTER_KM:
        return True, f"dest_cluster_{dist:.0f}km", {"dest_dist_km": dist}

    ratio = _detour_ratio(a, b)
    if ratio <= MAX_DETOUR_RATIO:
        return True, f"corridor_detour_{ratio:.0%}", {"detour_ratio": ratio}

    return False, f"incompatible_dest_{dist:.0f}km", {}


# ═══════════════════════════════════════════════════════════════════════
#  Step 6 — Time-window greedy grouping within a lane
# ═══════════════════════════════════════════════════════════════════════

def _group_by_window(shipments: List[dict]) -> List[List[dict]]:
    """
    Greedy time-window grouping for shipments already on the same lane.
    Sorted by earliest_delivery. Each shipment joins the most recent
    compatible group (reversed search = latest windows first, best overlap).
    """
    if not shipments:
        return []

    sorted_s = sorted(shipments, key=lambda s: s["earliest_delivery"])
    groups   = [[sorted_s[0]]]

    for s in sorted_s[1:]:
        placed = False
        for group in reversed(groups):
            if _window_overlap_seconds(group, s) >= MIN_OVERLAP_SECONDS:
                group.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])

    return groups


# ═══════════════════════════════════════════════════════════════════════
#  Step 7 — Delivery hub union-find
#  FIX 2: live centroid recomputation (same fix as origin clusters).
# ═══════════════════════════════════════════════════════════════════════

def _merge_delivery_buckets(
    lane_buckets: Dict[str, List[dict]]
) -> Dict[str, List[dict]]:
    """
    FIX 2 — Live centroid recomputation on delivery side.

    Original bug: identical stale-centroid issue as _build_origin_clusters.
    "Pimpri-Chinchwad" and "Chakan" (22km apart) centroids were never
    updated after a union, so third/fourth nearby hubs failed to merge.

    Fix: same members-dict pattern as FIX 1.
    """
    d_names = list(lane_buckets.keys())
    if len(d_names) <= 1:
        return lane_buckets

    d_parent: Dict[str, str] = {name: name for name in d_names}
    d_members: Dict[str, List[dict]] = {
        name: list(ships) for name, ships in lane_buckets.items()
    }

    for i, a in enumerate(d_names):
        for b in d_names[i + 1:]:
            ra = _find(d_parent, a)
            rb = _find(d_parent, b)
            if ra == rb:
                continue
            ca = _centroid(d_members[ra], "delivery")
            cb = _centroid(d_members[rb], "delivery")
            dist = _haversine_km(ca[0], ca[1], cb[0], cb[1])
            if dist <= DELIVERY_CLUSTER_KM:
                new_root = _union(d_parent, ra, rb)
                old_root = rb if new_root == ra else ra
                d_members[new_root] = d_members[ra] + d_members[rb]
                if old_root in d_members:
                    del d_members[old_root]
                logger.debug(
                    f"delivery merge: '{old_root}' → '{new_root}' "
                    f"({dist:.1f}km apart)"
                )

    merged: Dict[str, List[dict]] = defaultdict(list)
    for name, ships in lane_buckets.items():
        merged[_find(d_parent, name)].extend(ships)

    if len(merged) < len(lane_buckets):
        logger.info(
            f"  delivery hubs: {len(lane_buckets)} OSM names → "
            f"{len(merged)} merged dest buckets"
        )
    return dict(merged)


# ═══════════════════════════════════════════════════════════════════════
#  Step 8 — HITL review gate
# ═══════════════════════════════════════════════════════════════════════

def _needs_hitl_review(reason: str, ctx: dict) -> bool:
    """
    Auto-approve: exact lane, tight dest cluster (<=25km), low detour (<=10%)
    HITL review:  medium dest cluster (25-50km), medium detour (10-20%)
    Auto-reject:  handled upstream in _route_compatible (never reaches here)
    """
    if reason == "exact_lane":
        return False

    if reason.startswith("dest_cluster_"):
        dist = ctx.get("dest_dist_km", 0)
        if dist <= HITL_DEST_CLUSTER_HIGH_KM:
            return False
        return True

    if reason.startswith("corridor_detour_"):
        ratio = ctx.get("detour_ratio", 0)
        if ratio <= HITL_CORRIDOR_DETOUR_HIGH:
            return False
        return True

    return True


# ═══════════════════════════════════════════════════════════════════════
#  Step 9 — Main grouping: exact lanes first, then corridor
#  FIX 4: Phase B marks i as absorbed after it merges.
# ═══════════════════════════════════════════════════════════════════════

def _group_origin_cluster(
    cluster_shipments: List[dict],
    hitl_reviewer: HITLReviewer = _default_hitl_reviewer,
) -> List[List[dict]]:
    """
    Groups all shipments from one origin cluster.

    Phase A — Exact + nearby lanes (same/merged delivery hub + time window)
    Phase B — Corridor grouping for remaining singletons

    FIX 4: After group i absorbs group j, i is added to absorbed so it
    cannot be consumed by a later singleton iteration. Without this, a
    grown group (i now has 2 shipments) could be incorrectly re-absorbed
    into another singleton, producing a 3-way group that violates the
    pairwise compatibility checks.
    """
    # Phase A
    lane_buckets: Dict[str, List[dict]] = defaultdict(list)
    for s in cluster_shipments:
        lane_buckets[s["_delivery_hub"]].append(s)

    lane_buckets = _merge_delivery_buckets(lane_buckets)

    all_groups: List[List[dict]] = []
    for dest_hub, ships in sorted(lane_buckets.items()):
        tw_groups = _group_by_window(ships)
        all_groups.extend(tw_groups)

    # Phase B — corridor grouping on singletons
    # FIX 4: absorbed tracks BOTH i and j indices after any merge.
    singletons  = [i for i, g in enumerate(all_groups) if len(g) == 1]
    absorbed    = set()
    corridor_ct = 0
    hitl_ct     = 0

    for i in singletons:
        if i in absorbed:
            continue
        gi = all_groups[i]

        for j in singletons:
            if j <= i or j in absorbed:
                continue
            gj = all_groups[j]

            combined_w = (
                sum(s.get("weight_kg", 0) for s in gi)
                + sum(s.get("weight_kg", 0) for s in gj)
            )
            if combined_w > _MAX_CAPACITY:
                continue

            if not _groups_overlap(gi, gj):
                continue

            compatible, reason, ctx = _route_compatible(gi[0], gj[0])
            if not compatible:
                continue

            ctx["combined_kg"] = combined_w
            ctx["capacity_kg"] = _MAX_CAPACITY
            ctx["utilisation"] = combined_w / _MAX_CAPACITY

            needs_review = _needs_hitl_review(reason, ctx)
            if needs_review:
                approved = hitl_reviewer(gi, gj, reason, ctx)
                hitl_ct += 1
                if not approved:
                    logger.info(
                        f"  [HITL] Rejected corridor merge: "
                        f"{gi[0].get('shipment_id')} + {gj[0].get('shipment_id')} "
                        f"reason={reason}"
                    )
                    continue

            merged = gi + gj
            for s in merged:
                s["_multi_drop"]      = True
                s["_corridor_reason"] = reason
                s["_hitl_reviewed"]   = needs_review

            all_groups[i] = merged
            all_groups[j] = None        # type: ignore[assignment]
            absorbed.add(j)
            # FIX 4: mark i absorbed so no later singleton can consume
            # the now-grown group i as if it were still a singleton
            absorbed.add(i)
            corridor_ct += 1
            gi = all_groups[i]

            logger.debug(
                f"  corridor: {gi[0].get('_pickup_hub')} — "
                f"merged ids={[s.get('shipment_id') for s in gi]} "
                f"reason={reason} combined={combined_w:.0f}kg hitl={needs_review}"
            )
            # i is now absorbed — stop trying to pair it further
            break

    if corridor_ct:
        logger.info(
            f"  corridor pass: {corridor_ct} groups formed "
            f"({hitl_ct} required HITL review)"
        )

    return [g for g in all_groups if g is not None]


# ═══════════════════════════════════════════════════════════════════════
#  Step 10 — Physical ceiling split
# ═══════════════════════════════════════════════════════════════════════

def split_oversized(groups: List[List[dict]]) -> List[List[dict]]:
    """
    Splits groups exceeding _MAX_CAPACITY. Hard constraint, not optimisation.
    Heaviest shipments fill first — minimises number of resulting sub-groups.
    """
    result = []
    for group in groups:
        total_w = sum(s.get("weight_kg", 0) for s in group)
        if total_w <= _MAX_CAPACITY:
            result.append(group)
            continue

        sorted_s    = sorted(group, key=lambda s: -s.get("weight_kg", 0))
        bins:        List[List[dict]] = []
        bin_weights: List[float]      = []

        for s in sorted_s:
            w      = s.get("weight_kg", 0)
            placed = False
            for i, bw in enumerate(bin_weights):
                if bw + w <= _MAX_CAPACITY:
                    bins[i].append(s)
                    bin_weights[i] += w
                    placed = True
                    break
            if not placed:
                bins.append([s])
                bin_weights.append(w)

        logger.info(
            f"split_oversized: {total_w:.0f}kg > {_MAX_CAPACITY:.0f}kg ceiling "
            f"→ {len(bins)} sub-groups"
        )
        result.extend(bins)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Step 11 — Annotate groups with clustering metadata
#  FIX 5: per-shipment _lane instead of group[0]._lane for all.
# ═══════════════════════════════════════════════════════════════════════

def annotate(groups: List[List[dict]]) -> List[List[dict]]:
    """
    FIX 5 — Per-shipment _lane annotation.

    Original bug: _lane was taken from group[0] and stamped onto every
    shipment in the group. In a corridor group (SH_A→Pune + SH_B→Nashik),
    both got _lane = "Bhiwandi → Pune", misleading bin_packing.py on
    drop ordering.

    Fix: each shipment keeps its own _lane (already set by assign_hubs).
    A corridor group-level summary is written as _group_lane.
    """
    result = []
    for group in groups:
        total_w = sum(s.get("weight_kg", 0) for s in group)

        if len(group) == 1:
            con_type = "singleton"
        elif any(s.get("_multi_drop") for s in group):
            con_type = "corridor"
        else:
            con_type = "exact_lane"

        origins    = list(dict.fromkeys(s.get("_pickup_hub",   "?") for s in group))
        dests      = list(dict.fromkeys(s.get("_delivery_hub", "?") for s in group))
        group_lane = (
            f"{origins[0]} → {dests[0]}"
            if len(dests) == 1
            else f"{origins[0]} → [{', '.join(dests)}]"
        )

        new_group = []
        for s in group:
            s2 = copy.copy(s)
            # FIX 5: _lane already correct per-shipment from assign_hubs —
            # do NOT overwrite it with group[0]'s lane.
            s2["_group_lane"]         = group_lane
            s2["_group_weight_kg"]    = round(total_w, 2)
            s2["_consolidation_type"] = con_type
            s2["_hitl_reviewed"]      = s.get("_hitl_reviewed", False)
            new_group.append(s2)
        result.append(new_group)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def get_valid_groups(
    shipments: List[dict],
    resolution: Optional[int] = None,
    hitl_reviewer: HITLReviewer = _default_hitl_reviewer,
) -> List[List[dict]]:
    """
    Returns consolidation-candidate groups for bin_packing.py.

    Each group = "these shipments CAN share a truck."
    bin_packing.py decides vehicle selection and load efficiency.

    Args:
        shipments:      raw shipment dicts from the ingestion layer
        resolution:     unused, kept for backward API compatibility
        hitl_reviewer:  optional callable for human-in-the-loop approval.
                        Signature: (group_a, group_b, reason, context) -> bool

    Pipeline:
        1.  validate          — drop bad time windows
        2.  assign_hubs       — OSM city names via geocoder.py
        3.  origin clusters   — union-find with live centroids     [FIX 1]
        4.  delivery merging  — union-find with live centroids     [FIX 2]
        5.  exact lane groups — same/merged delivery hub + time window
        6.  corridor groups   — detour ratio + HITL gate           [FIX 4]
        7.  split_oversized   — hard physical ceiling only
        8.  annotate          — per-shipment _lane metadata        [FIX 5]

    Each shipment in output gets:
        _pickup_hub           — "Bhiwandi"
        _delivery_hub         — "Pune"
        _lane                 — individual lane "Bhiwandi → Pune"
        _group_lane           — group summary (multi-dest for corridor)
        _group_weight_kg      — combined weight of its group
        _consolidation_type   — "exact_lane" | "corridor" | "singleton"
        _multi_drop           — True if corridor candidate
        _corridor_reason      — e.g. "dest_cluster_22km"
        _hitl_reviewed        — True if a human approved this group
    """
    if not shipments:
        logger.warning("get_valid_groups: empty input")
        return []

    shipments = validate(shipments)
    if not shipments:
        logger.error("get_valid_groups: no valid shipments after validation")
        return []

    shipments = assign_hubs(shipments)
    clusters  = _build_origin_clusters(shipments)

    all_groups: List[List[dict]] = []
    for cluster_name, cluster_ships in clusters.items():
        logger.info(
            f"grouping origin cluster '{cluster_name}': "
            f"{len(cluster_ships)} shipments"
        )
        groups = _group_origin_cluster(cluster_ships, hitl_reviewer=hitl_reviewer)
        all_groups.extend(groups)

    all_groups = split_oversized(all_groups)
    all_groups = annotate(all_groups)

    singleton_ct  = sum(1 for g in all_groups if len(g) == 1)
    exact_lane_ct = sum(1 for g in all_groups
                        if len(g) > 1 and not any(s.get("_multi_drop") for s in g))
    corridor_ct   = sum(1 for g in all_groups
                        if any(s.get("_multi_drop") for s in g))
    hitl_ct       = sum(1 for g in all_groups
                        if any(s.get("_hitl_reviewed") for s in g))
    total_batched = sum(len(g) for g in all_groups if len(g) > 1)
    saved_trucks  = sum(len(g) - 1 for g in all_groups if len(g) > 1)

    logger.info(
        f"\nget_valid_groups: {len(shipments)} shipments → {len(all_groups)} groups\n"
        f"  exact lane groups : {exact_lane_ct}\n"
        f"  corridor groups   : {corridor_ct}  ({hitl_ct} HITL-reviewed)\n"
        f"  singletons        : {singleton_ct}\n"
        f"  shipments batched : {total_batched}\n"
        f"  trucks saved (est): {saved_trucks}"
    )
    return all_groups