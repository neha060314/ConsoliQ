"""
feedback.py  —  Continuous optimization via zone-level feedback learning
------------------------------------------------------------------------
This is the "AI / Learn from patterns" requirement from the problem statement.

How it works:
  After every consolidation run, outcomes (load_factor per zone) are stored
  in a JSON feedback store. On the next run, clustering.py reads this store
  and adjusts the H3 resolution per zone — tighter zones for underloaded
  areas, coarser zones for overloaded areas.

  This is a simple but real adaptive system:
    - No black-box ML needed
    - Fully explainable ("Zone X was at 62% last run → widened to res 7")
    - Improves measurably over multiple runs
    - Exactly what "operational feedback" means in logistics

  Inspired by: Delhivery's zone tuning system, Porter's dynamic radius.

Feedback store schema (feedback_store.json):
  {
    "zone_id": {
      "resolution":     8,          -- current H3 resolution for this zone
      "run_count":      5,          -- how many times we've seen this zone
      "avg_load_factor": 0.71,      -- exponential moving average
      "last_load_factor": 0.68,     -- most recent run
      "last_updated":   "2024-01-15T12:00:00",
      "adjustment_log": [           -- history of resolution changes
        {"run": 2, "from": 8, "to": 7, "reason": "low_load_factor"}
      ]
    }
  }

Usage:
  from engine.feedback import FeedbackStore

  store = FeedbackStore()                        # loads or creates store
  resolution = store.get_resolution(cell_id, default=8)
  store.update(cell_id, load_factor=0.81)        # after a run
  store.save()
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────
FEEDBACK_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "feedback_store.json")
EMA_ALPHA      = 0.5    # exponential moving average weight — raised from 0.3
                              # so current-run LF converges in ~2 runs instead of ~5
MIN_RESOLUTION = 6      # never go coarser than this
MAX_RESOLUTION = 10     # never go finer than this

# Thresholds that trigger resolution adjustment
LOW_LF_THRESHOLD  = 0.65   # load factor below this → coarsen (bigger zones)
HIGH_LF_THRESHOLD = 0.92   # load factor above this → fine (smaller zones, avoid overflow)
MIN_RUNS_TO_ADJUST = 2     # don't adjust until we have at least 2 data points


class FeedbackStore:
    """
    Persistent per-zone feedback store.

    Zones are identified by H3 cell IDs (strings like '88283082b3fffff').
    Each zone independently learns its optimal resolution.
    """

    def __init__(self, path: str = FEEDBACK_PATH):
        self.path  = path
        self._data: Dict[str, dict] = {}
        self._load()

    # ── Persistence ─────────────────────────────────────────────────

    def _load(self) -> None:
        """Load feedback store from disk. Creates empty store if not found."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self._data = json.load(f)
                logger.info(f"FeedbackStore: loaded {len(self._data)} zones from {self.path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"FeedbackStore: could not load {self.path}: {e} — starting fresh")
                self._data = {}
        else:
            logger.info(f"FeedbackStore: no store found at {self.path} — starting fresh")
            self._data = {}

    def save(self) -> None:
        """Persist feedback store to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)
            logger.info(f"FeedbackStore: saved {len(self._data)} zones → {self.path}")
        except IOError as e:
            logger.error(f"FeedbackStore: save failed: {e}")

    # ── Zone resolution ──────────────────────────────────────────────

    def reset(self, zone_id: str = None) -> int:
        """Reset EMA/run-count for a zone or all zones.

        Call this after regenerating shipment data with a new profile
        so stale EMA values from old runs don't pollute new learning.
        Resolution is preserved (only statistics are cleared).

        Args:
            zone_id: specific zone to reset, or None to reset ALL zones
        Returns:
            number of zones reset
        """
        def _blank(z):
            return {
                "resolution":       z.get("resolution", 8),
                "run_count":        0,
                "avg_load_factor":  0.0,
                "last_load_factor": 0.0,
                "adjustment_log":   [],
                "last_updated":     None,
            }

        if zone_id is not None:
            if zone_id in self._data:
                self._data[zone_id] = _blank(self._data[zone_id])
                self.save()
                return 1
            return 0

        n = len(self._data)
        for zid in list(self._data.keys()):
            self._data[zid] = _blank(self._data[zid])
        self.save()
        return n


    def get_resolution(self, zone_id: str, default: int = 8) -> int:
        """
        Return the learned resolution for a zone.
        Falls back to `default` for zones never seen before.
        """
        if zone_id not in self._data:
            return default
        return self._data[zone_id].get("resolution", default)

    def get_all_resolutions(self) -> Dict[str, int]:
        """Return {zone_id: resolution} for all known zones."""
        return {zid: z.get("resolution", 8) for zid, z in self._data.items()}

    # ── Update ───────────────────────────────────────────────────────

    def update(self, zone_id: str, load_factor: float, current_resolution: int) -> None:
        """
        Record a new load_factor observation for a zone and adjust
        resolution if the zone consistently under/over-performs.

        Args:
            zone_id:            H3 cell ID string
            load_factor:        Achieved load factor (0.0–1.0) for this zone this run
            current_resolution: H3 resolution used for this zone this run
        """
        now = datetime.utcnow().isoformat()

        if zone_id not in self._data:
            self._data[zone_id] = {
                "resolution":      current_resolution,
                "run_count":       0,
                "avg_load_factor": load_factor,
                "last_load_factor": load_factor,
                "last_updated":    now,
                "adjustment_log":  [],
            }

        z = self._data[zone_id]
        z["run_count"]       += 1
        z["last_load_factor"] = round(load_factor, 4)
        z["last_updated"]     = now

        # Exponential moving average
        z["avg_load_factor"] = round(
            EMA_ALPHA * load_factor + (1 - EMA_ALPHA) * z["avg_load_factor"], 4
        )

        # Adjust resolution if enough data
        if z["run_count"] >= MIN_RUNS_TO_ADJUST:
            self._maybe_adjust(zone_id, z)

    def _maybe_adjust(self, zone_id: str, z: dict) -> None:
        """
        Core learning logic:
          - Consistently low LF  → coarsen resolution (bigger zones = more batching)
          - Consistently high LF → fine resolution (smaller zones = less overflow)
        """
        avg_lf = z["avg_load_factor"]
        res    = z["resolution"]
        reason = None
        new_res = res

        if avg_lf < LOW_LF_THRESHOLD and res > MIN_RESOLUTION:
            new_res = res - 1
            reason  = f"low_load_factor (avg={avg_lf:.2f} < {LOW_LF_THRESHOLD})"

        elif avg_lf > HIGH_LF_THRESHOLD and res < MAX_RESOLUTION:
            new_res = res + 1
            reason  = f"high_load_factor (avg={avg_lf:.2f} > {HIGH_LF_THRESHOLD})"

        if reason and new_res != res:
            z["resolution"] = new_res
            z["adjustment_log"].append({
                "run":    z["run_count"],
                "from":   res,
                "to":     new_res,
                "reason": reason,
            })
            logger.info(
                f"FeedbackStore: zone {zone_id[:12]}… resolution {res}→{new_res} "
                f"({reason})"
            )

    # ── Batch update from a full run ────────────────────────────────

    def record_run(self, groups: List[List[dict]], resolution_used: int) -> None:
        """
        Convenience method: record outcomes for all zones in one call.
        Pass the output of pack_groups() (groups with _load_factor annotated).

        Zone ID is derived from a pure-Python lat/lng grid cell (no h3 needed).
        Grid size is controlled by resolution_used:
          res 6 → 1.0° grid  (~110km squares, coarse)
          res 7 → 0.5° grid  (~55km squares, default)
          res 8 → 0.25° grid (~28km squares, fine)
          res 9 → 0.1° grid  (~11km squares, very fine)
        This mirrors h3's resolution semantics without requiring C extensions.
        """
        # Resolution → grid step size mapping (degrees)
        GRID_STEP = {6: 1.0, 7: 0.5, 8: 0.25, 9: 0.1, 10: 0.05}
        step = GRID_STEP.get(resolution_used, 0.5)

        def _cell_id(lat: float, lng: float) -> str:
            """Snap lat/lng to nearest grid corner → deterministic zone key."""
            cell_lat = math.floor(lat / step) * step
            cell_lng = math.floor(lng / step) * step
            return f"grid_{cell_lat:.3f}_{cell_lng:.3f}_r{resolution_used}"

        zone_lf_samples: Dict[str, List[float]] = {}

        for group in groups:
            lf = group[0].get("_load_factor", 0.0)
            # Skip groups with zero or near-zero LF — these are uninitialized groups
            # that haven't been through bin_packing yet. Recording 0.0 would corrupt
            # the zone EMA and cause the summary to show artificially low global LF.
            if lf < 0.01:
                continue
            for s in group:
                try:
                    cell = _cell_id(s["pickup_lat"], s["pickup_lng"])
                    zone_lf_samples.setdefault(cell, []).append(lf)
                except Exception:
                    pass

        for cell, lf_list in zone_lf_samples.items():
            avg = sum(lf_list) / len(lf_list)
            self.update(cell, avg, resolution_used)

        self.save()
        logger.info(
            f"FeedbackStore.record_run: updated {len(zone_lf_samples)} zones"
        )

    # ── Analytics ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        High-level summary of what the feedback store has learned.
        Used by the dashboard's Learning panel.
        """
        if not self._data:
            return {"zones_tracked": 0, "message": "No feedback data yet."}

        zones      = list(self._data.values())
        # Only include zones with at least one real observation for the avg LF
        active_zones = [z for z in zones if z.get("avg_load_factor", 0) > 0.01]
        avg_lf     = (sum(z["avg_load_factor"] for z in active_zones) / len(active_zones)
                      if active_zones else 0.0)
        total_runs = sum(z["run_count"] for z in zones)
        adjusted   = sum(1 for z in zones if z["adjustment_log"])

        # Resolution distribution
        res_dist: Dict[int, int] = {}
        for z in zones:
            r = z["resolution"]
            res_dist[r] = res_dist.get(r, 0) + 1

        return {
            "zones_tracked":           len(zones),
            "total_observations":      total_runs,
            "zones_auto_adjusted":     adjusted,
            "global_avg_load_factor":  round(avg_lf, 3),
            "resolution_distribution": res_dist,
            "learning_active":         adjusted > 0,
        }

    def zone_history(self, zone_id: str) -> Optional[dict]:
        """Return full learning history for a specific zone."""
        return self._data.get(zone_id)

    def improvement_trend(self) -> List[dict]:
        """
        Returns zones where load factor has improved over time.
        Useful for dashboard — "These zones got better."
        """
        improvements = []
        for zone_id, z in self._data.items():
            log = z.get("adjustment_log", [])
            if log and z["avg_load_factor"] > LOW_LF_THRESHOLD:
                improvements.append({
                    "zone_id":          zone_id,
                    "adjustments_made": len(log),
                    "current_lf":       z["avg_load_factor"],
                    "current_res":      z["resolution"],
                })
        return sorted(improvements, key=lambda x: -x["current_lf"])


# ═══════════════════════════════════════════════════════════════════════
#  Integration helper — patch clustering to use per-zone resolutions
# ═══════════════════════════════════════════════════════════════════════

def get_zone_resolutions(
    shipments: List[dict],
    default_resolution: int,
    store: Optional[FeedbackStore] = None,
) -> Dict[str, int]:
    """
    For each shipment, look up the learned resolution for its zone.
    Returns {cell_at_default_res: learned_resolution}.

    Uses pure-Python lat/lng grid cells (no h3 required).
    """
    if store is None:
        return {}

    GRID_STEP = {6: 1.0, 7: 0.5, 8: 0.25, 9: 0.1, 10: 0.05}
    step = GRID_STEP.get(default_resolution, 0.5)

    def _cell_id(lat: float, lng: float) -> str:
        cell_lat = math.floor(lat / step) * step
        cell_lng = math.floor(lng / step) * step
        return f"grid_{cell_lat:.3f}_{cell_lng:.3f}_r{default_resolution}"

    zone_res: Dict[str, int] = {}
    for s in shipments:
        try:
            cell = _cell_id(s["pickup_lat"], s["pickup_lng"])
            if cell not in zone_res:
                zone_res[cell] = store.get_resolution(cell, default_resolution)
        except Exception:
            pass

    learned_count = sum(1 for r in zone_res.values() if r != default_resolution)
    if learned_count:
        logger.info(
            f"get_zone_resolutions: {learned_count}/{len(zone_res)} zones "
            f"using learned resolution (≠ default {default_resolution})"
        )

    return zone_res