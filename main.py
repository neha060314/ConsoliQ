import logging
import sys
import pandas as pd
from collections import defaultdict

from engine.clustering    import get_valid_groups
from engine.route_compat  import apply_route_filter
from engine.bin_packing   import pack_groups, avg_load_factor, avg_weight_utilization, avg_volume_utilization
from engine.feedback      import FeedbackStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "pickup_lat", "pickup_lng",
    "earliest_delivery", "latest_delivery",
    "weight_kg"
}


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_shipments(path: str) -> list[dict]:
    """Load and validate shipments from a CSV file."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"CSV missing required columns: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} row(s) with missing required fields.")

    if df.empty:
        logger.error("No valid shipments remaining after cleaning.")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} valid shipments from '{path}'.")
    return df.to_dict("records")


# ═══════════════════════════════════════════════════════════════════════
#  Console output
# ═══════════════════════════════════════════════════════════════════════

def print_metrics(shipments: list[dict], groups: list[list[dict]]) -> None:
    """Print high-level performance metrics."""
    n_ships    = len(shipments)
    n_groups   = len(groups)
    singletons = sum(1 for g in groups if len(g) == 1)
    batched    = n_groups - singletons
    batched_ships = sum(len(g) for g in groups if len(g) > 1)

    trip_reduction  = (1 - n_groups / n_ships) * 100 if n_ships else 0
    weight_util     = avg_weight_utilization(groups) * 100
    volume_util     = avg_volume_utilization(groups) * 100
    load_factor     = avg_load_factor(groups) * 100

    # Cost model: avg INR 850/trip saved, avg 12km/trip, 0.27kg CO₂/km
    trips_saved     = n_ships - n_groups
    cost_savings    = trips_saved * 850
    co2_saved_kg    = trips_saved * 12 * 0.27

    oversize        = sum(1 for g in groups if g[0].get("_oversize"))
    stack_warnings  = sum(1 for g in groups if g[0].get("_stackability_warning"))

    print("\n" + "═" * 60)
    print("  📊  PERFORMANCE METRICS")
    print("═" * 60)
    print(f"  Shipments processed     : {n_ships}")
    print(f"  Dispatch groups         : {n_groups}")
    print(f"  Batched groups          : {batched}  ({batched_ships} shipments)")
    print(f"  Solo dispatches         : {singletons}")
    print()
    print(f"  Trip reduction          : {trip_reduction:.1f}%")
    print(f"  Avg weight utilization  : {weight_util:.1f}%")
    print(f"  Avg volume utilization  : {volume_util:.1f}%")
    print(f"  Avg load factor         : {load_factor:.1f}%")
    print()
    print(f"  Estimated cost savings  : ₹{cost_savings:,.0f}")
    print(f"  CO₂ saved               : {co2_saved_kg:.1f} kg")
    print(f"  Vehicles saved          : {trips_saved}")
    print()
    if oversize:
        print(f"  ⚠️  Oversize loads        : {oversize} (need special vehicle)")
    if stack_warnings:
        print(f"  ⚠️  Stackability warnings : {stack_warnings} (fragile + stackable mix)")
    print("═" * 60 + "\n")


def print_dispatch_summary(groups: list[list[dict]]) -> None:
    """Print per-group dispatch manifest."""
    print("═" * 75)
    print("  📦  DISPATCH MANIFEST")
    print("═" * 75)

    vehicle_tally   = defaultdict(int)
    oversize_groups = []

    for i, group in enumerate(groups, 1):
        total_w    = sum(s.get("weight_kg", 0) for s in group)
        vehicle    = group[0].get("_assigned_vehicle", "Unknown")
        ids        = [s.get("shipment_id", f"#{j}") for j, s in enumerate(group)]
        earliest   = min(s["earliest_delivery"][11:16] for s in group)
        latest     = max(s["latest_delivery"][11:16]   for s in group)
        tag        = "SOLO" if len(group) == 1 else f"{len(group)} ships"
        lf         = group[0].get("_load_factor", 0) * 100
        w_util     = group[0].get("_weight_utilization", 0) * 100
        oversize   = group[0].get("_oversize", False)
        stack_warn = group[0].get("_stackability_warning", False)

        flags = ""
        if oversize:   flags += " ⚠️ OVERSIZE"
        if stack_warn: flags += " ⚠️ STACK"

        print(f"\n  Group {i:02d}  │  {vehicle:<22} │  {total_w:>7.1f} kg"
              f"  │  {tag:<8}  │  {earliest}–{latest}"
              f"  │  LF {lf:.0f}%{flags}")
        print(f"           Shipments : {', '.join(ids)}")

        vehicle_tally[vehicle] += 1
        if oversize:
            oversize_groups.append(i)

    # Fleet utilisation
    print("\n" + "─" * 75)
    print("  🚛  FLEET UTILISATION")
    print("─" * 75)
    for vehicle, count in sorted(vehicle_tally.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {vehicle:<25} {bar}  ({count})")

    # Exceptions
    if oversize_groups:
        print("\n" + "─" * 75)
        print("  ❗ EXCEPTIONS  (require manual dispatch decision)")
        print("─" * 75)
        for gid in oversize_groups:
            group = groups[gid - 1]
            w = sum(s.get("weight_kg", 0) for s in group)
            ids = [s.get("shipment_id", "?") for s in group]
            print(f"  Group {gid:02d} — {w:.1f}kg exceeds all fleet vehicles: {ids}")

    print("═" * 75 + "\n")


def print_feedback_summary(store: FeedbackStore) -> None:
    """Print what the feedback system has learned so far."""
    summary = store.summary()
    if summary.get("zones_tracked", 0) == 0:
        return

    print("═" * 60)
    print("  🧠  LEARNING ENGINE STATUS")
    print("═" * 60)
    print(f"  Zones tracked           : {summary['zones_tracked']}")
    print(f"  Total observations      : {summary['total_observations']}")
    print(f"  Zones auto-adjusted     : {summary['zones_auto_adjusted']}")
    print(f"  Global avg load factor  : {summary['global_avg_load_factor'] * 100:.1f}%")
    print(f"  Resolution distribution : {summary['resolution_distribution']}")
    print(f"  Learning active         : {'YES ✅' if summary['learning_active'] else 'Building baseline...'}")
    print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    path      = "data/shipments.csv"
    shipments = load_shipments(path)

    # ── Phase 1: Cluster ──────────────────────────────────────────────
    groups = get_valid_groups(shipments)

    # ── Phase 2: Route compatibility filter ───────────────────────────
    # groups = apply_route_filter(groups)

    # ── Phase 3: 3D bin packing (weight + volume) ─────────────────────
    # groups = pack_groups(groups)

    # ── Phase 4: Metrics + manifest ───────────────────────────────────
    print_metrics(shipments, groups)
    print_dispatch_summary(groups)

    # ── Phase 5: Feedback learning (saves for next run) ───────────────
    # store = FeedbackStore()
    # store.record_run(groups, resolution_used=8)
    # print_feedback_summary(store)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()