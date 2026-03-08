"""
geocoder.py — Reverse geocoding, one-stop shop
-----------------------------------------------
Single responsibility: turn (lat, lng) into a human place name.

Nothing in this file knows about shipments, lanes, trucks, or windows.
It only answers: "what place is at these coordinates?"

Public API:
    from engine.geocoder import get_hub_name, warm_cache, cache_stats, evict_bad_entries

    name = get_hub_name(19.296, 73.066)   # → "Bhiwandi"
    name = get_hub_name(18.658, 73.773)   # → "Taloja"

    # Before a batch run — pre-fetch all coords at once:
    warm_cache([(lat1, lng1), (lat2, lng2), ...])

Return contract — get_hub_name() ALWAYS returns a non-empty string, NEVER raises:
    "Bhiwandi"       — real place name  (best case)
    "Chennai"        — real place, resolved via zoom=8 fallback
    "18.6_72.2"      — coordinate bucket (coord in water / unmapped — fix your data)
    "invalid_coord"  — coordinate failed basic validation (NaN, wrong type, out of range)

Error handling layers (inside-out):
    1. Bad coord type/NaN/range → "invalid_coord" immediately, no network call wasted
    2. Coord outside India bbox → "invalid_coord" with clear log message
    3. Network failure          → logged with specific reason, tries zoom=8, then bucket
    4. HTTP error               → logged with status code, tries zoom=8, then bucket
    5. zoom=10 returns nothing  → silent retry at zoom=8 (city/district level)
    6. zoom=8 returns nothing   → WARNING logged (coord likely in water/bad test data)
    7. Both zooms empty         → coordinate bucket — always a string, never crashes caller
"""

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
_CACHE_DIR   = ".geocache"
_CACHE_FILE  = os.path.join(_CACHE_DIR, "nominatim.json")
_DELAY       = 1.1     # seconds between requests  (Nominatim ToS: ≤ 1 req/s)
_TIMEOUT     = 10      # seconds before giving up on a single request
_USER_AGENT  = "Lorri-Freight-Clustering/1.0 (logistics@lorri.in)"
_COORD_ROUND = 3       # decimal places for cache key  (~111m precision)
_BUCKET_SIZE = 0.2     # degrees per fallback bucket   (~22km)

# India bounding box (includes immediate neighbours for border zones)
_LAT_MIN, _LAT_MAX =  6.0,  38.0
_LNG_MIN, _LNG_MAX = 68.0,  98.0

# OSM address fields, most-specific → broadest, tuned for Indian industrial geography
_FIELD_PRIORITY: List[str] = [
    "industrial",     # MIDC estates, SEZs — best for pickup hubs
    "suburb",         # Bhiwandi, Chakan, Taloja — warehousing zones
    "neighbourhood",
    "city_district",  # zones within large metros
    "town",
    "city",
    "district",
    "state_district",
    "county",
    "state",          # last real name before bucket fallback
]


# ═══════════════════════════════════════════════════════════════════════
#  Coordinate validation
# ═══════════════════════════════════════════════════════════════════════

class _CoordError(Exception):
    """Raised internally for bad coordinates. Never escapes this module."""
    pass


def _validate_coord(lat: float, lng: float) -> None:
    """
    Raises _CoordError with a human-readable message for any bad coordinate.
    Checks: type, NaN/Inf, global range [-90/90, -180/180], India bbox.
    """
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        raise _CoordError(
            f"non-numeric: lat={lat!r} ({type(lat).__name__}), "
            f"lng={lng!r} ({type(lng).__name__})"
        )
    if math.isnan(lat) or math.isnan(lng):
        raise _CoordError(f"NaN coordinate: ({lat}, {lng})")
    if math.isinf(lat) or math.isinf(lng):
        raise _CoordError(f"infinite coordinate: ({lat}, {lng})")
    if not (-90 <= lat <= 90):
        raise _CoordError(f"latitude {lat} out of range [-90, 90]")
    if not (-180 <= lng <= 180):
        raise _CoordError(f"longitude {lng} out of range [-180, 180]")
    if not (_LAT_MIN <= lat <= _LAT_MAX and _LNG_MIN <= lng <= _LNG_MAX):
        raise _CoordError(
            f"({lat}, {lng}) outside India region "
            f"[lat {_LAT_MIN}–{_LAT_MAX}, lng {_LNG_MIN}–{_LNG_MAX}] — "
            f"likely ocean coord or wrong country in source data"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Coordinate bucket fallback
# ═══════════════════════════════════════════════════════════════════════

def _bucket(lat: float, lng: float) -> str:
    """
    Returns a ~22km coordinate bucket string used as last-resort fallback.
    Always non-empty — clustering can use it for lane grouping even if
    the name is ugly. Shows up in logs as a data quality signal.
    """
    return (
        f"{round(lat / _BUCKET_SIZE) * _BUCKET_SIZE:.1f}_"
        f"{round(lng / _BUCKET_SIZE) * _BUCKET_SIZE:.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Disk cache
# ═══════════════════════════════════════════════════════════════════════

class _NominatimCache:
    """
    JSON file cache for Nominatim address responses.

    Key format : "{lat},{lng},z{zoom}"   e.g. "19.296,73.066,z10"
    Value       : Nominatim address dict  {"suburb": "Bhiwandi", ...}
                  {} means confirmed failed lookup (water / unmapped)

    After first full run, all coords are cached — subsequent runs
    make zero network calls.
    """

    def __init__(self):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        self._data: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if not os.path.exists(_CACHE_FILE):
            return
        try:
            with open(_CACHE_FILE) as f:
                self._data = json.load(f)
            logger.debug(f"geocoder: cache loaded — {len(self._data)} entries")
        except json.JSONDecodeError as e:
            logger.error(
                f"geocoder: cache file corrupted ({e}) — "
                f"deleting and starting fresh. Re-run to re-fetch."
            )
            self._data = {}
            try:
                os.rename(_CACHE_FILE, _CACHE_FILE + ".corrupt")
            except OSError:
                pass
        except OSError as e:
            logger.error(f"geocoder: cannot read cache ({e}) — starting fresh")
            self._data = {}

    def _save(self):
        try:
            with open(_CACHE_FILE, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError as e:
            logger.error(
                f"geocoder: cannot write cache to {_CACHE_FILE} ({e}) — "
                f"results will not persist across runs"
            )

    def key(self, lat: float, lng: float, zoom: int) -> str:
        return f"{round(lat, _COORD_ROUND)},{round(lng, _COORD_ROUND)},z{zoom}"

    def is_cached(self, lat: float, lng: float, zoom: int = 10) -> bool:
        return self.key(lat, lng, zoom) in self._data

    def get(self, lat: float, lng: float, zoom: int) -> Optional[dict]:
        """Returns cached value or None if not cached yet."""
        return self._data.get(self.key(lat, lng, zoom))

    def put(self, lat: float, lng: float, zoom: int, addr: dict):
        """Stores a result and flushes to disk immediately."""
        self._data[self.key(lat, lng, zoom)] = addr
        self._save()

    def evict_empty(self) -> int:
        """
        Removes {} entries (failed lookups) so they re-fetch next run.
        Returns count removed.
        """
        bad = [k for k, v in self._data.items() if not v]
        for k in bad:
            del self._data[k]
        if bad:
            self._save()
            logger.info(f"geocoder: evicted {len(bad)} empty cache entries")
        return len(bad)

    def stats(self) -> dict:
        total = len(self._data)
        empty = sum(1 for v in self._data.values() if not v)
        return {
            "total":     total,
            "populated": total - empty,
            "empty":     empty,
            "zoom_10":   sum(1 for k in self._data if k.endswith("z10")),
            "zoom_8":    sum(1 for k in self._data if k.endswith("z8")),
        }


_cache = _NominatimCache()  # one instance per process


# ═══════════════════════════════════════════════════════════════════════
#  Nominatim HTTP — all network logic isolated here
# ═══════════════════════════════════════════════════════════════════════

def _fetch(lat: float, lng: float, zoom: int) -> Tuple[Optional[dict], Optional[str]]:
    """
    One Nominatim reverse-geocode call with full error handling.

    Returns (address_dict, error_string):
      ({"suburb": "Bhiwandi", ...}, None)  — success
      ({},                          None)  — success but nothing there (water)
      (None,                 "timed out")  — network failure

    address_dict=None means the call itself failed (network/HTTP error).
    address_dict={} means the call succeeded but the coord has no OSM data.
    These are different — {} gets cached (no point retrying water), None does not.
    """
    cached = _cache.get(lat, lng, zoom)
    if cached is not None:
        return cached, None

    time.sleep(_DELAY)

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json",
                    "addressdetails": 1, "zoom": zoom},
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        addr = resp.json().get("address", {})
        _cache.put(lat, lng, zoom, addr)
        return addr, None

    except Timeout:
        return None, f"timed out after {_TIMEOUT}s"
    except ConnectionError:
        return None, "no network — check connection"
    except HTTPError as e:
        code = e.response.status_code if e.response else "?"
        reason = e.response.reason if e.response else "unknown"
        if code == 429:
            return None, f"HTTP 429 rate-limited — reduce request frequency"
        if code == 403:
            return None, f"HTTP 403 forbidden — check User-Agent header"
        return None, f"HTTP {code} {reason}"
    except RequestException as e:
        return None, f"network error: {e}"
    except (ValueError, KeyError) as e:
        return None, f"unexpected response format: {e}"
    except Exception as e:
        return None, f"unexpected error: {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  Field resolution
# ═══════════════════════════════════════════════════════════════════════

def _resolve(addr: Optional[dict]) -> Optional[str]:
    """
    Returns the best human-readable name from an OSM address dict.
    Returns None if addr is None/empty or no usable field found.
    """
    if not addr:
        return None
    for field in _FIELD_PRIORITY:
        val = addr.get(field, "").strip()
        if val:
            return val
    return None



# ═══════════════════════════════════════════════════════════════════════
#  City fallback lookup (Fix: raw coordinates → human city name)
#  Resolves coordinates that Nominatim couldn't geocode (e.g. coastal
#  industrial areas, Chennai suburbs) into readable city labels.
#  Called automatically by get_hub_name() before returning a bucket string.
# ═══════════════════════════════════════════════════════════════════════

# (city_name, lat_min, lat_max, lng_min, lng_max) — more specific first
_CITY_FALLBACK: list = [
    ("Bhiwandi",           19.25, 19.40, 73.00, 73.12),
    ("Thane",              19.12, 19.26, 72.95, 73.05),
    ("Kalyan-Dombivli",    19.20, 19.29, 73.10, 73.22),
    ("Mumbai",             18.85, 19.20, 72.70, 73.00),
    ("Navi Mumbai",        18.95, 19.12, 73.00, 73.10),
    ("Raigad",             18.82, 19.00, 73.05, 73.28),
    ("Pimpri-Chinchwad",   18.60, 18.70, 73.75, 73.90),
    ("Pune",               18.40, 18.65, 73.68, 74.05),
    ("Nashik",             19.90, 20.10, 73.60, 73.95),
    ("Surat",              21.00, 21.35, 72.70, 73.05),
    ("Vadodara",           22.20, 22.55, 73.08, 73.35),
    ("Ahmedabad",          22.90, 23.22, 72.40, 72.82),
    ("Hyderabad",          17.20, 17.65, 78.18, 78.75),
    ("Bengaluru",          12.80, 13.10, 77.40, 77.82),
    ("Chennai",            12.70, 13.40, 79.78, 80.52),
    ("Kolkata",            22.38, 22.72, 88.18, 88.52),
    ("Delhi",              28.38, 28.92, 76.78, 77.42),
    ("Kanpur",             26.28, 26.62, 80.18, 80.52),
    ("Hubballi",           15.18, 15.52, 74.98, 75.32),
    ("Kolhapur",           16.58, 16.82, 74.08, 74.32),
    ("Dharwad",            15.38, 15.52, 75.00, 75.18),
    ("Nagpur",             21.05, 21.25, 79.00, 79.22),
    ("Bally",              22.58, 22.72, 88.28, 88.42),
    ("Bidhannagar",        22.55, 22.65, 88.38, 88.52),
]


def _city_fallback(lat: float, lng: float) -> Optional[str]:
    """
    Returns a human-readable city name from static bbox lookup.
    Called when Nominatim returns no result (water / unmapped industrial zones).
    Returns None if coords don't match any known city bbox.
    """
    for city, lat_min, lat_max, lng_min, lng_max in _CITY_FALLBACK:
        if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
            logger.debug(f"geocoder: ({lat},{lng}) city_fallback -> {city}")
            return city
    return None

# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def get_hub_name(lat: float, lng: float) -> str:
    """
    Returns a canonical place name for (lat, lng).

    ALWAYS returns a non-empty string. NEVER raises. NEVER returns None.
    Clustering calls this without any try/except — all errors handled here.

    Resolution order:
      1. Validate coords  → "invalid_coord" if bad, no network call wasted
      2. zoom=10 fetch    → suburb / industrial estate level (most specific)
      3. zoom=8 fetch     → city / district level (silent retry if zoom=10 empty)
      4. Bucket fallback  → "18.6_72.2" (coord in water or truly unmapped)

    What each return value means:
      "Bhiwandi"       — zoom=10 resolved correctly
      "Chennai"        — zoom=8 fallback (zoom=10 returned nothing)
      "18.6_72.2"      — bucket (Arabian Sea, fix generate_data.py)
      "invalid_coord"  — bad lat/lng in source data (NaN, wrong type, out of bounds)
    """

    # Step 1 — Validate: catch bad data before wasting a network call
    try:
        _validate_coord(lat, lng)
    except _CoordError as e:
        logger.error(f"geocoder: bad coordinate — {e}")
        return "invalid_coord"

    # Step 2 — zoom=10: suburb / industrial estate level
    addr10, err10 = _fetch(lat, lng, zoom=10)
    if err10:
        logger.warning(f"geocoder: ({lat},{lng}) z10 failed — {err10}")
        # Don't give up yet — fall through to zoom=8
    else:
        name = _resolve(addr10)
        if name:
            return name
        # addr10 = {} means coord exists but has no useful OSM data at this zoom

    # Step 3 — zoom=8: city / district level (broader, more likely to return something)
    addr8, err8 = _fetch(lat, lng, zoom=8)
    if err8:
        logger.warning(f"geocoder: ({lat},{lng}) z8 failed — {err8}")
    else:
        name = _resolve(addr8)
        if name:
            logger.debug(f"geocoder: ({lat},{lng}) z8 fallback → {name}")
            return name

    # Step 4 — City fallback: static bbox lookup before using raw coordinate bucket
    city = _city_fallback(lat, lng)
    if city:
        logger.info(f"geocoder: ({lat},{lng}) city_fallback resolved -> {city}")
        return city

    # Step 5 — Bucket: last resort, always non-empty
    b = _bucket(lat, lng)
    logger.warning(
        f"geocoder: cannot resolve ({lat},{lng}) — "
        f"Nominatim returned no place name at zoom=10 or zoom=8. "
        f"Likely ocean coord or unmapped area in source data. "
        f"Check generate_data.py → bucket {b}"
    )
    return b


def warm_cache(coord_pairs: List[Tuple[float, float]]) -> dict:
    """
    Pre-fetches geocoding for a list of (lat, lng) pairs in one batch.

    Call this before processing shipments so all geocoding happens
    upfront in one clear phase — grouping logic then runs against
    cache only with zero network calls.

    Invalid coords are skipped (not fetched, not crashed on).
    Failed fetches are logged but don't stop the batch.

    Returns:
        {
          "total":           200,   # input pairs (including dupes)
          "unique":          180,   # after deduplication
          "already_cached":  176,   # found in cache, no fetch needed
          "fetched":           3,   # successfully fetched from network
          "failed":            1,   # network/HTTP error (will bucket at runtime)
          "skipped_invalid":   0,   # bad coords in source data
        }
    """
    unique          = list({(lat, lng) for lat, lng in coord_pairs})
    already_cached  = 0
    to_fetch        = []
    skipped_invalid = 0

    for lat, lng in unique:
        try:
            _validate_coord(lat, lng)
        except _CoordError as e:
            logger.warning(f"geocoder: warm_cache skipping invalid coord — {e}")
            skipped_invalid += 1
            continue
        if _cache.is_cached(lat, lng, zoom=10):
            already_cached += 1
        else:
            to_fetch.append((lat, lng))

    if to_fetch:
        logger.info(
            f"geocoder: fetching {len(to_fetch)} new coordinates "
            f"(~{len(to_fetch) * _DELAY:.0f}s) | "
            f"{already_cached} already cached | "
            f"{skipped_invalid} invalid skipped"
        )

    fetched = 0
    failed  = 0
    for lat, lng in to_fetch:
        _, err = _fetch(lat, lng, zoom=10)
        if err:
            logger.warning(f"geocoder: ({lat},{lng}) fetch failed — {err}")
            failed += 1
        else:
            fetched += 1

    result = {
        "total":           len(coord_pairs),
        "unique":          len(unique),
        "already_cached":  already_cached,
        "fetched":         fetched,
        "failed":          failed,
        "skipped_invalid": skipped_invalid,
    }
    logger.info(
        f"geocoder: warm_cache done — "
        f"{fetched} fetched, {failed} failed, {already_cached} cached. "
        f"Cache: {_cache.stats()['total']} total entries."
    )
    return result


def cache_stats() -> dict:
    """
    Returns cache health stats.

        {
          "total":     412,   # all entries
          "populated": 410,   # entries with real address data
          "empty":       2,   # failed lookups  (water / unmapped)
          "zoom_10":   200,   # zoom=10 entries
          "zoom_8":     12,   # zoom=8 fallback entries
        }

    If "empty" > 0, those coords will bucket-fallback at runtime.
    Call evict_bad_entries() to clear them and re-fetch on next run.
    """
    return _cache.stats()


def evict_bad_entries() -> int:
    """
    Removes empty cache entries (failed Nominatim lookups).
    Use after fixing bad coordinates in source data, or after a
    Nominatim outage left empty entries in the cache.
    Returns number of entries removed.
    """
    return _cache.evict_empty()