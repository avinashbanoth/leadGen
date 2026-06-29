import logging
import re

import aiohttp
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_NOMINATIM_URL    = "https://nominatim.openstreetmap.org/search"
_OVERPASS_URL     = "http://overpass-api.de/api/interpreter"
_USER_AGENT       = "lead-gen-agent/1.0"
_GEOCODE_TIMEOUT  = 10.0
_OVERPASS_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Geocoding — Nominatim (free, no auth; requires User-Agent per ToS)
# ---------------------------------------------------------------------------

async def _geocode(location: str) -> tuple[float, float, float, float] | None:
    """
    Converts a location name to a (south, west, north, east) bounding box.
    Uses Nominatim (OpenStreetMap geocoder). Returns None on any failure.
    Nominatim returns [south, north, west, east]; we reorder for Overpass.
    """
    params  = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": _USER_AGENT}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                _NOMINATIM_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=_GEOCODE_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if not data:
            logger.warning("overpass_tool: Nominatim returned no results for '%s'", location)
            return None

        bb = data[0].get("boundingbox", [])
        if len(bb) < 4:
            return None

        # Nominatim: [south, north, west, east]
        # Overpass:   south, west, north, east
        s, n, w, e = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        logger.info(
            "overpass_tool: geocoded '%s' → bbox (%.2f, %.2f, %.2f, %.2f)",
            location, s, w, n, e,
        )
        return (s, w, n, e)

    except Exception as exc:
        logger.warning("overpass_tool: geocode failed for '%s' — %s", location, exc)
        return None


# ---------------------------------------------------------------------------
# Overpass QL query builders
# ---------------------------------------------------------------------------

def _safe_pattern(keywords: list[str]) -> str:
    """
    Builds a case-insensitive OR regex pattern from keywords for Overpass QL.
    Strips characters that are special in POSIX ERE (Overpass regex dialect).
    """
    clean = []
    for kw in keywords[:3]:
        kw = kw.strip()
        if len(kw) < 2:
            continue
        # Remove chars that break Overpass QL string literals / ERE
        kw = re.sub(r'["\\\[\](){}.*+?^$]', "", kw)
        if kw:
            clean.append(kw)
    return "|".join(clean) if clean else ""


def _build_keyword_query(bbox: tuple, pattern: str, limit: int) -> str:
    """Primary query: businesses whose OSM name matches the keyword pattern."""
    s, w, n, e = bbox
    bb = f"{s},{w},{n},{e}"
    return f"""[out:json][timeout:25];
(
  node["office"]["name"~"{pattern}",i]({bb});
  way["office"]["name"~"{pattern}",i]({bb});
  node["shop"]["name"~"{pattern}",i]({bb});
  way["shop"]["name"~"{pattern}",i]({bb});
  node["craft"]["name"~"{pattern}",i]({bb});
  way["craft"]["name"~"{pattern}",i]({bb});
  node["industrial"]["name"~"{pattern}",i]({bb});
);
out body center {limit};"""


def _build_office_query(bbox: tuple, limit: int) -> str:
    """Fallback query: any named office within the bbox, no keyword filter."""
    s, w, n, e = bbox
    bb = f"{s},{w},{n},{e}"
    return f"""[out:json][timeout:25];
(
  node["office"]["name"]({bb});
  way["office"]["name"]({bb});
);
out body center {limit};"""


# ---------------------------------------------------------------------------
# Overpass HTTP call
# ---------------------------------------------------------------------------

async def _run_overpass(query: str) -> list[dict]:
    """POSTs an Overpass QL query and returns the raw elements list."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OVERPASS_URL,
                data={"data": query},
                timeout=aiohttp.ClientTimeout(total=_OVERPASS_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return data.get("elements", [])
    except Exception as exc:
        logger.warning("overpass_tool: Overpass request failed — %s", exc)
        return []


# ---------------------------------------------------------------------------
# OSM element → company dict
# ---------------------------------------------------------------------------

def _extract_company(element: dict) -> dict | None:
    """
    Converts a raw OSM element to a company dict.
    Returns None when the element has no usable name.
    """
    tags = element.get("tags", {})
    name = (tags.get("name") or "").strip()
    if not name or len(name) < 2:
        return None

    website = (
        tags.get("website")
        or tags.get("contact:website")
        or tags.get("url")
        or ""
    ).strip()
    if website and not website.startswith("http"):
        website = f"https://{website}"

    phone = (tags.get("phone") or tags.get("contact:phone") or "").strip()

    addr_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
        tags.get("addr:state", ""),
    ]
    address = ", ".join(p for p in addr_parts if p)

    return {
        "name"      : name,
        "website"   : website,
        "address"   : address,
        "phone"     : phone,
        "source"    : "overpass_osm",
        "confidence": 0.55 if website else 0.35,
    }


# ---------------------------------------------------------------------------
# @tool
# ---------------------------------------------------------------------------

@tool
async def search_overpass_businesses(
    keywords   : list[str],
    location   : str,
    max_results: int = 10,
) -> list[dict]:
    """
    Queries OpenStreetMap via the free Overpass API for businesses that match
    the given keywords and are physically located in the specified area.
    No API key required. Activates only when a location is known.

    Strategy:
      1. Geocode location to bounding box via Nominatim.
      2. Run keyword-filtered query (office/shop/craft nodes whose OSM name
         contains one of the keywords).
      3. If keyword query returns nothing, fall back to all named offices in
         the bbox (useful when company names don't contain industry keywords).
      4. Deduplicate by name, return up to max_results results.

    Returns a list of dicts: name, website, address, phone, source, confidence.
    website and phone are empty strings when not recorded in OSM.
    """
    if not location:
        return []

    # Step 1 — geocode
    bbox = await _geocode(location)
    if not bbox:
        logger.warning("overpass_tool: geocode failed for '%s' — skipping.", location)
        return []

    # Step 2 — build keyword pattern
    pattern = _safe_pattern(keywords)
    fetch_limit = max(max_results * 3, 30)   # fetch extra; we deduplicate below

    # Step 3 — primary query (keyword-filtered)
    elements: list[dict] = []
    if pattern:
        query    = _build_keyword_query(bbox, pattern, fetch_limit)
        elements = await _run_overpass(query)
        logger.info(
            "overpass_tool: keyword query '%s' in '%s' → %d elements.",
            pattern, location, len(elements),
        )

    # Step 4 — fallback (general office search)
    if not elements:
        fallback = _build_office_query(bbox, fetch_limit)
        elements = await _run_overpass(fallback)
        logger.info(
            "overpass_tool: fallback office query in '%s' → %d elements.",
            location, len(elements),
        )

    # Step 5 — extract and deduplicate
    seen:      set[str]  = set()
    companies: list[dict] = []
    for el in elements:
        company = _extract_company(el)
        if not company:
            continue
        key = company["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        companies.append(company)
        if len(companies) >= max_results:
            break

    logger.info(
        "overpass_tool: returning %d unique businesses in '%s'.",
        len(companies), location,
    )
    return companies
