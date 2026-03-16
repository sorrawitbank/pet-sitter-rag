"""
Load canonical pet types, provinces, and districts from CSV;
resolve LLM-extracted English names to IDs for metadata filtering.
"""

import csv
from pathlib import Path
from typing import Optional

from src.gemini.client import create_structured_llm
from src.rag.schemas import QueryMetadataExtraction, ResolvedMetadata
from langchain_core.messages import HumanMessage, SystemMessage

# Default paths relative to project root (where main.py / docs/ live)
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "docs"
PET_TYPES_CSV = _DEFAULT_DATA_DIR / "pet_types.csv"
PROVINCES_CSV = _DEFAULT_DATA_DIR / "provinces.csv"
DISTRICTS_CSV = _DEFAULT_DATA_DIR / "districts.csv"


def _normalize(s: str) -> str:
    """Normalize for lookup: lowercase, strip, collapse spaces."""
    return " ".join(s.strip().lower().split())


def _load_pet_types(path: Path = PET_TYPES_CSV) -> dict[str, int]:
    """Load pet_type name (normalized) -> pet_type_id."""
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name", "").strip()
            if not name:
                continue
            try:
                pid = int(row["pet_type_id"])
            except (KeyError, ValueError):
                continue
            out[_normalize(name)] = pid
    return out


def _load_provinces(path: Path = PROVINCES_CSV) -> dict[str, int]:
    """Load province name (normalized) -> province_id."""
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name", "").strip()
            if not name:
                continue
            try:
                pid = int(row["province_id"])
            except (KeyError, ValueError):
                continue
            out[_normalize(name)] = pid
    return out


def _load_districts(path: Path = DISTRICTS_CSV) -> dict[str, int]:
    """Load district name (normalized) -> district_id."""
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name", "").strip()
            if not name:
                continue
            try:
                did = int(row["district_id"])
            except (KeyError, ValueError):
                continue
            out[_normalize(name)] = did
    return out


# Module-level caches (lazy-loaded)
_pet_lookup: Optional[dict[str, int]] = None
_province_lookup: Optional[dict[str, int]] = None
_district_lookup: Optional[dict[str, int]] = None


def get_pet_type_lookup() -> dict[str, int]:
    global _pet_lookup
    if _pet_lookup is None:
        _pet_lookup = _load_pet_types()
    return _pet_lookup


def get_province_lookup() -> dict[str, int]:
    global _province_lookup
    if _province_lookup is None:
        _province_lookup = _load_provinces()
    return _province_lookup


def get_district_lookup() -> dict[str, int]:
    global _district_lookup
    if _district_lookup is None:
        _district_lookup = _load_districts()
    return _district_lookup


def resolve_pet_type_ids(names: list[str]) -> list[int]:
    """Resolve English pet type names to pet_type_id. Exact match (normalized)."""
    lookup = get_pet_type_lookup()
    ids: list[int] = []
    seen: set[int] = set()
    for n in names:
        key = _normalize(n)
        pid = lookup.get(key)
        if pid is not None and pid not in seen:
            ids.append(pid)
            seen.add(pid)
    return ids


def resolve_province_ids(names: list[str]) -> list[int]:
    """Resolve English province names to province_id. Exact match (normalized)."""
    lookup = get_province_lookup()
    ids: list[int] = []
    seen: set[int] = set()
    for n in names:
        key = _normalize(n)
        pid = lookup.get(key)
        if pid is not None and pid not in seen:
            ids.append(pid)
            seen.add(pid)
    return ids


def resolve_district_ids(names: list[str]) -> list[int]:
    """Resolve English district names to district_id. Exact match (normalized)."""
    lookup = get_district_lookup()
    ids: list[int] = []
    seen: set[int] = set()
    for n in names:
        key = _normalize(n)
        did = lookup.get(key)
        if did is not None and did not in seen:
            ids.append(did)
            seen.add(did)
    return ids


def _resolve_with_fuzzy(
    names: list[str],
    lookup: dict[str, int],
    score_cutoff: int = 80,
) -> list[int]:
    """
    Resolve names to IDs: exact match first, then best fuzzy match if score >= score_cutoff.
    Requires optional dependency: pip install rapidfuzz
    """
    try:
        from rapidfuzz import process
    except ImportError:
        # Fallback to exact-only
        keys = list(lookup.keys())
        ids: list[int] = []
        seen: set[int] = set()
        for n in names:
            key = _normalize(n)
            pid = lookup.get(key)
            if pid is not None and pid not in seen:
                ids.append(pid)
                seen.add(pid)
        return ids
    ids = []
    seen: set[int] = set()
    keys = list(lookup.keys())
    for n in names:
        key = _normalize(n)
        if key in lookup:
            pid = lookup[key]
            if pid not in seen:
                ids.append(pid)
                seen.add(pid)
            continue
        match = process.extractOne(key, keys, score_cutoff=score_cutoff)
        if match:
            canonical_key, score, _ = match
            pid = lookup[canonical_key]
            if pid not in seen:
                ids.append(pid)
                seen.add(pid)
    return ids


def resolve_extraction(
    extraction: QueryMetadataExtraction,
    *,
    use_fuzzy: bool = False,
    fuzzy_score_cutoff: int = 80,
) -> ResolvedMetadata:
    """
    Map LLM extraction (English names) to canonical IDs using CSV lookups.
    Use the returned ids in WHERE metadata.pet_type_id IN (...), etc.
    If use_fuzzy=True and rapidfuzz is installed, fallback to fuzzy match when exact fails.
    """
    if use_fuzzy:
        return ResolvedMetadata(
            pet_type_ids=_resolve_with_fuzzy(
                extraction.pet_types,
                get_pet_type_lookup(),
                score_cutoff=fuzzy_score_cutoff,
            ),
            province_ids=_resolve_with_fuzzy(
                extraction.provinces,
                get_province_lookup(),
                score_cutoff=fuzzy_score_cutoff,
            ),
            district_ids=_resolve_with_fuzzy(
                extraction.districts,
                get_district_lookup(),
                score_cutoff=fuzzy_score_cutoff,
            ),
        )
    return ResolvedMetadata(
        pet_type_ids=resolve_pet_type_ids(extraction.pet_types),
        province_ids=resolve_province_ids(extraction.provinces),
        district_ids=resolve_district_ids(extraction.districts),
    )


def get_canonical_lists_for_prompt() -> dict[str, list[str]]:
    """
    Return canonical English names for each dimension so you can inject
    them into the LLM system prompt (e.g. "Output only from: ...").
    """
    return {
        "pet_types": sorted(get_pet_type_lookup().keys()),
        "provinces": sorted(get_province_lookup().keys()),
        "districts": sorted(get_district_lookup().keys()),
    }


# ---------------------------------------------------------------------------
# LLM extraction: user query -> QueryMetadataExtraction (English-only)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """You are a metadata extractor. From the user's message, extract:
1. pet_types: which pet types they want (e.g. dog, cat, bird, rabbit). Output in English only: use exactly one of Dog, Cat, Bird, Rabbit for each.
2. provinces: which provinces they mention. Output in English only, using standard official names (e.g. Bangkok, Samut Prakarn, Nonthaburi).
3. districts: which districts they mention. Output in English only, using standard official names (e.g. Bang Bon, Bang Khun Thian, Phra Nakhon).

Rules:
- Output ALL values in English only. If the user writes in Thai, translate to the canonical English name.
- Use empty list [] for any dimension the user did not mention.
- Use only the exact canonical names (e.g. "Bangkok" not "BKK", "Dog" not "dogs")."""


def _build_extraction_prompt(canonical: Optional[dict[str, list[str]]] = None) -> str:
    """Build system prompt with optional canonical lists so LLM picks from closed set."""
    if not canonical:
        return _EXTRACTION_SYSTEM
    parts = [_EXTRACTION_SYSTEM, ""]
    if canonical.get("pet_types"):
        parts.append(
            "Allowed pet_types (use only these): " + ", ".join(canonical["pet_types"])
        )
    if canonical.get("provinces"):
        parts.append(
            "Allowed provinces (use only these when applicable): "
            + ", ".join(canonical["provinces"][:30])
        )
        if len(canonical["provinces"]) > 30:
            parts.append("... and " + str(len(canonical["provinces"]) - 30) + " more.")
    if canonical.get("districts"):
        parts.append(
            "Allowed districts (use only these when applicable): "
            + ", ".join(canonical["districts"][:40])
        )
        if len(canonical["districts"]) > 40:
            parts.append("... and " + str(len(canonical["districts"]) - 40) + " more.")
    return "\n".join(parts)


async def extract_query_metadata(
    query: str,
    *,
    include_canonical_in_prompt: bool = True,
) -> QueryMetadataExtraction:
    """
    Use LLM to extract pet_types, provinces, districts from the user query.
    All extracted values are in English so they can be resolved to IDs via resolve_extraction().
    """
    schema = QueryMetadataExtraction.model_json_schema()
    llm = create_structured_llm(schema, temperature=0.2)
    canonical = (
        get_canonical_lists_for_prompt() if include_canonical_in_prompt else None
    )
    system_content = _build_extraction_prompt(canonical)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ]
    result = await llm.ainvoke(messages)
    if isinstance(result, QueryMetadataExtraction):
        return result
    return QueryMetadataExtraction.model_validate(result)
