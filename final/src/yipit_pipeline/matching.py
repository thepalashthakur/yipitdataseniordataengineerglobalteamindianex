"""Auditable company identity resolution."""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Mapping, Tuple
from uuid import uuid5

from .config import AMBIGUOUS_COMPANY_NAMES, COMPANY_ALIASES, COMPANY_NAMESPACE
from .schemas import CompanyResolution


def normalize_company_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def company_id_for(entity_name: str) -> str:
    return str(uuid5(COMPANY_NAMESPACE, "company:{}".format(normalize_company_name(entity_name))))


def _normalized_lookup(metadata: Mapping[str, object]) -> Dict[str, str]:
    return {normalize_company_name(name): name for name in metadata}


def _normalized_aliases() -> Dict[str, str]:
    return {normalize_company_name(alias): canonical for alias, canonical in COMPANY_ALIASES.items()}


def fuzzy_company_suggestion(raw_name: str, metadata: Mapping[str, object]) -> Tuple[str, float]:
    normalized = normalize_company_name(raw_name)
    candidates = [
        (name, SequenceMatcher(None, normalized, normalize_company_name(name)).ratio())
        for name in metadata
    ]
    return max(candidates, key=lambda item: item[1]) if candidates else ("", 0.0)


def resolve_company(raw_name: str, metadata: Mapping[str, object]) -> CompanyResolution:
    raw = str(raw_name or "").strip()
    if raw in metadata:
        return CompanyResolution(raw, company_id_for(raw), raw, raw, "MATCHED", "EXACT", 1.0, True)

    normalized_lookup = _normalized_lookup(metadata)
    normalized = normalize_company_name(raw)
    if normalized in normalized_lookup:
        canonical = normalized_lookup[normalized]
        return CompanyResolution(raw, company_id_for(canonical), canonical, canonical, "MATCHED", "NORMALIZED", 0.99, True)

    alias_lookup = _normalized_aliases()
    canonical = alias_lookup.get(normalized)
    if canonical and canonical in metadata:
        return CompanyResolution(raw, company_id_for(canonical), canonical, canonical, "MATCHED", "ALIAS", 0.98, True)

    suggested_name, score = fuzzy_company_suggestion(raw, metadata)
    status = "AMBIGUOUS" if raw in AMBIGUOUS_COMPANY_NAMES else "UNRESOLVED"
    method = "AMBIGUOUS_SOURCE" if status == "AMBIGUOUS" else "UNRESOLVED"
    # The suggestion is intentionally not used as the entity identity.
    confidence = round(score, 4) if suggested_name else 0.0
    return CompanyResolution(raw, company_id_for(raw), None, raw, status, method, confidence, False)
