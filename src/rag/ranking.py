"""
Sitter ranking logic: group docs by source_id, score by doc count and rank sum, return top 3.
"""

from collections import defaultdict
import json
from typing import Any, Dict, List, Tuple


def rank_sitters(
    docs: List[Dict[str, Any]],
    effective_top_k: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Rank sitters by: (1) number of documents per sitter (more = better);
    (2) tie-break by sum of rank scores (rank 1 = effective_top_k pts, rank 2 = effective_top_k-1, ...).
    Returns top 3 sitters as list of (sitter_id, docs_for_sitter).
    """
    if not docs:
        return []

    for i, doc in enumerate(docs):
        doc["_rank_score"] = effective_top_k - i

    by_sitter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        sid = doc.get("source_id")
        if sid is not None:
            by_sitter[str(sid)].append(doc)

    sitter_scores: List[Tuple[str, int, int]] = []
    for sitter_id, sitter_docs in by_sitter.items():
        doc_count = len(sitter_docs)
        rank_sum = sum(d["_rank_score"] for d in sitter_docs)
        sitter_scores.append((sitter_id, doc_count, rank_sum))

    sitter_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_sitter_ids = [s[0] for s in sitter_scores[:3]]

    return [(sid, by_sitter[sid]) for sid in top_sitter_ids]


def get_trade_name(doc: Dict[str, Any]) -> str:
    """Get tradeName from document metadata."""
    meta = doc.get("metadata")

    # If metadata is a JSON string, parse it first
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return ""

    if isinstance(meta, dict):
        name = meta.get("tradeName")
        if isinstance(name, str) and name.strip():
            return name.strip()

    return ""
