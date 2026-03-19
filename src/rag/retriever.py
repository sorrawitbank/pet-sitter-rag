"""
LangChain custom retriever: vector search + sitter ranking, returns top-3 sitters as Documents.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.api.repositories.document import TOP_K_DEFAULT, get_similar_rag_documents
from src.gemini.client import get_text_embedding
from src.rag.ranking import get_trade_name, rank_sitters
from src.rag.schemas import ResolvedMetadata

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "docs"
PROVINCES_CSV = _DEFAULT_DATA_DIR / "provinces.csv"
DISTRICTS_CSV = _DEFAULT_DATA_DIR / "districts.csv"
PET_TYPES_CSV = _DEFAULT_DATA_DIR / "pet_types.csv"

_province_name_by_id: Optional[dict[int, str]] = None
_district_name_by_id: Optional[dict[int, str]] = None
_pet_type_name_by_id: Optional[dict[int, str]] = None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_id_name_map(path: Path, id_col: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get(id_col)
            raw_name = row.get("name", "")
            name = str(raw_name).strip()
            parsed_id = _safe_int(raw_id)
            if parsed_id is None or not name:
                continue
            out[parsed_id] = name
    return out


def get_province_name_by_id() -> dict[int, str]:
    global _province_name_by_id
    if _province_name_by_id is None:
        _province_name_by_id = _load_id_name_map(PROVINCES_CSV, "province_id")
    return _province_name_by_id


def get_district_name_by_id() -> dict[int, str]:
    global _district_name_by_id
    if _district_name_by_id is None:
        _district_name_by_id = _load_id_name_map(DISTRICTS_CSV, "district_id")
    return _district_name_by_id


def get_pet_type_name_by_id() -> dict[int, str]:
    global _pet_type_name_by_id
    if _pet_type_name_by_id is None:
        _pet_type_name_by_id = _load_id_name_map(PET_TYPES_CSV, "pet_type_id")
    return _pet_type_name_by_id


class SitterRankingRetriever(BaseRetriever):
    """
    Retriever that: (1) embeds the query and fetches top-k similar chunks from rag_documents;
    (2) ranks sitters by doc count then rank-sum; (3) returns one Document per top-3 sitter.
    Each Document has page_content (SITTER_ID, TRADE_NAME, content blocks) and metadata
    (sitter_id, trade_name) for downstream use.
    """

    top_k: int = TOP_K_DEFAULT
    """Number of document chunks to fetch from the vector store before ranking (max 10)."""
    filters: Optional[ResolvedMetadata] = None

    class Config:
        arbitrary_types_allowed = True

    def _effective_top_k(self) -> int:
        k = self.top_k
        if k > 10:
            k = 10
        if k < 1:
            k = 1
        return k

    @staticmethod
    def _extract_sitter_metadata(sitter_docs: List[dict[str, Any]]) -> dict[str, Any]:
        """
        Extract optional structured metadata from ranked sitter docs.
        Metadata may be dict or JSON string; keys may be missing.
        """
        for d in sitter_docs:
            raw_meta = d.get("metadata")
            meta: dict[str, Any] | None = None
            if isinstance(raw_meta, dict):
                meta = raw_meta
            elif isinstance(raw_meta, str):
                try:
                    parsed = json.loads(raw_meta)
                    if isinstance(parsed, dict):
                        meta = parsed
                except json.JSONDecodeError:
                    meta = None

            if not meta:
                continue

            out: dict[str, Any] = {}
            province_id = _safe_int(meta.get("provinceId"))
            if province_id is not None:
                province_name = get_province_name_by_id().get(province_id)
                if province_name:
                    out["provinceName"] = province_name

            district_id = _safe_int(meta.get("districtId"))
            if district_id is not None:
                district_name = get_district_name_by_id().get(district_id)
                if district_name:
                    out["districtName"] = district_name

            raw_pet_type_ids = meta.get("petTypeIds")
            pet_type_ids: list[int] = []
            if isinstance(raw_pet_type_ids, list):
                pet_type_ids = [
                    parsed
                    for parsed in (_safe_int(v) for v in raw_pet_type_ids)
                    if parsed is not None
                ]
            else:
                parsed_single = _safe_int(raw_pet_type_ids)
                if parsed_single is not None:
                    pet_type_ids = [parsed_single]

            if pet_type_ids:
                pet_type_lookup = get_pet_type_name_by_id()
                pet_type_names = [
                    pet_type_lookup[pid]
                    for pid in pet_type_ids
                    if pid in pet_type_lookup
                ]
                if pet_type_names:
                    out["petTypeNames"] = pet_type_names
            if out:
                return out
        return {}

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[Document]:
        effective_top_k = self._effective_top_k()
        query_embedding = get_text_embedding(query)
        docs = await get_similar_rag_documents(
            query_embedding,
            top_k=effective_top_k,
            filters=self.filters,
        )
        top_sitters = rank_sitters(docs, effective_top_k)

        result: List[Document] = []
        for sitter_id, sitter_docs in top_sitters:
            trade_name = ""
            parts: List[str] = [f"SITTER_ID: {sitter_id}"]
            optional_metadata = self._extract_sitter_metadata(sitter_docs)
            for d in sitter_docs:
                trade_name = get_trade_name(d) or trade_name
                parts.append(str(d.get("content", "")))
            if trade_name:
                parts[0] = f"SITTER_ID: {sitter_id}\nTRADE_NAME: {trade_name}"
            page_content = "\n".join(parts)
            result.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "sitter_id": sitter_id,
                        "trade_name": trade_name or f"Sitter {sitter_id}",
                        **optional_metadata,
                    },
                )
            )
        return result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[Document]:
        """Sync wrapper for callers that do not use ainvoke; runs async retrieval."""
        return asyncio.run(
            self._aget_relevant_documents(query, run_manager=run_manager)
        )


def get_sitter_retriever(top_k: Optional[int] = None) -> SitterRankingRetriever:
    """Factory: return a SitterRankingRetriever with optional top_k."""
    k = top_k if top_k is not None else TOP_K_DEFAULT
    return SitterRankingRetriever(top_k=k)
