"""
LangChain custom retriever: vector search + sitter ranking, returns top-3 sitters as Documents.
"""

import asyncio
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.api.repositories.document import TOP_K_DEFAULT, get_similar_rag_documents
from src.gemini.client import get_text_embedding
from src.rag.ranking import get_trade_name, rank_sitters
from src.rag.schemas import ResolvedMetadata


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
