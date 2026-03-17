from typing import Any, Dict, List, Optional

from src.api.services.document import answer_query_with_rag


async def handle_document_query(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Orchestrate answering a user query using RAG and return structured response:
    introduction, sitters (sitterId, tradeName, description), confidence.
    """
    result = await answer_query_with_rag(query, top_k=top_k)
    return {
        "query": query,
        "introduction": result["introduction"],
        "sitters": result["sitters"],
        "confidence": result["confidence"],
    }
