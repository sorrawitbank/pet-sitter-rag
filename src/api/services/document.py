from typing import Any, Dict, List, Literal, Optional

from httpx import Response
from langchain_core.messages import HumanMessage, SystemMessage

from src.gemini.client import create_structured_llm
from src.rag.query_metadata import extract_query_metadata, resolve_extraction
from src.rag.retriever import get_sitter_retriever
from src.rag.schemas import RAGResponseSchema, ResolvedMetadata, QueryMetadataExtraction

BASE_SITTER_URL = "https://pet-sitter-app-two.vercel.app/petsitter"

# System prompt (same rules as before, no free-form JSON instruction; structured output is enforced by schema)
SYSTEM_PROMPT = (
    "You are an assistant for a pet-sitter platform.\n"
    "- Your primary role is to help users find suitable pet sitters.\n"
    "- If the user asks what you can do (or similar), briefly explain that you are a helper for finding pet sitters.\n"
    "- If the user asks about anything unrelated to finding pet sitters, clearly answer that your only role is to help find pet sitters.\n"
    "\n"
    "CONTEXT AND GROUNDING RULES:\n"
    "- Use only the information provided in the CONTEXT to answer the question.\n"
    "- If the CONTEXT does not contain any suitable sitter for the user's request (for example, CONTEXT is '(No relevant documents found.)' or there are zero sitters), politely say that you could not find a pet sitter that matches what the user is looking for, and return an empty sitters array.\n"
    "- If the CONTEXT is partially relevant but still not enough to answer the question confidently, explicitly say that the information is not available based on the provided context.\n"
    "- Do not invent facts that are not supported by the CONTEXT.\n"
    "- Always respond in the same language as the user's question.\n"
    "- For each sitter in CONTEXT, you MUST return exactly one sitter object in the sitters array, in the same order as the CONTEXT blocks (first block = first sitter).\n"
    "- For each sitter object, provide a short description (description) and use the TRADE_NAME from CONTEXT for trade_name.\n"
    "- Never skip any sitter if there is a CONTEXT block for that sitter.\n"
    "- Set confidence to High, Medium, or Low based on how clearly the CONTEXT supports your answer.\n"
)


async def answer_query_with_rag(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieve context via LangChain SitterRankingRetriever, then generate a structured
    response (introduction, sitters with tradeName/url/description, confidence) using
    ChatGoogleGenerativeAI with_structured_output.
    """
    extraction: QueryMetadataExtraction = await extract_query_metadata(
        query,
        include_canonical_in_prompt=True,
    )

    resolved: ResolvedMetadata = resolve_extraction(
        extraction,
        use_fuzzy=True,
        fuzzy_score_cutoff=80,
    )

    retriever = get_sitter_retriever(top_k=top_k)
    retriever.filters = resolved
    lc_docs = await retriever.ainvoke(query)

    if not lc_docs:
        # No documents: still call LLM so the reply uses the same language as the query
        context_text = "(No relevant documents found.)"
        sitter_info = []
    else:
        context_text = "\n\n---\n\n".join(d.page_content for d in lc_docs)
        sitter_info = [
            {
                "sitter_id": d.metadata.get("sitter_id", ""),
                "trade_name": d.metadata.get("trade_name", "")
                or f"Sitter {d.metadata.get('sitter_id', '')}",
            }
            for d in lc_docs
        ]

    user_content = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"

    structured_llm = create_structured_llm(RAGResponseSchema.model_json_schema())
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = structured_llm.invoke(messages)

    # Normalize to dict (LLM may return dict or Pydantic model)
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    elif hasattr(response, "dict"):
        response = response.dict()
    elif not isinstance(response, dict):
        response = {
            "introduction": str(response),
            "sitters": [],
            "confidence": "Medium",
        }

    introduction = str(response.get("introduction", ""))
    raw_sitters = response.get("sitters") or []
    if not isinstance(raw_sitters, list):
        raw_sitters = []
    confidence: Literal["High", "Medium", "Low"] = "Medium"
    c = response.get("confidence")
    if c in ("High", "Medium", "Low"):
        confidence = c

    sitters_response: List[Dict[str, Any]] = []
    for i, info in enumerate(sitter_info):
        trade_name = info.get("trade_name", "")
        sitter_id = info.get("sitter_id", "")
        url = f"{BASE_SITTER_URL}/{sitter_id}" if sitter_id else ""
        description = ""
        if i < len(raw_sitters):
            s = raw_sitters[i]
            if isinstance(s, dict):
                description = str(s.get("description", ""))
            else:
                description = str(getattr(s, "description", ""))
        sitters_response.append(
            {
                "tradeName": trade_name,
                "url": url,
                "description": description,
            }
        )

    return {
        "introduction": introduction,
        "sitters": sitters_response,
        "confidence": confidence,
    }
