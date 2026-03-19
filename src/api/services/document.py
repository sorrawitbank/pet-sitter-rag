from typing import Any, Dict, List, Literal, Optional

from httpx import Response
from langchain_core.messages import HumanMessage, SystemMessage

from src.gemini.client import create_structured_llm
from src.rag.query_metadata import extract_query_metadata, resolve_extraction
from src.rag.retriever import get_sitter_retriever
from src.rag.schemas import RAGResponseSchema, ResolvedMetadata, QueryMetadataExtraction


# System prompt (same rules as before, no free-form JSON instruction; structured output is enforced by schema)
SYSTEM_PROMPT = (
    "You are an assistant for a pet-sitter platform.\n"
    "- Your primary role is to help users find suitable pet sitters.\n"
    "- If the user asks what you can do (or similar), briefly explain that you are a helper for finding pet sitters.\n"
    "- If the user asks about anything unrelated to finding pet sitters, clearly answer that your only role is to help find pet sitters.\n"
    "\n"
    "CONTEXT AND GROUNDING RULES:\n"
    "- Use only the information provided in the CONTEXT to answer the question.\n"
    "- If the CONTEXT is exactly '(No relevant documents found.)' (meaning there are zero sitters), politely say that you could not find a pet sitter that matches what the user is looking for, and return an empty sitters array.\n"
    "- If there is at least one sitter in the CONTEXT, you MUST NOT say that you could not find a sitter. Instead, you must describe the sitter(s) that appear in the CONTEXT.\n"
    "- If the CONTEXT is partially relevant but still not enough to answer the question confidently, explicitly say that the information is not available based on the provided context, but still describe the sitter(s) that appear in the CONTEXT.\n"
    "- Do not invent facts that are not supported by the CONTEXT.\n"
    "- Always respond in the same language as the user's question.\n"
    "- For each sitter in CONTEXT, you MUST return exactly one sitter object in the sitters array, in the same order as the CONTEXT blocks (first block = first sitter).\n"
    "- For each sitter object, provide a short description (description) and use the TRADE_NAME from CONTEXT for trade_name.\n"
    "- If a CONTEXT block includes location metadata such as PROVINCE or DISTRICT, treat that as explicit location evidence.\n"
    "- Never skip any sitter if there is a CONTEXT block for that sitter.\n"
    "- Set confidence to High, Medium, or Low based on how clearly the CONTEXT supports your answer.\n"
)


async def answer_query_with_rag(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieve context via LangChain SitterRankingRetriever, then generate a structured
    response (introduction, sitters with id/tradeName/description, confidence) using
    ChatGoogleGenerativeAI with_structured_output.
    """

    def _build_metadata_lines(metadata: Dict[str, Any]) -> str:
        """
        Build optional metadata lines for context. Keys may not exist.
        """
        lines: List[str] = []
        province_name = metadata.get("provinceName")
        if isinstance(province_name, str) and province_name.strip():
            lines.append(f"PROVINCE: {province_name.strip()}")

        district_name = metadata.get("districtName")
        if isinstance(district_name, str) and district_name.strip():
            lines.append(f"DISTRICT: {district_name.strip()}")

        pet_type_names = metadata.get("petTypeNames")
        if isinstance(pet_type_names, list):
            cleaned = [str(v).strip() for v in pet_type_names if str(v).strip()]
            if cleaned:
                lines.append(f"PET_TYPES: {', '.join(cleaned)}")

        return "\n".join(lines)

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
        context_blocks: List[str] = []
        for d in lc_docs:
            metadata_lines = _build_metadata_lines(d.metadata)
            if metadata_lines:
                context_blocks.append(f"{d.page_content}\n{metadata_lines}")
            else:
                context_blocks.append(d.page_content)
        context_text = "\n\n---\n\n".join(context_blocks)
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
        description = ""
        if i < len(raw_sitters):
            s = raw_sitters[i]
            if isinstance(s, dict):
                description = str(s.get("description", ""))
            else:
                description = str(getattr(s, "description", ""))
        sitters_response.append(
            {
                "sitterId": sitter_id,
                "tradeName": trade_name,
                "description": description,
            }
        )

    return {
        "introduction": introduction,
        "sitters": sitters_response,
        "confidence": confidence,
    }
