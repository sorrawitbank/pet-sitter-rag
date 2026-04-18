from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.controllers.document import handle_document_query


class DocumentQueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class SitterItem(BaseModel):
    sitterId: str
    tradeName: str
    description: str


class DocumentQueryResponse(BaseModel):
    query: str
    introduction: str
    sitters: list[SitterItem]
    confidence: Literal["High", "Medium", "Low"]


router = APIRouter(prefix="/api/document", tags=["document"])


@router.post("/query", response_model=DocumentQueryResponse)
async def query_document(body: DocumentQueryRequest) -> DocumentQueryResponse:
    """
    Receive a query and return a structured RAG response: introduction,
    top 3 sitters (sitterId, tradeName, description), and confidence.
    """
    try:
        result = await handle_document_query(query=body.query, top_k=body.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return DocumentQueryResponse(
        query=result["query"],
        introduction=result["introduction"],
        sitters=result["sitters"],
        confidence=result["confidence"],
    )


@router.post("/mock", response_model=DocumentQueryResponse)
def mock_document(body: DocumentQueryRequest) -> DocumentQueryResponse:
    """
    Mock the document query response.
    """
    return DocumentQueryResponse(
        query=body.query,
        introduction=(
            "I found a few pet sitters in Bangkok who can care for your dog. "
            "Here are their details:"
        ),
        sitters=[
            {
                "sitterId": "1",
                "tradeName": "MooMu House!",
                "description": (
                    "MooMu House! is run by Bank Sorrawit in Thung Khru, Bangkok. "
                    "Bank provides a safe and loving environment in a spacious house for "
                    "cats, dogs, and rabbits, with designated areas for play, relaxation, and sleep."
                ),
            },
            {
                "sitterId": "6",
                "tradeName": "City Pet Companion",
                "description": (
                    "City Pet Companion is located in a pet-friendly area in Bang Kapi, Bangkok. "
                    "They offer enough indoor space for animals to move around safely, prioritizing "
                    "cleanliness and safety for dogs and cats."
                ),
            },
            {
                "sitterId": "7",
                "tradeName": "PetSimplified",
                "description": (
                    "PetSimplified is located in a quiet residential area in Phra Nakhon, Bangkok. "
                    "Their space is designed for pet comfort and safety, featuring a secure fenced yard "
                    "for outdoor play and a cozy indoor area with air conditioning for dogs, cats, "
                    "birds, and rabbits."
                ),
            },
        ],
        confidence="High",
    )
