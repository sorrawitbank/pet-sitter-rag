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


router = APIRouter(prefix="/document", tags=["document"])


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
        introduction="Here are some pet sitters who can take care of both dogs and cats based on the available information.",
        sitters=[
            {
                "sitterId": "6",
                "tradeName": "City Pet Companion",
                "description": "Provides care for both dogs and cats, with scheduled feeding, playtime, and walks for dogs, while cats receive developmental toys and a dedicated relaxing space.",
            },
            {
                "sitterId": "3",
                "tradeName": "Green Garden Pet Care",
                "description": "Ploy loves spending time with pets and is dedicated to providing a warm and attentive environment.",
            },
            {
                "sitterId": "1",
                "tradeName": "Happy House!",
                "description": "Jane Maison is a trusted pet sitter in Sena Nikhom, Bangkok, with a spacious home that offers a safe and loving environment for cats, dogs, and rabbits.",
            },
        ],
        confidence="Medium",
    )
