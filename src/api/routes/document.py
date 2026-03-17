from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.controllers.document import handle_document_query


class DocumentQueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class SitterItem(BaseModel):
    tradeName: str
    url: str
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
    top 3 sitters (tradeName, url, description), and confidence.
    """
    result = await handle_document_query(query=body.query, top_k=body.top_k)
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
                "tradeName": "City Pet Companion",
                "url": "https://pet-sitter-app-two.vercel.app/petsitter/6",
                "description": "Provides care for both dogs and cats, with scheduled feeding, playtime, and walks for dogs, while cats receive developmental toys and a dedicated relaxing space.",
            },
            {
                "tradeName": "Green Garden Pet Care",
                "url": "https://pet-sitter-app-two.vercel.app/petsitter/3",
                "description": "Ploy loves spending time with pets and is dedicated to providing a warm and attentive environment.",
            },
            {
                "tradeName": "Happy House!",
                "url": "https://pet-sitter-app-two.vercel.app/petsitter/1",
                "description": "Jane Maison is a trusted pet sitter in Sena Nikhom, Bangkok, with a spacious home that offers a safe and loving environment for cats, dogs, and rabbits.",
            },
        ],
        confidence="Medium",
    )
