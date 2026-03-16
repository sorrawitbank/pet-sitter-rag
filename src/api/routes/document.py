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
        introduction="นี่คือพี่เลี้ยงสัตว์เลี้ยงบางส่วนที่สามารถดูแลทั้งสุนัขและแมวได้ตามข้อมูลที่มีค่ะ",
        sitters=[
            {
                "tradeName": "City Pet Companion",
                "url": "https://pet-sitter-app-two.vercel.app/petsitter/6",
                "description": "มีบริการดูแลทั้งสุนัขและแมว โดยมีตารางการให้อาหาร การเล่น และการเดินสำหรับสุนัข ส่วนแมวจะได้รับของเล่นเสริมสร้างพัฒนาการและพื้นที่ผ่อนคลาย",
            },
            {
                "tradeName": "Green Garden Pet Care",
                "url": "https://pet-sitter-app-two.vercel.app/petsitter/3",
                "description": "คุณพลอยรักการใช้เวลากับสัตว์เลี้ยงและมุ่งมั่นที่จะมอบสภาพแวดล้อมที่อบอุ่นและเอาใจใส่",
            },
            {
                "tradeName": "Happy House!",
                "url": "https://pet-sitter-app-two.vercel.app/petsitter/1",
                "description": "คุณ Jane Maison เป็นพี่เลี้ยงสัตว์เลี้ยงที่เชื่อถือได้ในเสนานิคม กรุงเทพฯ และมีบ้านที่กว้างขวางเพื่อมอบสภาพแวดล้อมที่ปลอดภัยและเป็นที่รักสำหรับแมว สุนัข และกระต่าย",
            },
        ],
        confidence="Medium",
    )
