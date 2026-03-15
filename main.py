from datetime import datetime

from fastapi import APIRouter, FastAPI


app = FastAPI()

router = APIRouter()


@router.get("/")
async def root():
    return "Welcome to Pet Sitter RAG"


@router.get("/health")
async def health():
    return {"status": "OK", "timestamp": datetime.now()}


app.include_router(router)
