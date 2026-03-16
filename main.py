from datetime import datetime

from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI

from src.api.routes.document import router as document_router
from src.db import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect_db()
    yield
    await db.close_db()


app = FastAPI(lifespan=lifespan)

router = APIRouter()


@router.get("/")
async def root():
    return "Welcome to Pet Sitter RAG"


@router.get("/health")
async def health():
    return {"status": "OK", "timestamp": datetime.now()}


app.include_router(router)
app.include_router(document_router)
