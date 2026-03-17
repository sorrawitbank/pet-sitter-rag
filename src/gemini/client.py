import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


_GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")


if not _GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment.")


def _create_client() -> genai.Client:
    """
    Create a Gemini client using GEMINI_API_KEY.
    """
    return genai.Client(api_key=_GEMINI_API_KEY)


def get_text_embedding(text: str) -> List[float]:
    """
    Create a text embedding using the configured Gemini embedding model.
    """
    client = _create_client()
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 1536},
    )
    # embed_content returns an object with an `embeddings` field containing vectors.
    if not getattr(response, "embeddings", None):
        raise RuntimeError("Gemini did not return any embeddings.")
    return response.embeddings[0].values


def create_structured_llm(
    schema: Dict[str, Any], *, model: str = "gemini-2.5-flash", temperature: float = 0.5
):
    """
    Create a LangChain ChatGoogleGenerativeAI bound to structured output (json_schema).
    Caller passes the JSON schema (e.g. from PydanticBaseModel.model_json_schema()).
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        api_key=_GEMINI_API_KEY,
        temperature=temperature,
    )
    return llm.with_structured_output(schema=schema, method="json_schema")
