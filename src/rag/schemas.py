"""
Pydantic schemas for LangChain structured output (LLM response shape).
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query metadata extraction (for pre-filtering documents by pet type, province, district)
# LLM must output English-only values that match canonical names in our CSV reference data.
# ---------------------------------------------------------------------------


class QueryMetadataExtraction(BaseModel):
    """
    Structured extraction from the user query for filtering.
    All string values must be in English and match canonical names
    (e.g. "Dog", "Cat", "Bangkok", "Samut Prakarn", "Bang Bon", "Bang Khun Thian").
    Use empty list when the user did not mention that dimension.
    """

    pet_types: list[str] = Field(
        default_factory=list,
        description="Pet types the user wants (English only). E.g. ['Dog', 'Cat', 'Bird']. Empty if not specified.",
    )
    provinces: list[str] = Field(
        default_factory=list,
        description="Provinces the user mentioned (English only). E.g. ['Bangkok', 'Samut Prakarn']. Empty if not specified.",
    )
    districts: list[str] = Field(
        default_factory=list,
        description="Districts the user mentioned (English only). E.g. ['Bang Bon', 'Bang Khun Thian']. Empty if not specified.",
    )


class ResolvedMetadata(BaseModel):
    """
    Query metadata after resolving English names to canonical IDs
    (for use in WHERE metadata.pet_type_id IN (...), etc.).
    """

    pet_type_ids: list[int] = Field(default_factory=list)
    province_ids: list[int] = Field(default_factory=list)
    district_ids: list[int] = Field(default_factory=list)


class SitterDescription(BaseModel):
    """One sitter entry in the LLM response."""

    trade_name: str = Field(
        description="Display name / trade name of the pet sitter from CONTEXT"
    )
    description: str = Field(
        description="Short answer or description for this pet sitter based on CONTEXT, in the same language as the user question"
    )


class RAGResponseSchema(BaseModel):
    """Structured response from the RAG LLM: introduction, sitters, confidence."""

    introduction: str = Field(
        description="A short lead-in sentence (one or two sentences) before describing each sitter, in the same language as the user question"
    )
    sitters: list[SitterDescription] = Field(
        description="Array of sitter descriptions in the same order as in the CONTEXT (first block = first sitter)"
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="Confidence level based on how clearly the CONTEXT supports the answer"
    )
