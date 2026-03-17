from typing import Any, Dict, List, Optional

from src.db.db import get_pool
from src.rag.schemas import ResolvedMetadata


TOP_K_DEFAULT = 5


async def get_similar_rag_documents(
    query_embedding: List[float],
    top_k: int = TOP_K_DEFAULT,
    filters: Optional[ResolvedMetadata] = None,
) -> List[Dict[str, Any]]:
    """
    Query rag_documents table by vector similarity using pgvector.
    Optionally filter by metadata (petTypeIds, provinceId, districtId).
    """
    base_sql = """
        SELECT
            rag_document_id,
            source_table,
            source_id,
            content,
            metadata,
            embedding <-> $1::vector AS distance
        FROM rag_documents
    """

    where_clauses = []
    params: List[Any] = []
    param_index = 3  # $1 = embedding, $2 = top_k

    if filters:
        if filters.pet_type_ids:
            # metadata.petTypeIds
            where_clauses.append(
                f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'petTypeIds') AS x(val) "
                f"WHERE (x.val)::int = ANY(${param_index}::int[]))"
            )
            params.append(filters.pet_type_ids)
            param_index += 1

        if filters.province_ids:
            # provinceId
            where_clauses.append(
                f"(metadata->>'provinceId')::int = ANY(${param_index}::int[])"
            )
            params.append(filters.province_ids)
            param_index += 1

        if filters.district_ids:
            # districtId
            where_clauses.append(
                f"(metadata->>'districtId')::int = ANY(${param_index}::int[])"
            )
            params.append(filters.district_ids)
            param_index += 1

    sql = base_sql
    if where_clauses:
        sql += "\nWHERE " + " AND ".join(where_clauses)

    sql += """
        ORDER BY embedding <-> $1::vector
        LIMIT $2;
    """

    pool = get_pool()
    async with pool.acquire() as connection:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        rows = await connection.fetch(
            sql,
            embedding_str,
            top_k,
            *params,
        )
        return [dict(row) for row in rows]
