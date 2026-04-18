# Pet Sitter RAG – API Documentation

## Overview

FastAPI service that answers natural-language questions about pet sitters using retrieval-augmented generation (RAG): vector search over `rag_documents`, optional metadata filtering (pet type, province, district), and a structured Gemini response.

**Base information**

- **Base URL:** `http://localhost:8000` (default when running Uvicorn; use whatever host/port you configure)
- **Content-Type (default):** `application/json`
- **Allowed methods:** `GET`, `POST`
- **Interactive docs:** OpenAPI UI at `/docs`, ReDoc at `/redoc` (provided by FastAPI)

---

## Contents

- [Authentication](#authentication)
- [General](#general)
- [Document - `/api/document`](#document---apidocument)

---

## Authentication

This RAG API does **not** use Bearer tokens or API keys in HTTP headers. Access control is expected at the network or gateway layer (for example, only your Pet Sitter server calls this service).

The service reads `DATABASE_URL` and `GEMINI_API_KEY` from the environment at runtime; misconfiguration can result in `500` responses from downstream operations.

---

## General

### Root

| Method | Path      | Description                            |
| ------ | --------- | -------------------------------------- |
| GET    | `/`       | Welcome message string                 |
| GET    | `/health` | Health check with status and timestamp |

### GET /

Returns a plain welcome string.

**Success (200)**

```json
"Welcome to Pet Sitter RAG"
```

---

### GET /health

Returns service health and server time.

**Success (200)**

```json
{
  "status": "OK",
  "timestamp": "2026-04-18T12:34:56.789012"
}
```

The `timestamp` value is an ISO 8601 datetime string produced from the server clock.

---

### Error Response (Generic Shape)

**FastAPI `HTTPException` (for example `500` on `/api/document/query`)**

```json
{
  "detail": "Internal server error"
}
```

**Request body validation (`422` Unprocessable Entity)**

FastAPI returns a `detail` array describing each validation error (field path, message, type). Example shape:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

Common status codes:

- `422` - Request body validation failed (invalid or missing JSON fields)
- `500` - Internal server error (for example unhandled failure in RAG pipeline or database)

Individual endpoints may add more specific rules and messages.

---

## Document - `/api/document`

Structured RAG endpoints: one runs the full pipeline (embed → retrieve → rank → Gemini structured output), the other returns a fixed mock for development or demos.

### POST /api/document/query

Run the full RAG pipeline for a user question.

**Body (JSON)**

| Field | Type    | Required | Constraints                                                                                                                 |
| ----- | ------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| query | string  | yes      | User question in natural language                                                                                           |
| top_k | integer | no       | Number of similar chunks to fetch from the database before ranking sitters. If omitted, the service uses its default (`5`). |

**Success (200)**

```json
{
  "query": "string",
  "introduction": "string",
  "sitters": [
    {
      "sitterId": "string",
      "tradeName": "string",
      "description": "string"
    }
  ],
  "confidence": "High" | "Medium" | "Low"
}
```

- **`introduction`** – Short lead-in in the **same language as** `query` (model instruction).
- **`sitters`** – Up to **three** sitters after ranking; each entry includes `sitterId`, `tradeName`, and a short `description` grounded in retrieved context.
- **`confidence`** – Model-reported certainty: `"High"`, `"Medium"`, or `"Low"`.

**Errors**

- `422` - Invalid body (for example missing `query`, or wrong types)
- `500` - `"Internal server error"` (any unhandled exception in the handler is mapped to this message)

---

### POST /api/document/mock

Return a **fixed** mock response with the same shape as `POST /api/document/query`, without calling the database or Gemini.

**Body (JSON)**

| Field | Type    | Required | Constraints                                                         |
| ----- | ------- | -------- | ------------------------------------------------------------------- |
| query | string  | yes      | Echoed back in the response as `query`                              |
| top_k | integer | no       | Accepted for parity with `/api/document/query`; ignored by the mock |

**Success (200)**

```json
{
  "query": "string",
  "introduction": "I found a few pet sitters in Bangkok who can care for your dog. Here are their details:",
  "sitters": [
    {
      "sitterId": "1",
      "tradeName": "MooMu House!",
      "description": "MooMu House! is run by Bank Sorrawit in Thung Khru, Bangkok. Bank provides a safe and loving environment in a spacious house for cats, dogs, and rabbits, with designated areas for play, relaxation, and sleep."
    },
    {
      "sitterId": "6",
      "tradeName": "City Pet Companion",
      "description": "City Pet Companion is located in a pet-friendly area in Bang Kapi, Bangkok. They offer enough indoor space for animals to move around safely, prioritizing cleanliness and safety for dogs and cats."
    },
    {
      "sitterId": "7",
      "tradeName": "PetSimplified",
      "description": "PetSimplified is located in a quiet residential area in Phra Nakhon, Bangkok. Their space is designed for pet comfort and safety, featuring a secure fenced yard for outdoor play and a cozy indoor area with air conditioning for dogs, cats, birds, and rabbits."
    }
  ],
  "confidence": "High"
}
```

**Errors**

- `422` - Invalid body (for example missing `query`)

---
