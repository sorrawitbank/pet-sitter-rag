# Pet Sitter RAG

RAG service for the Pet Sitter platform. This FastAPI application retrieves relevant sitter context from PostgreSQL and generates structured recommendations with Gemini.

<p align="center">
    <a href="https://github.com/visneeb/pet-sitter-app" alt="Pet Sitter App">
        <img src="https://img.shields.io/badge/github-pet--sitter--app-ff7037?logo=github&logoColor=white&style=flat-square" /></a>
    <a href="https://github.com/sorrawitbank/pet-sitter-server" alt="Pet Sitter Server">
        <img src="https://img.shields.io/badge/github-pet--sitter--server-1ccd83?logo=github&logoColor=white&style=flat-square" /></a>
    <a href="https://github.com/sorrawitbank/pet-sitter-rag" alt="Pet Sitter RAG">
        <img src="https://img.shields.io/badge/github-pet--sitter--rag-76c0fc?logo=github&logoColor=white&style=flat-square" /></a>
</p>

## Role In Ecosystem

This repository is the recommendation engine in the Pet Sitter system:

- `pet-sitter-app`: frontend application for users.
- `pet-sitter-server`: main backend API and business logic.
- `pet-sitter-rag` (this repo): retrieval and generation service used by chatbot/recommendation flows.

## Tech Stack

- Python
- FastAPI
- PostgreSQL (`asyncpg`)
- Google Gemini (`google-genai`, `langchain-google-genai`)

## Prerequisites

- Python 3.11+ (recommended)
- PostgreSQL database
- Gemini API key

## Project Structure

- `main.py` - FastAPI app entry point (`/`, `/health`, and router wiring)
- `src/api` - API routes, controllers, repositories, and services
- `src/rag` - retrieval, ranking, metadata extraction, and schemas
- `src/db` - database connection setup
- `src/gemini` - Gemini client integration
- `docs/api/README.md` - full API contract

## Getting Started

### 1) Clone and install dependencies

```bash
git clone <your-repo-url>
cd pet-sitter-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment variables

Copy `.env.example` to `.env` and set values.

Required groups include:

- Database: `DATABASE_URL`
- AI: `GEMINI_API_KEY`

### 3) Run the service

```c
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Service defaults:

- Base URL: `http://localhost:8000`
- OpenAPI UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Overview

Key endpoints:

- `GET /` - welcome message
- `GET /health` - health status and server timestamp
- `POST /api/documents/query` - run full RAG pipeline
- `POST /api/documents/mock` - return fixed mock response

For full request/response schema and error cases, see:

- `docs/api/README.md`

## Notes

- This service does not use bearer-token auth at route level by default.
- Protect access at network or gateway level (for example, allow internal calls from `pet-sitter-server` only).
- Incorrect `DATABASE_URL` or `GEMINI_API_KEY` can cause downstream `500` errors.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
