# Developer README Design

## Goal

Rewrite the project README as an accurate developer guide for local setup, operation, and maintenance after the LangChain and LangGraph migration.

## Audience

Developers who need to run, extend, test, or debug the application locally.

## Content Structure

1. Project purpose and medical-use limitation.
2. Prerequisites, dependency installation, environment configuration, and one-command or manual startup.
3. Runtime architecture covering the FastAPI API, Streamlit UI, LangGraph workflow, LangChain DeepSeek client, Chroma retrieval, profile store, and local fallback behavior.
4. Environment-variable reference for model, profile-store, and RAG settings.
5. Knowledge-base build instructions using the existing PDF vectorization script and Chroma persistence directory.
6. REST API examples for chat, profile, and health endpoints.
7. Test commands, including the offline `APP_LLM_ENABLED=false` mode.
8. Repository layout and operational safety boundaries.

## Accuracy Rules

The README must reflect the current implementation only. It will describe LangChain, LangGraph, Chroma, local embedding, SQLite/MySQL profiles, and DeepSeek's OpenAI-compatible API. It will not claim Qdrant, hybrid retrieval, reranking, Railway deployment, or other components that the code does not implement.

## Validation

Review all commands, environment variables, endpoints, and file paths against the repository before publishing. Run a Markdown-oriented text check for placeholder terms and confirm no superseded technologies remain.
