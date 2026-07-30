# LangChain and LangGraph Migration Design

## Goal

Replace the application's custom LLM client, PDF RAG retrieval, in-memory conversation handling, and agent orchestration with LangChain and LangGraph. Preserve the existing FastAPI and Streamlit contracts, profile persistence, emergency safety behavior, and offline local-response fallback.

## Scope

The migration covers the service layer behind `Orchestrator.chat()`. Public API paths, request/response payloads, Streamlit behavior, SQLite/MySQL profile stores, and the current emergency keyword rules remain compatible.

The migration does not add new user-facing endpoints, change the medical policy, or require a live model/vector database for tests.

## Runtime Architecture

### LangChain Runtime

`app/langgraph_runtime.py` will construct and expose the shared integrations:

- `ChatOpenAI` configured for the DeepSeek OpenAI-compatible endpoint and `DEEPSEEK_API_KEY`.
- A Chroma-backed retriever over the existing persistent `vector_db` collection.
- Prompt templates for intent classification, symptoms, medication, diagnosis, diet, general health, and response aggregation.
- LangChain message-history storage keyed by `session_id`, retaining the five most recent turns, with LangGraph `MemorySaver` checkpointing workflow state.

The runtime must be dependency-injectable so unit tests can use fake chat models, retrievers, and histories without external services.

### LangGraph Workflow

`app/langgraph_workflow.py` will define a `StateGraph` over a typed `ChatState`. State contains the request identifiers, input, profile, message history, intent, retrieved documents, individual agent outputs, sources, severity, emergency flag, and final response.

The graph proceeds as follows:

1. Run the deterministic emergency safety check.
2. If it matches, branch directly to the emergency response node and end.
3. Load the user profile and recent session history.
4. Classify intent and retrieve relevant PDF chunks when the vector store is available.
5. Route to one professional-agent node or multiple nodes for mixed requests.
6. Aggregate responses, taking the highest severity and the most conservative recommendation when outputs conflict.
7. Persist user and assistant messages for successful non-emergency replies, then generate the existing response envelope.

Professional-agent nodes use `ChatPromptTemplate | ChatOpenAI | StrOutputParser` with the stored LangChain history supplied as prompt messages. Retrieved documents are rendered into their prompt context. The response format remains compatible with the existing API: reply, intent, severity, emergency flag, sources, and disclaimer.

## Fallback and Error Handling

- Emergency keyword detection remains deterministic and precedes all model and retriever calls.
- Missing `DEEPSEEK_API_KEY`, LLM timeouts, malformed model output, or model-provider errors route the affected node to its current local rule-based reply.
- Missing, unbuilt, or failing Chroma storage creates an empty RAG context; it does not prevent a reply.
- Internal dependency exceptions are handled in the workflow and never exposed in API output.
- A successful ordinary reply updates the session history. Emergency and failed workflow requests do not write partial history.

## Service Layer Changes

`app/services.py` will retain profile persistence, the deterministic safety rules, the local fallback agents, and a slim `Orchestrator` facade. It will delegate all normal request orchestration to the compiled LangGraph workflow.

The old direct HTTP LLM client and custom retrieval/memory/orchestration code will be removed once the LangChain runtime replaces each use. `requirements.txt` will add LangChain, LangGraph, LangChain OpenAI, and LangChain Chroma integration dependencies. Unused direct-LLM dependencies will be removed only when no longer referenced.

## Tests

Existing tests for API contracts, profiles, emergency behavior, intent routing, and local replies remain. New isolated tests will verify:

- The graph short-circuits before retrieval and model invocation for emergencies.
- Single and mixed intents select the expected graph branches.
- RAG document context and source metadata reach professional-agent nodes.
- LLM and retriever failures activate the intended fallback behavior.
- Histories are limited to five turns and isolated by session ID.
- Aggregation preserves the strongest severity and conservative recommendation.

Tests use injected fake LangChain chat models and retrievers. No test calls DeepSeek, downloads an embedding model, or requires an existing Chroma collection.

## Acceptance Criteria

- The application uses LangChain for model invocation, prompt composition, retrieval integration, and message history.
- The application uses LangGraph for request routing, multi-agent execution, and emergency short-circuiting.
- Existing HTTP API responses and offline behavior remain compatible.
- Emergency requests do not invoke the LLM or retriever.
- The complete test suite runs without external network access.
