from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .agent_runtime import load_project_env
from .models import Profile, SourceRef


DEFAULT_VECTOR_DB_DIR = Path(__file__).resolve().parents[1] / "vector_db"
DEFAULT_VECTOR_COLLECTION = "pdf_knowledge"


class LocalEmbeddingAdapter(Embeddings):
    """Lazily adapts the project's existing local embedding model for LangChain."""

    def __init__(self, model_name: str, cache_dir: Path | str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from .ingestion import LocalEmbeddingClient

            self._client = LocalEmbeddingClient(self.model_name, self.cache_dir)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)


@dataclass
class SessionHistoryStore:
    window_size: int = 5
    _histories: dict[str, InMemoryChatMessageHistory] = field(default_factory=dict)

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        return self._histories.setdefault(session_id, InMemoryChatMessageHistory())

    def get_request_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Return a copy so multi-agent prompts do not persist duplicate turns."""
        return InMemoryChatMessageHistory(messages=list(self.get(session_id).messages))

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        history = self.get(session_id)
        history.add_messages([HumanMessage(content=user_message), AIMessage(content=assistant_message)])
        history.messages = history.messages[-self.window_size * 2 :]


@dataclass
class LangChainRuntime:
    llm: Any | None
    retriever: Any | None
    history_store: SessionHistoryStore = field(default_factory=SessionHistoryStore)
    source_retriever: Any | None = None

    def is_llm_configured(self) -> bool:
        return self.llm is not None

    def is_vector_store_available(self) -> bool:
        if self.source_retriever is not None:
            return bool(self.source_retriever.is_available())
        if self.retriever is None:
            return False
        try:
            return self.retriever.vectorstore._collection.count() > 0
        except Exception:
            return False

    def retrieve(self, query: str) -> list[SourceRef]:
        if self.source_retriever is not None:
            try:
                return list(self.source_retriever.retrieve(query))
            except Exception:
                return []
        if self.retriever is None or not query.strip() or self.llm is None:
            return []
        try:
            documents = self.retriever.invoke(query)
        except Exception:
            return []

        sources: list[SourceRef] = []
        for document in documents:
            metadata = document.metadata or {}
            title = str(metadata.get("pdf_name") or metadata.get("source_path") or "PDF reference")
            source = str(metadata.get("source_path") or DEFAULT_VECTOR_COLLECTION)
            if page := metadata.get("page"):
                source = f"{source}#page={page}"
            sources.append(SourceRef(title=title, excerpt=document.page_content[:300], source=source))
        return sources

    def invoke_agent(
        self,
        *,
        agent: str,
        system_prompt: str,
        session_id: str,
        message: str,
        profile: Profile,
        sources: list[SourceRef],
    ) -> str | None:
        if self.llm is None:
            return None

        context = "\n\n".join(
            f"[{item.title}] {item.excerpt}" for item in sources
        ) or "No retrieved reference material is available."
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "User message:\n{message}\n\n"
                    "Health profile:\n{profile}\n\n"
                    "Retrieved references:\n{context}\n\n"
                    "Reply in cautious, patient-friendly Chinese. Do not give a definitive diagnosis.",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            return str(
                chain.invoke(
                    {
                        "message": message,
                        "profile": profile.to_dict(),
                        "context": context,
                        "agent": agent,
                        "history": self.history_store.get(session_id).messages,
                    }
                )
            ).strip() or None
        except Exception:
            return None


def create_langchain_runtime(
    *,
    vector_db_dir: Path | str | None = None,
    collection_name: str | None = None,
    top_k: int | None = None,
) -> LangChainRuntime:
    load_project_env()
    llm = _create_chat_model()
    retriever = _create_retriever(vector_db_dir, collection_name, top_k)
    return LangChainRuntime(llm=llm, retriever=retriever)


def _create_chat_model() -> Any | None:
    if os.getenv("APP_LLM_ENABLED", "true").strip().lower() in {"0", "false", "no"}:
        return None
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2")),
            timeout=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30")),
        )
    except Exception:
        return None


def _create_retriever(
    vector_db_dir: Path | str | None,
    collection_name: str | None,
    top_k: int | None,
) -> Any | None:
    directory = Path(vector_db_dir or os.getenv("APP_VECTOR_DB_DIR", str(DEFAULT_VECTOR_DB_DIR)))
    if not directory.exists():
        return None
    try:
        from langchain_chroma import Chroma

        embeddings = LocalEmbeddingAdapter(
            os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            os.getenv("LOCAL_EMBEDDING_CACHE_DIR", "models/embedding"),
        )
        vectorstore = Chroma(
            collection_name=collection_name or os.getenv("APP_VECTOR_COLLECTION", DEFAULT_VECTOR_COLLECTION),
            persist_directory=str(directory),
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever(search_kwargs={"k": int(top_k or os.getenv("APP_RAG_TOP_K", "3"))})
    except Exception:
        return None
