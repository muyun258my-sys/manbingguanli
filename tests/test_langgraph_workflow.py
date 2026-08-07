from app.langgraph_runtime import LangChainRuntime, SessionHistoryStore
from langchain_core.runnables import RunnableLambda
from app.models import ChatRequest, SourceRef
from app.services import Orchestrator, SafetyGate


class StubRetriever:
    def __init__(self, sources=None):
        self.sources = sources or []
        self.called = False

    def retrieve(self, query):
        self.called = True
        return list(self.sources)

    def is_available(self):
        return True


class EmergencyGate:
    def check(self, text):
        return {"emergency": True, "matched_rule": "test-rule"}


def test_langgraph_emergency_short_circuits_before_retrieval_and_history():
    retriever = StubRetriever()
    runtime = LangChainRuntime(llm=None, retriever=None, source_retriever=retriever)
    orchestrator = Orchestrator(safety_gate=EmergencyGate(), runtime=runtime)

    response = orchestrator.chat(ChatRequest(session_id="emergency", user_id="u1", message="any input"))

    assert response["data"]["emergency"] is True
    assert retriever.called is False
    assert runtime.history_store.get("emergency").messages == []


def test_langgraph_uses_injected_rag_sources_and_persists_one_turn():
    source = SourceRef(title="reference.pdf", excerpt="reference content", source="reference.pdf#page=1")
    retriever = StubRetriever([source])
    runtime = LangChainRuntime(llm=None, retriever=None, source_retriever=retriever)
    orchestrator = Orchestrator(runtime=runtime)

    response = orchestrator.chat(ChatRequest(session_id="offline", user_id="u2", message="hello"))

    assert retriever.called is True
    assert response["data"]["sources"][0]["title"] == "健康信息补充提示"
    assert any(item["title"] == "reference.pdf" for item in response["data"]["sources"])
    assert len(runtime.history_store.get("offline").messages) == 2


def test_langchain_prompt_chain_uses_fake_model_and_persists_one_turn():
    runtime = LangChainRuntime(llm=RunnableLambda(lambda _: "模型回复"), retriever=None)
    orchestrator = Orchestrator(runtime=runtime)

    response = orchestrator.chat(ChatRequest(session_id="model", user_id="u3", message="hello"))

    assert response["data"]["reply"] == "模型回复"
    assert len(runtime.history_store.get("model").messages) == 2


# ── SessionHistoryStore ──────────────────────────────────────────────────────

def test_session_history_store_trims_window():
    store = SessionHistoryStore(window_size=2)
    for i in range(10):
        store.append_turn("s", f"msg{i}", f"reply{i}")
    messages = store.get("s").messages
    assert len(messages) <= 4  # window_size * 2


def test_session_history_store_isolates_sessions():
    store = SessionHistoryStore()
    store.append_turn("s1", "hello", "hi")
    store.append_turn("s2", "world", "there")
    assert store.get("s1").messages != store.get("s2").messages
    assert store.get("s1").messages[0].content == "hello"


def test_get_request_history_returns_copy():
    from langchain_core.messages import HumanMessage

    store = SessionHistoryStore()
    store.append_turn("s", "hello", "hi")
    snapshot = store.get_request_history("s")
    snapshot.add_messages([HumanMessage(content="extra")])
    # 快照的修改不影响原始会话
    assert len(store.get("s").messages) == 2
    assert len(snapshot.messages) == 3


# ── 混合意图：LangGraph 串行 symptom → medication ────────────────────────────

def test_langgraph_mixed_query_runs_symptom_then_medication():
    runtime = LangChainRuntime(llm=None, retriever=None)
    orchestrator = Orchestrator(runtime=runtime)

    response = orchestrator.chat(
        ChatRequest(session_id="mixed-lg", user_id="u-mixed", message="乏力头晕，在吃降压药，有副作用禁忌吗")
    )

    data = response["data"]
    assert data["intent"] == "mixed_query"
    # 两个 fallback agent 的结果被合并：症状评估 + 用药核对
    assert "症状评估" in data["reply"]
    assert "用药核对" in data["reply"]
    assert data["severity"] in ("yellow", "red")
    assert len(data["sources"]) >= 2
