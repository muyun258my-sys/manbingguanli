from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .langgraph_runtime import LangChainRuntime
from .models import AgentOutput, ChatRequest, ChatResponse, Intent, Profile, Severity, SourceRef


class ChatState(TypedDict, total=False):
    request: ChatRequest
    profile: Profile
    safety: dict[str, Any]
    intent: Intent
    is_mixed: bool
    sources: list[SourceRef]
    agent_outputs: list[AgentOutput]
    response: ChatResponse


AGENT_PROMPTS = {
    "symptom_analysis": (
        "You are the Symptom Analysis Agent for a chronic disease health management assistant. "
        "Assess possible risk factors, identify missing information, and recommend offline care when symptoms persist or worsen."
    ),
    "medication_query": (
        "You are the Medication Management Agent. Check allergies, current medications, contraindications, interactions, and dosage cautions. "
        "Never tell a user to start, stop, combine, or change prescription medication without clinician confirmation."
    ),
    "diagnosis_query": (
        "You are the Diagnosis Guidance Agent. Give conservative triage and practical department guidance without diagnosing. "
        "When uncertain, recommend the safer next step."
    ),
    "diet_query": (
        "You are the Diet Guidance Agent for chronic disease management. Provide cautious food and portion guidance while considering the profile and medicines."
    ),
    "general_health": (
        "You are a chronic disease health assistant. Ask for the missing symptoms, duration, medicines, and allergies needed for safe next steps."
    ),
}


class LangGraphWorkflow:
    def __init__(
        self,
        *,
        runtime: LangChainRuntime,
        safety_gate: Any,
        classifier: Any,
        profile_store: Any,
        fallback_agents: dict[str, Any],
        merge_results: Callable[[list[AgentOutput]], AgentOutput],
        profile_prompt_renderer: Callable[[Profile], str],
    ) -> None:
        self.runtime = runtime
        self.safety_gate = safety_gate
        self.classifier = classifier
        self.profile_store = profile_store
        self.fallback_agents = fallback_agents
        self.merge_results = merge_results
        self.profile_prompt_renderer = profile_prompt_renderer
        self.graph = self._build()

    def invoke(self, request: ChatRequest) -> ChatResponse:
        result = self.graph.invoke(
            {"request": request, "agent_outputs": [], "sources": []},
            config={"configurable": {"thread_id": request.session_id}},
        )
        return result["response"]

    def _build(self) -> Any:
        graph = StateGraph(ChatState)
        graph.add_node("safety", self._safety)
        graph.add_node("emergency", self._emergency)
        graph.add_node("prepare", self._prepare)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("symptom", lambda state: self._run_agent(state, "symptom_analysis"))
        graph.add_node("medication", lambda state: self._run_agent(state, "medication_query"))
        graph.add_node("diagnosis", lambda state: self._run_agent(state, "diagnosis_query"))
        graph.add_node("diet", lambda state: self._run_agent(state, "diet_query"))
        graph.add_node("general", lambda state: self._run_agent(state, "general_health"))
        graph.add_node("aggregate", self._aggregate)

        graph.add_edge(START, "safety")
        graph.add_conditional_edges("safety", self._after_safety, {"emergency": "emergency", "prepare": "prepare"})
        graph.add_edge("emergency", END)
        graph.add_edge("prepare", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route,
            {
                "symptom": "symptom",
                "medication": "medication",
                "diagnosis": "diagnosis",
                "diet": "diet",
                "general": "general",
            },
        )
        graph.add_conditional_edges("symptom", self._after_symptom, {"medication": "medication", "aggregate": "aggregate"})
        graph.add_edge("medication", "aggregate")
        graph.add_edge("diagnosis", "aggregate")
        graph.add_edge("diet", "aggregate")
        graph.add_edge("general", "aggregate")
        graph.add_edge("aggregate", END)
        return graph.compile(checkpointer=MemorySaver())

    def _safety(self, state: ChatState) -> ChatState:
        return {"safety": self.safety_gate.check(state["request"].message)}

    @staticmethod
    def _after_safety(state: ChatState) -> str:
        return "emergency" if state["safety"]["emergency"] else "prepare"

    def _emergency(self, state: ChatState) -> ChatState:
        request = state["request"]
        response = ChatResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            message=request.message,
            reply="当前输入触发了高风险安全规则。请立即停止自我处理，尽快联系当地急救或前往最近急诊。",
            intent="high_risk_input",
            severity="red",
            emergency=True,
            sources=[SourceRef(title="安全网关", excerpt=state["safety"]["matched_rule"], source="local_rule")],
        )
        return {"response": response}

    def _prepare(self, state: ChatState) -> ChatState:
        request = state["request"]
        return {
            "profile": self.profile_store.get(request.user_id),
            "intent": self.classifier.classify(request.message),
            "is_mixed": self.classifier.classify(request.message) == "mixed_query",
        }

    def _retrieve(self, state: ChatState) -> ChatState:
        return {"sources": self.runtime.retrieve(state["request"].message)}

    def _route(self, state: ChatState) -> str:
        intent = state["intent"]
        if intent == "mixed_query":
            return "symptom"
        return {
            "symptom_analysis": "symptom",
            "medication_query": "medication",
            "diagnosis_query": "diagnosis",
            "diet_query": "diet",
        }.get(intent, "general")

    def _run_agent(self, state: ChatState, agent_name: str) -> ChatState:
        request = state["request"]
        profile = state["profile"]
        sources = state.get("sources", [])
        content = self.runtime.invoke_agent(
            agent=agent_name,
            system_prompt=AGENT_PROMPTS[agent_name],
            session_id=request.session_id,
            message=request.message,
            profile=profile,
            sources=sources,
        )
        if content is None:
            output = self.fallback_agents[agent_name].run(
                request,
                profile,
                [{"role": "system", "content": self.profile_prompt_renderer(profile)}],
                sources,
            )
        else:
            output = AgentOutput(
                agent=agent_name,
                content=content,
                severity=self._severity_for(agent_name),
                sources=sources or [SourceRef(title=agent_name, excerpt=request.message[:80], source="deepseek")],
                confidence="medium",
            )
        outputs = [*state.get("agent_outputs", []), output]
        return {"agent_outputs": outputs}

    @staticmethod
    def _after_symptom(state: ChatState) -> str:
        return "medication" if state.get("is_mixed") else "aggregate"

    @staticmethod
    def _severity_for(agent_name: str) -> Severity:
        return {"symptom_analysis": "yellow", "medication_query": "yellow", "diagnosis_query": "red", "diet_query": "green"}.get(agent_name, "green")  # type: ignore[return-value]

    def _aggregate(self, state: ChatState) -> ChatState:
        request = state["request"]
        output = self.merge_results(state["agent_outputs"])
        response = ChatResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            message=request.message,
            reply=output.content,
            intent="mixed_query" if len(state["agent_outputs"]) > 1 else state["intent"],
            severity=output.severity,
            emergency=False,
            sources=output.sources,
        )
        self.runtime.history_store.append_turn(request.session_id, request.message, response.reply)
        return {"response": response}
