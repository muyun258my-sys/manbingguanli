from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from .agent_runtime import AgentConfig, AgentConfigError, AgentConfigStore, OpenAICompatibleChatClient
from .models import (
    AgentOutput,
    ChatRequest,
    ChatResponse,
    Confidence,
    Intent,
    Profile,
    Severity,
    SourceRef,
    envelope,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PROFILE_DB = _DATA_DIR / "profiles.db"


def _resolve_db_path(db_path: Optional[Path | str]) -> Path:
    if db_path is not None:
        return Path(db_path)
    env_path = os.getenv("APP_PROFILE_DB")
    if env_path:
        return Path(env_path)
    return DEFAULT_PROFILE_DB


class ProfileStore:
    """用户健康档案存储，落盘到 SQLite，重启后数据仍在。"""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = _resolve_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    condition_description TEXT NOT NULL DEFAULT '',
                    conditions TEXT NOT NULL DEFAULT '[]',
                    medications TEXT NOT NULL DEFAULT '[]',
                    allergies TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def get(self, user_id: str) -> Profile:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return Profile(user_id=user_id)
        return Profile(
            user_id=row["user_id"],
            condition_description=row["condition_description"],
            conditions=json.loads(row["conditions"]),
            medications=json.loads(row["medications"]),
            allergies=json.loads(row["allergies"]),
            updated_at=row["updated_at"],
        )

    def update(
        self,
        user_id: str,
        *,
        condition_description: Optional[str] = None,
        conditions: Optional[Sequence[str]] = None,
        medications: Optional[Sequence[str]] = None,
        allergies: Optional[Sequence[str]] = None,
    ) -> Profile:
        profile = self.get(user_id)
        if condition_description is not None:
            profile.condition_description = condition_description.strip()
        if conditions is not None:
            profile.conditions = list(conditions)
        if medications is not None:
            profile.medications = list(medications)
        if allergies is not None:
            profile.allergies = list(allergies)
        profile.updated_at = now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profiles
                    (user_id, condition_description, conditions, medications, allergies, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    condition_description = excluded.condition_description,
                    conditions = excluded.conditions,
                    medications = excluded.medications,
                    allergies = excluded.allergies,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    profile.condition_description,
                    json.dumps(profile.conditions, ensure_ascii=False),
                    json.dumps(profile.medications, ensure_ascii=False),
                    json.dumps(profile.allergies, ensure_ascii=False),
                    profile.updated_at,
                ),
            )
        return profile


def render_profile_prompt(profile: Profile) -> str:
    lines = ["用户健康档案上下文："]
    has_context = False

    if profile.condition_description:
        lines.append(f"- 当前病情描述：{profile.condition_description}")
        has_context = True
    if profile.conditions:
        lines.append(f"- 已知病史：{'、'.join(profile.conditions)}")
        has_context = True
    if profile.medications:
        lines.append(f"- 当前用药：{'、'.join(profile.medications)}")
        has_context = True
    if profile.allergies:
        lines.append(f"- 过敏史：{'、'.join(profile.allergies)}")
        has_context = True

    if not has_context:
        lines.append("- 暂无已保存的健康档案。")
    lines.append("请在每轮回答中参考以上信息；如涉及用药或风险判断，优先核对过敏史、当前用药和既往病史。")
    return "\n".join(lines)


class ConversationMemory:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._history: Dict[str, List[Dict[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        items = self._history.setdefault(session_id, [])
        items.append({"role": role, "content": content})
        if len(items) > self.window_size * 2:
            del items[: len(items) - self.window_size * 2]

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._history.get(session_id, [])[-self.window_size * 2 :])


class SafetyGate:
    # 正则匹配，容忍口语化表达与中间插入的修饰字
    # （如“胸口剧痛”“左臂也麻了”），避免精确连续匹配漏判。
    EMERGENCY_PATTERNS = [
        r"胸(口|部|前区)?(剧烈|剧|隐|闷|绞|刺)?(痛|疼)",        # 胸痛/胸口剧痛/胸口疼
        r"呼吸(困难|急促|费力)|喘不(过|上)气|憋气",            # 呼吸困难/呼吸急促
        r"昏迷|昏厥|晕厥|不省人事",                            # 昏迷/晕厥
        r"意识(不清|模糊|丧失|障碍)|叫不醒",                   # 意识不清/意识模糊
        r"(左|右|单|双)?(臂|手臂|胳膊|肢体?)[^，。,.\s]{0,3}麻",  # 左臂麻/左臂也麻了/手臂发麻
        r"(一侧|单侧|半边|左侧|右侧)(肢体|手脚|身体|手|脚|腿|胳膊)?(无力|不能动)|偏瘫|肢体无力",  # 一侧肢体无力
        r"口角?(歪斜|歪)|嘴(歪|角歪)",                         # 口角歪斜/嘴歪
        r"(剧烈|严重|爆炸样|最剧烈)头痛|头痛欲裂",             # 剧烈头痛
        r"大(量)?出血|血流不止|咯血|呕血",                     # 大出血/咯血
        r"抽搐|惊厥|痉挛",                                     # 抽搐
    ]

    def check(self, text: str) -> Dict[str, Any]:
        lowered = text.strip()
        for pattern in self.EMERGENCY_PATTERNS:
            if re.search(pattern, lowered):
                return {"emergency": True, "matched_rule": pattern}
        return {"emergency": False, "matched_rule": ""}


class IntentClassifier:
    MEDICATION_WORDS = ["药", "服用", "副作用", "禁忌", "过敏", "相互作用", "能一起吃", "能不能吃"]
    DIAGNOSIS_WORDS = ["挂什么科", "要不要去医院", "严重吗", "怎么诊断", "是不是", "建议就医"]
    PROFILE_WORDS = ["更新档案", "保存档案", "修改档案", "我的病史", "我的过敏", "我的用药"]
    SYMPTOM_WORDS = ["头晕", "发热", "发烧", "咳嗽", "头痛", "恶心", "腹痛", "胸闷", "心慌", "乏力"]

    def classify(self, text: str, emergency: bool = False) -> Intent:
        if emergency:
            return "high_risk_input"

        has_medication = self._contains_any(text, self.MEDICATION_WORDS)
        has_diagnosis = self._contains_any(text, self.DIAGNOSIS_WORDS)
        has_profile = self._contains_any(text, self.PROFILE_WORDS)
        has_symptom = self._contains_any(text, self.SYMPTOM_WORDS)

        score = sum([has_medication, has_diagnosis, has_profile, has_symptom])
        if score > 1:
            return "mixed_query"
        if has_profile:
            return "profile_management"
        if has_medication:
            return "medication_query"
        if has_diagnosis:
            return "diagnosis_query"
        if has_symptom:
            return "symptom_analysis"
        return "general_health"

    @staticmethod
    def _contains_any(text: str, words: Sequence[str]) -> bool:
        return any(word in text for word in words)


def _profile_snapshot(profile: Profile) -> Dict[str, Any]:
    return {
        "user_id": profile.user_id,
        "condition_description": profile.condition_description,
        "conditions": profile.conditions,
        "medications": profile.medications,
        "allergies": profile.allergies,
        "updated_at": profile.updated_at,
    }


class ConfiguredLLMAgent:
    def __init__(
        self,
        config: AgentConfig,
        client: OpenAICompatibleChatClient,
        *,
        output_agent: str,
        fallback_severity: Severity,
        fallback_confidence: Confidence,
    ) -> None:
        self.config = config
        self.client = client
        self.output_agent = output_agent
        self.fallback_severity = fallback_severity
        self.fallback_confidence = fallback_confidence

    def run_llm(
        self,
        request: ChatRequest,
        profile: Profile,
        memory: List[Dict[str, str]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentOutput]:
        if not self.client.is_configured(self.config):
            return None

        user_message = {
            "role": "user",
            "content": (
                "User message:\n"
                f"{request.message}\n\n"
                "User profile JSON:\n"
                f"{_profile_snapshot(profile)}"
            ),
        }
        try:
            content = self.client.complete(
                self.config,
                messages=[*memory, user_message],
                metadata=metadata,
            )
        except (AgentConfigError, requests.RequestException, ValueError):
            return None

        return AgentOutput(
            agent=self.output_agent,
            content=content,
            severity=self.fallback_severity,
            sources=[SourceRef(title=self.config.name, excerpt=request.message[:80], source="deepseek")],
            confidence=self.fallback_confidence,
        )


class SymptomAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        client: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.llm_agent = (
            ConfiguredLLMAgent(
                config,
                client or OpenAICompatibleChatClient(),
                output_agent="symptom_analysis",
                fallback_severity="yellow",
                fallback_confidence="medium",
            )
            if config is not None
            else None
        )

    def run(self, request: ChatRequest, profile: Profile, memory: List[Dict[str, str]]) -> AgentOutput:
        if self.llm_agent is not None:
            llm_output = self.llm_agent.run_llm(
                request,
                profile,
                memory,
                metadata={"agent": "symptom_analysis", "profile": _profile_snapshot(profile)},
            )
            if llm_output is not None:
                return llm_output

        detail = request.message.strip()
        profile_hint = self._profile_hint(profile)
        reply = (
            f"根据你的描述，当前更像是症状评估场景。"
            f"{profile_hint}"
            f"如果症状持续、加重，或伴随新的危险信号，建议尽快线下就医。"
        )
        sources = [
            SourceRef(title="慢病症状评估通用建议", excerpt=detail[:80], source="local_stub"),
        ]
        return AgentOutput(agent="symptom_analysis", content=reply, severity="yellow", sources=sources, confidence="medium")

    @staticmethod
    def _profile_hint(profile: Profile) -> str:
        if profile.conditions:
            return f"你已记录的病史包括：{'、'.join(profile.conditions)}。"
        return ""


class MedicationAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        client: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.llm_agent = (
            ConfiguredLLMAgent(
                config,
                client or OpenAICompatibleChatClient(),
                output_agent="medication_query",
                fallback_severity="yellow",
                fallback_confidence="medium",
            )
            if config is not None
            else None
        )

    CONTRA_KEYWORDS = ["过敏", "禁忌", "肾功能", "肝功能", "孕", "哺乳"]

    def run(self, request: ChatRequest, profile: Profile, memory: List[Dict[str, str]]) -> AgentOutput:
        if self.llm_agent is not None:
            llm_output = self.llm_agent.run_llm(
                request,
                profile,
                memory,
                metadata={"agent": "medication_query", "profile": _profile_snapshot(profile)},
            )
            if llm_output is not None:
                return llm_output

        concern = self._extract_concern(request.message)
        warning = ""
        if profile.allergies:
            warning = f"你记录的过敏史有：{'、'.join(profile.allergies)}。"
        reply = (
            f"这是用药核对场景。"
            f"{warning}"
            f"我建议重点核查药名、剂量、服用频次，以及是否存在相互作用或禁忌。"
            f"{concern}"
        )
        sources = [SourceRef(title="用药安全核对", excerpt=request.message[:80], source="local_stub")]
        severity: Severity = "yellow" if concern else "green"
        return AgentOutput(agent="medication_query", content=reply, severity=severity, sources=sources, confidence="medium")

    def _extract_concern(self, text: str) -> str:
        for keyword in self.CONTRA_KEYWORDS:
            if keyword in text:
                return f" 你提到{keyword}相关问题，建议优先核查禁忌与说明书。"
        return ""


class DiagnosisAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        client: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.llm_agent = (
            ConfiguredLLMAgent(
                config,
                client or OpenAICompatibleChatClient(),
                output_agent="diagnosis_query",
                fallback_severity="red",
                fallback_confidence="high",
            )
            if config is not None
            else None
        )

    def run(self, request: ChatRequest, profile: Profile, memory: List[Dict[str, str]]) -> AgentOutput:
        if self.llm_agent is not None:
            llm_output = self.llm_agent.run_llm(
                request,
                profile,
                memory,
                metadata={"agent": "diagnosis_query", "profile": _profile_snapshot(profile)},
            )
            if llm_output is not None:
                return llm_output

        reply = (
            "从就医建议角度看，这类情况更适合做进一步线下评估。"
            "如果是新发、持续、或明显影响日常生活的症状，建议尽快挂号相关科室。"
        )
        if profile.conditions:
            reply += f" 结合你已有病史：{'、'.join(profile.conditions)}，就诊时可以主动说明。"
        sources = [SourceRef(title="分级就医建议", excerpt=request.message[:80], source="local_stub")]
        return AgentOutput(agent="diagnosis_query", content=reply, severity="red", sources=sources, confidence="high")


class GeneralAgent:
    def run(self, request: ChatRequest, profile: Profile, memory: List[Dict[str, str]]) -> AgentOutput:
        reply = "我先帮你整理一下信息。你可以补充症状、持续时间、现用药和过敏史，我再继续判断。"
        sources = [SourceRef(title="健康信息补充提示", excerpt=request.message[:80], source="local_stub")]
        return AgentOutput(agent="general_health", content=reply, severity="green", sources=sources, confidence="low")


class Orchestrator:
    def __init__(
        self,
        *,
        safety_gate: SafetyGate | None = None,
        classifier: IntentClassifier | None = None,
        profile_store: ProfileStore | None = None,
        memory: ConversationMemory | None = None,
        agent_config_store: AgentConfigStore | None = None,
        chat_client: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.safety_gate = safety_gate or SafetyGate()
        self.classifier = classifier or IntentClassifier()
        self.profile_store = profile_store or ProfileStore()
        self.memory = memory or ConversationMemory()
        self.agent_config_store = agent_config_store or AgentConfigStore()
        self.chat_client = chat_client or OpenAICompatibleChatClient()
        self.symptom_agent = SymptomAgent(
            self._load_agent_config("symptom-analysis-agent.json"),
            self.chat_client,
        )
        self.medication_agent = MedicationAgent(
            self._load_agent_config("medication-management-agent.json"),
            self.chat_client,
        )
        self.diagnosis_agent = DiagnosisAgent(
            self._load_agent_config("diagnosis-guidance-agent.json"),
            self.chat_client,
        )
        self.general_agent = GeneralAgent()

    def _load_agent_config(self, filename: str) -> AgentConfig | None:
        try:
            return self.agent_config_store.load(filename)
        except (AgentConfigError, OSError, ValueError, KeyError):
            return None

    def _llm_configured(self) -> bool:
        for agent in (self.symptom_agent, self.medication_agent, self.diagnosis_agent):
            llm_agent = getattr(agent, "llm_agent", None)
            if llm_agent is not None and self.chat_client.is_configured(llm_agent.config):
                return True
        return False

    def chat(self, request: ChatRequest) -> Dict[str, Any]:
        safety = self.safety_gate.check(request.message)
        profile = self.profile_store.get(request.user_id)
        profile_prompt = render_profile_prompt(profile)

        self.memory.append(request.session_id, "user", request.message)
        history = [{"role": "system", "content": profile_prompt}, *self.memory.get(request.session_id)]

        if safety["emergency"]:
            response = ChatResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                message=request.message,
                reply=(
                    "当前输入触发了高风险安全规则。"
                    "请立即停止自我处理，尽快联系当地急救或前往最近急诊。"
                ),
                intent="high_risk_input",
                severity="red",
                emergency=True,
                sources=[SourceRef(title="安全网关", excerpt=safety["matched_rule"], source="local_stub")],
            )
            self.memory.append(request.session_id, "assistant", response.reply)
            return envelope(0, "ok", response.to_dict(), disclaimer=self._disclaimer())

        intent = self.classifier.classify(request.message, emergency=False)
        agent_output = self._route(intent, request, profile, history)
        response = ChatResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            message=request.message,
            reply=agent_output.content,
            intent=intent,
            severity=agent_output.severity,
            emergency=False,
            sources=agent_output.sources,
        )
        self.memory.append(request.session_id, "assistant", response.reply)
        return envelope(0, "ok", response.to_dict(), disclaimer=self._disclaimer())

    def _route(
        self,
        intent: Intent,
        request: ChatRequest,
        profile: Profile,
        history: List[Dict[str, str]],
    ) -> AgentOutput:
        if intent == "symptom_analysis":
            return self.symptom_agent.run(request, profile, history)
        if intent == "medication_query":
            return self.medication_agent.run(request, profile, history)
        if intent == "diagnosis_query":
            return self.diagnosis_agent.run(request, profile, history)
        if intent == "mixed_query":
            symptom_result = self.symptom_agent.run(request, profile, history)
            med_result = self.medication_agent.run(request, profile, history)
            return self._merge_results([symptom_result, med_result])
        if intent == "profile_management":
            return self.general_agent.run(request, profile, history)
        return self.general_agent.run(request, profile, history)

    @staticmethod
    def _merge_results(results: Sequence[AgentOutput]) -> AgentOutput:
        primary = results[0]
        severity_order = {"green": 0, "yellow": 1, "red": 2}
        highest = max(results, key=lambda item: severity_order[item.severity or "green"])
        content = " ".join(result.content for result in results)
        sources: List[SourceRef] = []
        for result in results:
            sources.extend(result.sources)
        confidence: Confidence = "medium"
        if all(result.confidence == "high" for result in results):
            confidence = "high"
        elif any(result.confidence == "low" for result in results):
            confidence = "low"
        return AgentOutput(
            agent="mixed_query",
            content=content,
            severity=highest.severity,
            sources=sources,
            confidence=confidence,
        )

    @staticmethod
    def _disclaimer() -> str:
        return "本系统仅供参考，不构成医疗诊断或治疗建议。"

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        return envelope(0, "ok", self.profile_store.get(user_id).to_dict(), disclaimer=self._disclaimer())

    def update_profile(
        self,
        user_id: str,
        *,
        conditions: Optional[Sequence[str]] = None,
        medications: Optional[Sequence[str]] = None,
        allergies: Optional[Sequence[str]] = None,
        condition_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        profile = self.profile_store.update(
            user_id,
            condition_description=condition_description,
            conditions=conditions,
            medications=medications,
            allergies=allergies,
        )
        return envelope(0, "ok", profile.to_dict(), disclaimer=self._disclaimer())

    def health(self) -> Dict[str, Any]:
        return envelope(
            0,
            "ok",
            {
                "status": "healthy",
                "dependencies": {
                    "profile_store": True,
                    "memory": True,
                    "llm": self._llm_configured(),
                    "vector_store": False,
                },
                "timestamp": now_iso(),
            },
            disclaimer=self._disclaimer(),
        )
