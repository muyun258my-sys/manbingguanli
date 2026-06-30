from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

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
DEFAULT_VECTOR_DB_DIR = Path(__file__).resolve().parents[1] / "vector_db"
DEFAULT_VECTOR_COLLECTION = "pdf_knowledge"


def _resolve_db_path(db_path: Optional[Path | str]) -> Path:
    if db_path is not None:
        return Path(db_path)
    env_path = os.getenv("APP_PROFILE_DB")
    if env_path:
        return Path(env_path)
    return DEFAULT_PROFILE_DB


class ProfileRepository(Protocol):
    def get(self, user_id: str) -> Profile:
        ...

    def update(
        self,
        user_id: str,
        *,
        condition_description: Optional[str] = None,
        conditions: Optional[Sequence[str]] = None,
        medications: Optional[Sequence[str]] = None,
        allergies: Optional[Sequence[str]] = None,
    ) -> Profile:
        ...

    def is_available(self) -> bool:
        ...


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

    def is_available(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False


class MySQLProfileStore:
    """User health profile storage backed by MySQL."""

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        self.host = host or os.getenv("MYSQL_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("MYSQL_PORT", "3306"))
        self.user = user or os.getenv("MYSQL_USER", "root")
        self.password = password if password is not None else os.getenv("MYSQL_PASSWORD", "")
        self.database = database or os.getenv("MYSQL_DATABASE", "xm2")
        self._init_db()

    def _connect(self):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install pymysql to use MySQL profile storage.") from exc

        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def _init_db(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS profiles (
                        user_id VARCHAR(191) PRIMARY KEY,
                        condition_description TEXT NOT NULL,
                        conditions JSON NOT NULL,
                        medications JSON NOT NULL,
                        allergies JSON NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                    """
                )

    def get(self, user_id: str) -> Profile:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM profiles WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO profiles
                        (user_id, condition_description, conditions, medications, allergies, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        condition_description = VALUES(condition_description),
                        conditions = VALUES(conditions),
                        medications = VALUES(medications),
                        allergies = VALUES(allergies),
                        updated_at = VALUES(updated_at)
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

    def is_available(self) -> bool:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False


def create_profile_store() -> ProfileRepository:
    backend = os.getenv("APP_PROFILE_STORE", "sqlite").strip().lower()
    if backend == "mysql":
        return MySQLProfileStore()
    return ProfileStore()


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


class KnowledgeRetriever:
    def __init__(
        self,
        *,
        vector_db_dir: Optional[Path | str] = None,
        collection_name: Optional[str] = None,
        top_k: int = 3,
    ) -> None:
        self.vector_db_dir = Path(vector_db_dir or os.getenv("APP_VECTOR_DB_DIR", str(DEFAULT_VECTOR_DB_DIR)))
        self.collection_name = collection_name or os.getenv("APP_VECTOR_COLLECTION", DEFAULT_VECTOR_COLLECTION)
        self.top_k = int(os.getenv("APP_RAG_TOP_K", str(top_k)))

    def retrieve(self, query: str) -> List[SourceRef]:
        if not query.strip() or not self.vector_db_dir.exists():
            return []
        try:
            from .ingestion import retrieve

            hits = retrieve(
                query,
                vector_db_dir=self.vector_db_dir,
                collection_name=self.collection_name,
                top_k=self.top_k,
            )
        except Exception:
            return []

        sources: List[SourceRef] = []
        for hit in hits:
            metadata = hit.get("metadata") or {}
            title = str(metadata.get("pdf_name") or metadata.get("source_path") or "PDF reference")
            page = metadata.get("page")
            source = str(metadata.get("source_path") or self.collection_name)
            if page:
                source = f"{source}#page={page}"
            sources.append(
                SourceRef(
                    title=title,
                    excerpt=str(hit.get("text") or "")[:300],
                    source=source,
                )
            )
        return sources

    def is_available(self) -> bool:
        if not self.vector_db_dir.exists():
            return False
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.vector_db_dir))
            collection = client.get_collection(self.collection_name)
            return collection.count() > 0
        except Exception:
            return False


def render_knowledge_prompt(sources: Sequence[SourceRef]) -> str:
    if not sources:
        return ""
    lines = ["Retrieved PDF reference snippets:"]
    for index, source in enumerate(sources, start=1):
        lines.append(f"[{index}] {source.title} ({source.source})\n{source.excerpt}")
    lines.append("Use these snippets as reference material when relevant, and do not claim they are a diagnosis.")
    return "\n\n".join(lines)


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
    DIET_WORDS = ["吃", "喝", "食物", "水果", "饮食", "忌口", "能不能吃", "可以吃", "能吃", "可以喝", "能喝"]
    FOOD_WORDS = [
        "咸菜", "腌菜", "盐", "酒", "茶", "咖啡", "西瓜", "水果", "米饭", "面条", "鸡蛋", "牛奶",
        "海鲜", "动物内脏", "火锅", "甜食", "饮料", "肉", "豆腐", "豆制品", "蔬菜", "坚果",
    ]
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
        has_diet = self._is_diet_query(text)

        if has_diet and not has_profile:
            return "diet_query"

        score = sum([has_medication, has_diagnosis, has_profile, has_symptom, has_diet])
        if score > 1:
            return "mixed_query"
        if has_profile:
            return "profile_management"
        if has_diet:
            return "diet_query"
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

    def _is_diet_query(self, text: str) -> bool:
        return self._contains_any(text, self.DIET_WORDS) and (
            self._contains_any(text, self.FOOD_WORDS)
            or "能不能吃" in text
            or "可以吃" in text
            or "能吃" in text
            or "忌口" in text
        )


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
        knowledge_sources: Sequence[SourceRef] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentOutput]:
        if not self.client.is_configured(self.config):
            return None

        knowledge_prompt = render_knowledge_prompt(knowledge_sources)
        user_message = {
            "role": "user",
            "content": (
                "User message:\n"
                f"{request.message}\n\n"
                "User profile JSON:\n"
                f"{_profile_snapshot(profile)}"
                f"\n\n{knowledge_prompt}"
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
            sources=list(knowledge_sources) or [SourceRef(title=self.config.name, excerpt=request.message[:80], source="deepseek")],
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

    def run(
        self,
        request: ChatRequest,
        profile: Profile,
        memory: List[Dict[str, str]],
        knowledge_sources: Sequence[SourceRef] = (),
    ) -> AgentOutput:
        if self.llm_agent is not None:
            llm_output = self.llm_agent.run_llm(
                request,
                profile,
                memory,
                knowledge_sources=knowledge_sources,
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
        sources.extend(knowledge_sources)
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

    def run(
        self,
        request: ChatRequest,
        profile: Profile,
        memory: List[Dict[str, str]],
        knowledge_sources: Sequence[SourceRef] = (),
    ) -> AgentOutput:
        if self.llm_agent is not None:
            llm_output = self.llm_agent.run_llm(
                request,
                profile,
                memory,
                knowledge_sources=knowledge_sources,
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
        sources.extend(knowledge_sources)
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

    def run(
        self,
        request: ChatRequest,
        profile: Profile,
        memory: List[Dict[str, str]],
        knowledge_sources: Sequence[SourceRef] = (),
    ) -> AgentOutput:
        if self.llm_agent is not None:
            llm_output = self.llm_agent.run_llm(
                request,
                profile,
                memory,
                knowledge_sources=knowledge_sources,
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
        sources.extend(knowledge_sources)
        return AgentOutput(agent="diagnosis_query", content=reply, severity="red", sources=sources, confidence="high")


class GeneralAgent:
    def run(
        self,
        request: ChatRequest,
        profile: Profile,
        memory: List[Dict[str, str]],
        knowledge_sources: Sequence[SourceRef] = (),
    ) -> AgentOutput:
        reply = "我先帮你整理一下信息。你可以补充症状、持续时间、现用药和过敏史，我再继续判断。"
        sources = [SourceRef(title="健康信息补充提示", excerpt=request.message[:80], source="local_stub")]
        sources.extend(knowledge_sources)
        return AgentOutput(agent="general_health", content=reply, severity="green", sources=sources, confidence="low")


class DietAgent:
    AVOID_RULES = [
        (["高血压", "血压"], ["咸菜", "腌菜", "高盐"], "这类高盐食物建议尽量少吃或避免，经常吃会增加钠摄入，不利于血压控制。"),
        (["高血压", "血压"], ["酒", "白酒", "啤酒", "红酒"], "血压偏高或控制不稳时不建议饮酒，饮酒可能影响血压控制和用药安全。"),
        (["糖尿病", "血糖"], ["甜饮料", "饮料", "甜食"], "含糖饮料和甜食建议避免或严格限制，容易造成血糖快速升高。"),
        (["痛风", "尿酸", "高尿酸"], ["海鲜", "动物内脏", "啤酒"], "痛风或尿酸高时建议避免高嘌呤食物和啤酒，可能诱发尿酸升高或痛风发作。"),
        (["高血脂", "血脂"], ["油炸", "肥肉", "动物内脏"], "高血脂人群建议限制高脂、高胆固醇食物，优先选择清淡烹调。"),
    ]
    LIMIT_RULES = [
        (["糖尿病", "血糖"], ["西瓜", "水果"], "可以少量吃，但要控制份量，尽量放在两餐之间，并观察餐后血糖。"),
        (["高血脂", "血脂"], ["鸡蛋"], "通常可以吃鸡蛋，但建议控制数量，少吃油煎做法，并结合血脂水平调整。"),
        (["高血压", "血压"], ["咖啡", "茶"], "一般可少量饮用，但如果喝后心慌、血压升高或睡眠变差，应减少或避免。"),
    ]

    def run(
        self,
        request: ChatRequest,
        profile: Profile,
        memory: List[Dict[str, str]],
        knowledge_sources: Sequence[SourceRef] = (),
    ) -> AgentOutput:
        text = request.message
        condition_hint = self._condition_hint(text, profile)
        advice, severity = self._match_advice(text, condition_hint)
        profile_note = self._profile_note(profile)
        reply = (
            f"{advice}"
            f"{profile_note}"
            "如果你能补充具体疾病诊断、最近指标数值、正在用药和一次大概吃多少，我可以再帮你把建议收窄到更具体的份量。"
        )
        sources = [SourceRef(title="饮食咨询建议", excerpt=request.message[:80], source="local_rule")]
        sources.extend(knowledge_sources)
        return AgentOutput(agent="diet_query", content=reply, severity=severity, sources=sources, confidence="medium")

    def _match_advice(self, text: str, condition_hint: str) -> tuple[str, Severity]:
        context = f"{condition_hint} {text}"
        for diseases, foods, advice in self.AVOID_RULES:
            if self._contains_any(context, diseases) and self._contains_any(text, foods):
                return f"建议：{advice}", "yellow"
        for diseases, foods, advice in self.LIMIT_RULES:
            if self._contains_any(context, diseases) and self._contains_any(text, foods):
                return f"建议：{advice}", "green"
        return (
            "建议：可以先按“少量、清淡、观察指标变化”的原则处理；如果这种食物高盐、高糖、高脂或高嘌呤，就要更谨慎。",
            "green",
        )

    @staticmethod
    def _contains_any(text: str, words: Sequence[str]) -> bool:
        return any(word in text for word in words)

    @staticmethod
    def _condition_hint(text: str, profile: Profile) -> str:
        profile_text = " ".join([profile.condition_description, *profile.conditions])
        return f"{profile_text} {text}"

    @staticmethod
    def _profile_note(profile: Profile) -> str:
        notes: List[str] = []
        if profile.conditions:
            notes.append(f"我会同时参考你档案里的病史：{'、'.join(profile.conditions)}。")
        if profile.medications:
            notes.append(f"你当前记录的用药有：{'、'.join(profile.medications)}，饮食调整不要自行替代药物。")
        if profile.allergies:
            notes.append(f"你记录的过敏史有：{'、'.join(profile.allergies)}，相关食物也要避开。")
        return (" " + " ".join(notes) + " ") if notes else " "


class Orchestrator:
    def __init__(
        self,
        *,
        safety_gate: SafetyGate | None = None,
        classifier: IntentClassifier | None = None,
        profile_store: ProfileRepository | None = None,
        memory: ConversationMemory | None = None,
        agent_config_store: AgentConfigStore | None = None,
        chat_client: OpenAICompatibleChatClient | None = None,
        knowledge_retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.safety_gate = safety_gate or SafetyGate()
        self.classifier = classifier or IntentClassifier()
        self.profile_store = profile_store or create_profile_store()
        self.memory = memory or ConversationMemory()
        self.agent_config_store = agent_config_store or AgentConfigStore()
        self.chat_client = chat_client or OpenAICompatibleChatClient()
        self.knowledge_retriever = knowledge_retriever or KnowledgeRetriever()
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
        self.diet_agent = DietAgent()
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
        knowledge_sources = self.knowledge_retriever.retrieve(request.message)
        if knowledge_sources:
            history = [{"role": "system", "content": render_knowledge_prompt(knowledge_sources)}, *history]
        agent_output = self._route(intent, request, profile, history, knowledge_sources)
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
        knowledge_sources: Sequence[SourceRef] = (),
    ) -> AgentOutput:
        if intent == "symptom_analysis":
            return self.symptom_agent.run(request, profile, history, knowledge_sources)
        if intent == "medication_query":
            return self.medication_agent.run(request, profile, history, knowledge_sources)
        if intent == "diagnosis_query":
            return self.diagnosis_agent.run(request, profile, history, knowledge_sources)
        if intent == "diet_query":
            return self.diet_agent.run(request, profile, history, knowledge_sources)
        if intent == "mixed_query":
            symptom_result = self.symptom_agent.run(request, profile, history, knowledge_sources)
            med_result = self.medication_agent.run(request, profile, history, knowledge_sources)
            return self._merge_results([symptom_result, med_result])
        if intent == "profile_management":
            return self.general_agent.run(request, profile, history, knowledge_sources)
        return self.general_agent.run(request, profile, history, knowledge_sources)

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
                    "profile_store": self.profile_store.is_available(),
                    "profile_store_backend": os.getenv("APP_PROFILE_STORE", "sqlite").strip().lower(),
                    "memory": True,
                    "llm": self._llm_configured(),
                    "vector_store": self.knowledge_retriever.is_available(),
                },
                "timestamp": now_iso(),
            },
            disclaimer=self._disclaimer(),
        )
