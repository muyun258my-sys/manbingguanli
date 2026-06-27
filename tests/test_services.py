import pytest
from app.models import ChatRequest
from app.services import (
    ConversationMemory,
    IntentClassifier,
    Orchestrator,
    ProfileStore,
    SafetyGate,
)


# ── health ──────────────────────────────────────────────────────────────────

def test_health():
    orc = Orchestrator()
    payload = orc.health()
    assert payload["code"] == 0
    assert payload["data"]["status"] == "healthy"


# ── profile roundtrip ────────────────────────────────────────────────────────

def test_profile_roundtrip():
    orc = Orchestrator()
    orc.update_profile("u1", conditions=["高血压"], medications=["氨氯地平"], allergies=["青霉素"])
    profile = orc.get_profile("u1")
    assert profile["data"]["conditions"] == ["高血压"]
    assert profile["data"]["medications"] == ["氨氯地平"]
    assert profile["data"]["allergies"] == ["青霉素"]


def test_profile_partial_update():
    orc = Orchestrator()
    orc.update_profile("u2", conditions=["糖尿病"])
    orc.update_profile("u2", medications=["二甲双胍"])
    p = orc.get_profile("u2")["data"]
    assert p["conditions"] == ["糖尿病"]
    assert p["medications"] == ["二甲双胍"]


def test_profile_persists_across_store_instances(tmp_path):
    """关键回归：档案落盘后，新建的 ProfileStore（模拟重启）仍能读回。"""
    db = tmp_path / "profiles.db"
    store1 = ProfileStore(db_path=db)
    store1.update("u_persist", conditions=["高血压"], allergies=["青霉素"], medications=["二甲双胍"])

    store2 = ProfileStore(db_path=db)
    p = store2.get("u_persist")
    assert p.conditions == ["高血压"]
    assert p.allergies == ["青霉素"]
    assert p.medications == ["二甲双胍"]


# ── safety gate ─────────────────────────────────────────────────────────────

def test_emergency_short_circuit():
    orc = Orchestrator()
    response = orc.chat(ChatRequest(session_id="s1", user_id="u1", message="突然胸痛，而且呼吸困难"))
    assert response["data"]["emergency"] is True
    assert response["data"]["intent"] == "high_risk_input"


@pytest.mark.parametrize("text", [
    "剧烈胸痛",
    "昏迷不醒",
    "意识不清",
    "左臂麻",
    "一侧肢体无力",
    "口角歪斜",
    "剧烈头痛",
    "大出血",
    "抽搐",
])
def test_safety_gate_patterns(text):
    gate = SafetyGate()
    result = gate.check(text)
    assert result["emergency"] is True


def test_safety_gate_normal():
    gate = SafetyGate()
    assert gate.check("头有点晕，血压 130/85")["emergency"] is False


@pytest.mark.parametrize("text", [
    "突然胸口剧痛，左臂也麻了",   # README 旗舰高风险示例
    "胸口剧痛",
    "胸口疼",
    "左臂也麻了",
    "手臂发麻",
    "胳膊麻木",
    "呼吸急促",
    "意识模糊",
    "突然晕厥",
    "一侧手脚无力",
    "头痛欲裂",
])
def test_safety_gate_colloquial_variants(text):
    """口语化/带修饰字的高风险表达也应触发 Short-circuit。"""
    assert SafetyGate().check(text)["emergency"] is True


@pytest.mark.parametrize("text", [
    "最近有点乏力",
    "浑身没劲",
    "偶尔头晕",
    "胸口不太舒服想问问",
])
def test_safety_gate_no_false_positive(text):
    assert SafetyGate().check(text)["emergency"] is False


def test_emergency_readme_example_short_circuits():
    orc = Orchestrator()
    resp = orc.chat(ChatRequest(session_id="se", user_id="ue", message="突然胸口剧痛，左臂也麻了"))
    assert resp["data"]["emergency"] is True
    assert resp["data"]["intent"] == "high_risk_input"


# ── intent classifier ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("最近头晕，血压 160/100", "symptom_analysis"),
    ("二甲双胍和阿司匹林能一起吃吗", "medication_query"),
    ("这个情况要挂什么科", "diagnosis_query"),
    ("更新档案，我有高血压", "profile_management"),
])
def test_classifier_single_intent(text, expected):
    clf = IntentClassifier()
    assert clf.classify(text) == expected


def test_classifier_mixed():
    clf = IntentClassifier()
    # 头晕 → symptom; 药 → medication; score >= 2 → mixed_query
    intent = clf.classify("最近头晕，一直吃着降压药，要不要去医院")
    assert intent == "mixed_query"


def test_classifier_emergency_overrides():
    clf = IntentClassifier()
    assert clf.classify("任何文字", emergency=True) == "high_risk_input"


def test_classifier_general_fallback():
    clf = IntentClassifier()
    assert clf.classify("你好") == "general_health"


# ── conversation memory ──────────────────────────────────────────────────────

def test_memory_window():
    mem = ConversationMemory(window_size=2)
    for i in range(6):
        mem.append("s", "user", f"msg{i}")
    history = mem.get("s")
    assert len(history) <= 4   # window_size * 2


def test_memory_session_isolation():
    mem = ConversationMemory()
    mem.append("s1", "user", "hello")
    mem.append("s2", "user", "world")
    assert mem.get("s1") != mem.get("s2")
    assert mem.get("s1")[0]["content"] == "hello"


# ── per-agent smoke tests ────────────────────────────────────────────────────

def test_symptom_agent_reply():
    orc = Orchestrator()
    resp = orc.chat(ChatRequest(session_id="sa", user_id="ua", message="最近头晕，心慌"))
    d = resp["data"]
    assert d["intent"] == "symptom_analysis"
    assert d["emergency"] is False
    assert d["reply"]


def test_medication_agent_reply():
    orc = Orchestrator()
    resp = orc.chat(ChatRequest(session_id="sm", user_id="um", message="这个药有什么副作用禁忌"))
    d = resp["data"]
    assert d["intent"] == "medication_query"
    assert d["reply"]


def test_diagnosis_agent_reply():
    orc = Orchestrator()
    resp = orc.chat(ChatRequest(session_id="sd", user_id="ud", message="这种情况严重吗，建议就医吗"))
    d = resp["data"]
    assert d["intent"] == "diagnosis_query"
    assert d["severity"] == "red"


def test_mixed_query_reply():
    orc = Orchestrator()
    resp = orc.chat(ChatRequest(session_id="smx", user_id="umx", message="乏力头晕，在吃降压药，有副作用禁忌吗"))
    d = resp["data"]
    assert d["intent"] == "mixed_query"
    assert d["reply"]


# ── severity propagation ─────────────────────────────────────────────────────

def test_disclaimer_always_present():
    orc = Orchestrator()
    resp = orc.chat(ChatRequest(session_id="s_disc", user_id="u_disc", message="你好"))
    assert resp["disclaimer"]
    assert "参考" in resp["disclaimer"]


def test_profile_hint_in_reply():
    orc = Orchestrator()
    orc.update_profile("u_hint", conditions=["高血压"])
    resp = orc.chat(ChatRequest(session_id="s_hint", user_id="u_hint", message="最近头晕乏力"))
    assert "高血压" in resp["data"]["reply"]
