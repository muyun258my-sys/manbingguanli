from app.models import AgentOutput, ChatRequest
from app.services import Orchestrator, ProfileStore, render_profile_prompt


def test_profile_stores_condition_description():
    store = ProfileStore()
    profile = store.update(
        "u1",
        condition_description="反复头晕两周",
        conditions=["高血压"],
        medications=["氨氯地平"],
        allergies=["青霉素"],
    )

    assert profile.condition_description == "反复头晕两周"
    assert profile.conditions == ["高血压"]
    assert profile.medications == ["氨氯地平"]
    assert profile.allergies == ["青霉素"]


def test_render_profile_prompt_includes_saved_context():
    profile = ProfileStore().update(
        "u2",
        condition_description="夜间胸闷，持续三天",
        conditions=["高血压"],
        medications=["氨氯地平"],
        allergies=["青霉素"],
    )

    prompt = render_profile_prompt(profile)

    assert "夜间胸闷，持续三天" in prompt
    assert "高血压" in prompt
    assert "氨氯地平" in prompt
    assert "青霉素" in prompt


def test_chat_injects_profile_prompt_into_each_agent_call():
    class SpyAgent:
        def __init__(self):
            self.memory = []

        def run(self, request, profile, memory, knowledge_sources=()):
            self.memory = memory
            return AgentOutput(agent="general_health", content="ok", severity="green")

    orchestrator = Orchestrator()
    spy_agent = SpyAgent()
    orchestrator.general_agent = spy_agent
    orchestrator.update_profile(
        "u3",
        condition_description="反复头晕",
        conditions=["高血压"],
        medications=["氨氯地平"],
        allergies=["青霉素"],
    )

    orchestrator.chat(ChatRequest(session_id="s3", user_id="u3", message="你好"))

    assert spy_agent.memory[0]["role"] == "system"
    assert "反复头晕" in spy_agent.memory[0]["content"]
    assert "高血压" in spy_agent.memory[0]["content"]
    assert "氨氯地平" in spy_agent.memory[0]["content"]
    assert "青霉素" in spy_agent.memory[0]["content"]
