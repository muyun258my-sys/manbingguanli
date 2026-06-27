"""慢病健康管理助理 — Streamlit 前端"""
from __future__ import annotations

import uuid
import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8080"

SEVERITY_BADGE = {
    "red": "🔴 立即就医",
    "yellow": "🟡 建议近期就诊",
    "green": "🟢 可自行观察",
}

DISCLAIMER = "本系统仅供参考，不构成医疗诊断或治疗建议。"


def _api_base() -> str:
    return st.session_state.get("api_base", DEFAULT_API_BASE).rstrip("/")


def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _chat(message: str) -> dict:
    resp = requests.post(
        f"{_api_base()}/chat",
        json={
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.user_id,
            "message": message,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _get_profile() -> dict:
    resp = requests.get(f"{_api_base()}/profile/{st.session_state.user_id}", timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", {})


def _update_profile(condition_description: str, conditions: list, medications: list, allergies: list) -> None:
    requests.put(
        f"{_api_base()}/profile/{st.session_state.user_id}",
        json={
            "condition_description": condition_description,
            "conditions": conditions,
            "medications": medications,
            "allergies": allergies,
        },
        timeout=10,
    ).raise_for_status()


def render_profile_panel() -> None:
    st.subheader("健康档案")
    try:
        profile = _get_profile()
    except Exception as e:
        st.error(f"无法读取档案：{e}")
        return

    with st.form("profile_form"):
        condition_description = st.text_area(
            "当前病情描述",
            value=profile.get("condition_description") or "",
            placeholder="例如：最近两周头晕，血压多次在 150/95 左右，偶尔心慌。",
        )
        conditions = st.text_area(
            "已知病史（每行一项）",
            value="\n".join(profile.get("conditions") or []),
        )
        medications = st.text_area(
            "当前用药（每行一项）",
            value="\n".join(profile.get("medications") or []),
        )
        allergies = st.text_area(
            "过敏史（每行一项）",
            value="\n".join(profile.get("allergies") or []),
        )
        if st.form_submit_button("保存档案"):
            try:
                _update_profile(
                    condition_description.strip(),
                    [c.strip() for c in conditions.splitlines() if c.strip()],
                    [m.strip() for m in medications.splitlines() if m.strip()],
                    [a.strip() for a in allergies.splitlines() if a.strip()],
                )
                st.success("档案已保存")
            except Exception as e:
                st.error(f"保存失败：{e}")


def render_chat() -> None:
    st.subheader("对话")

    # render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("severity") and msg["role"] == "assistant":
                badge = SEVERITY_BADGE.get(msg["severity"], "")
                if badge:
                    st.markdown(f"**{badge}**")
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("引用来源"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src.get('title', '')}** — {src.get('source', '')}")
            if msg.get("disclaimer"):
                st.caption(msg["disclaimer"])

    if prompt := st.chat_input("请描述您的症状、用药问题或就医疑问…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("正在处理…")
            try:
                result = _chat(prompt)
                data = result.get("data", {})
                reply = data.get("reply", "（无回复）")
                severity = data.get("severity")
                emergency = data.get("emergency", False)
                sources = data.get("sources", [])
                disclaimer = result.get("disclaimer", DISCLAIMER)

                if emergency:
                    st.error("⚠️ 高风险提示：请立即拨打急救电话或前往最近急诊！")

                if severity:
                    badge = SEVERITY_BADGE.get(severity, "")
                    if badge:
                        st.markdown(f"**{badge}**")

                placeholder.markdown(reply)

                if sources:
                    with st.expander("引用来源"):
                        for src in sources:
                            st.markdown(f"- **{src.get('title', '')}** — {src.get('source', '')}")

                st.caption(disclaimer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "severity": severity,
                    "sources": sources,
                    "disclaimer": disclaimer,
                })

            except Exception as e:
                placeholder.error(f"请求失败：{e}")


def main() -> None:
    st.set_page_config(page_title="慢病健康管理助理", page_icon="🏥", layout="wide")
    st.sidebar.text_input("Backend URL", value=DEFAULT_API_BASE, key="api_base")
    st.title("🏥 慢病健康管理助理")
    st.caption(DISCLAIMER)

    _init_session()

    col_chat, col_profile = st.columns([2, 1])
    with col_chat:
        render_chat()
    with col_profile:
        render_profile_panel()


if __name__ == "__main__":
    main()
