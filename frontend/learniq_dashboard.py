"""
LearnIQ — Streamlit Frontend

Talks to the FastAPI backend (app/main.py) over HTTP via /api/test-agent.
Reads BACKEND_URL from an environment variable / Streamlit secret so the
exact same file works locally (localhost:8000) and once deployed
(Render URL), with no code changes.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
# Reads from Streamlit Cloud "Secrets" if present, otherwise from .env,
# otherwise falls back to localhost for local development.
BACKEND_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000"))

st.set_page_config(
    page_title="LearnIQ",
    page_icon="🎓",
    layout="centered"
)

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str, "intent": str|None}

if "subject" not in st.session_state:
    st.session_state.subject = "Machine Learning"

# ── Backend call ───────────────────────────────────────────────────────────
def call_learniq(subject: str, query: str) -> dict:
    """Calls the deployed /api/test-agent endpoint."""
    try:
        res = requests.post(
            f"{BACKEND_URL}/api/test-agent",
            params={"subject": subject, "query": query},
            timeout=60
        )
        res.raise_for_status()
        return res.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. If the backend is on a free tier, it may be waking up — try again in a few seconds."}
    except requests.exceptions.ConnectionError:
        return {"error": f"Could not reach the backend at {BACKEND_URL}. Confirm it's deployed and running."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Backend returned an error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

def extract_display_text(result: dict) -> str:
    """Pulls the most relevant text out of whatever shape orchestrator.py returns."""
    if result.get("error"):
        return f"⚠️ {result['error']}"

    payload = result.get("result", {}) or {}
    text = (
        payload.get("explanation")
        or payload.get("summary")
        or payload.get("questions")
        or payload.get("flashcards")
    )
    if text is None:
        return "No content returned."
    if isinstance(text, (list, dict)):
        import json
        return json.dumps(text, indent=2)
    return str(text)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 LearnIQ")
    st.caption("AI Learning Companion")

    st.session_state.subject = st.text_input(
        "Subject",
        value=st.session_state.subject,
        help="The course/subject context LearnIQ should use for RAG retrieval"
    )

    st.divider()

    st.caption("Backend status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success(f"Connected — v{health.json().get('version', '?')}")
        else:
            st.warning(f"Backend responded with {health.status_code}")
    except Exception:
        st.error("Backend unreachable")

    st.caption(f"`{BACKEND_URL}`")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat UI ───────────────────────────────────────────────────────────
st.title("Ask LearnIQ")
st.caption("Explain a concept, request a quiz, ask for a summary, or generate flashcards.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("intent"):
            st.caption(f"Intent detected: `{msg['intent']}`")
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("What is Gradient Descent?"):
    st.session_state.messages.append({"role": "user", "content": query, "intent": None})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = call_learniq(st.session_state.subject, query)
            display_text = extract_display_text(result)
            intent = result.get("intent")

        if intent:
            st.caption(f"Intent detected: `{intent}`")
        st.markdown(display_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": display_text,
        "intent": intent
    })

# ── Quick example buttons ──────────────────────────────────────────────────
st.divider()
st.caption("Try an example:")
cols = st.columns(3)
examples = [
    "What is Gradient Descent?",
    "Give me a quiz on overfitting",
    "Summarize regularization techniques",
]
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": example, "intent": None})
        result = call_learniq(st.session_state.subject, example)
        display_text = extract_display_text(result)
        st.session_state.messages.append({
            "role": "assistant",
            "content": display_text,
            "intent": result.get("intent")
        })
        st.rerun()