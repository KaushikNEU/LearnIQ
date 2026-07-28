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

# set_page_config MUST be the very first Streamlit command in the script —
# even accessing st.secrets before this line counts as an earlier command
# and raises StreamlitAPIException, so this has to come first.
st.set_page_config(
    page_title="LearnIQ",
    page_icon="🎓",
    layout="centered"
)

# ── Config ─────────────────────────────────────────────────────────────────
# Reads from Streamlit Cloud "Secrets" if present, otherwise from .env,
# otherwise falls back to localhost for local development.
# st.secrets.get() raises FileNotFoundError (not just a missing-key default)
# when no secrets.toml exists at all — which is the normal case for local
# dev — so we catch that explicitly rather than relying on .get()'s default.
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (FileNotFoundError, KeyError):
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

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

def format_quiz(questions: list) -> str:
    """Renders quiz_agent.py's structured question list as readable markdown."""
    if not questions:
        return "No questions returned."

    lines = []
    for q in questions:
        lines.append(f"**Q{q.get('id', '?')}. {q.get('question', '')}**")
        if q.get("type") == "mcq":
            for opt in q.get("options", []):
                lines.append(f"- {opt}")
            lines.append(f"\n*Correct answer: {q.get('correct_answer', '?')}*")
            if q.get("explanation"):
                lines.append(f"*{q['explanation']}*")
        else:  # short_answer
            if q.get("sample_answer"):
                lines.append(f"\n*Sample answer: {q['sample_answer']}*")
            key_points = q.get("key_points", [])
            if key_points:
                lines.append("*Key points: " + ", ".join(key_points) + "*")
        lines.append("")  # blank line between questions
    return "\n".join(lines)

def format_flashcards(cards: list) -> str:
    """Renders flashcard.py's card list as readable markdown (adjust keys if yours differ)."""
    if not cards:
        return "No flashcards returned."

    lines = []
    for i, card in enumerate(cards, 1):
        front = card.get("front") or card.get("term") or card.get("question") or "?"
        back = card.get("back") or card.get("definition") or card.get("answer") or "?"
        lines.append(f"**{i}. {front}**")
        lines.append(f"   {back}\n")
    return "\n".join(lines)

def extract_display_text(result: dict) -> str:
    """Pulls the most relevant text out of whatever shape orchestrator.py returns,
    formatted appropriately for each agent's actual output structure."""
    if result.get("error"):
        return f"⚠️ {result['error']}"

    payload = result.get("result", {}) or {}

    if payload.get("explanation"):
        return payload["explanation"]

    if payload.get("summary"):
        return payload["summary"]

    if "questions" in payload:
        return format_quiz(payload["questions"])

    if "flashcards" in payload:
        return format_flashcards(payload["flashcards"])

    return "No content returned."

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

    # ── Document upload ──────────────────────────────────────────────────
    st.subheader("📄 Upload course material")
    uploaded_file = st.file_uploader(
        "Add a document for LearnIQ to use in answers",
        type=["pdf", "docx", "txt"],
        help="This calls the same /api/upload/ endpoint visible in the FastAPI docs"
    )

    if uploaded_file is not None:
        if st.button("Upload", use_container_width=True):
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {"subject": st.session_state.subject}
                    res = requests.post(
                        f"{BACKEND_URL}/api/upload/",
                        files=files,
                        data=data,
                        timeout=60
                    )
                    if res.status_code == 200:
                        st.success(f"✓ Uploaded {uploaded_file.name} to '{st.session_state.subject}'")
                    else:
                        st.error(f"Upload failed ({res.status_code}): {res.text[:200]}")
                except Exception as e:
                    st.error(f"Upload error: {e}")

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