from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from app.agents.explainer import run_explainer
from app.agents.quiz_agent import generate_quiz, evaluate_answer
from app.agents.summarizer import summarize
from app.agents.flashcard import generate_flashcards
from app.agents.critic import validate_response
from app.memory.session import get_session_history, append_to_session
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()

# ── State definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    session_id: str
    subject: str
    query: str
    intent: str
    eli5: bool
    topic: str
    num_items: int
    mode: str
    result: dict
    error: str

# ── Intent classifier ─────────────────────────────────────────────────────────
async def classify_intent(state: AgentState) -> AgentState:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Classify the user query into exactly one intent.
Return ONLY one word from: explain, quiz, summarize, flashcards, evaluate

- explain: asking about a concept, definition, or how something works
- quiz: requesting practice questions or a test
- summarize: asking for a summary or overview
- flashcards: requesting flashcards or study cards
- evaluate: submitting an answer to be graded"""
            },
            {"role": "user", "content": state["query"]}
        ],
        temperature=0.0
    )

    intent = response.choices[0].message.content.strip().lower()
    if intent not in ["explain", "quiz", "summarize", "flashcards", "evaluate"]:
        intent = "explain"

    return {**state, "intent": intent}

# ── Agent nodes ───────────────────────────────────────────────────────────────
async def explainer_node(state: AgentState) -> AgentState:
    try:
        result = await run_explainer(
            query=state["query"],
            subject=state["subject"],
            eli5=state.get("eli5", False)
        )
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e), "result": {}}

async def quiz_node(state: AgentState) -> AgentState:
    try:
        result = await generate_quiz(
            subject=state["subject"],
            topic=state.get("topic", state["query"]),
            num_questions=state.get("num_items", 5)
        )
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e), "result": {}}

async def summarizer_node(state: AgentState) -> AgentState:
    try:
        result = await summarize(
            subject=state["subject"],
            topic=state.get("topic", state["query"]),
            mode=state.get("mode", "per_doc")
        )
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e), "result": {}}

async def flashcard_node(state: AgentState) -> AgentState:
    try:
        result = await generate_flashcards(
            subject=state["subject"],
            topic=state.get("topic", state["query"]),
            num_cards=state.get("num_items", 10)
        )
        return {**state, "result": result}
    except Exception as e:
        return {**state, "error": str(e), "result": {}}

async def critic_node(state: AgentState) -> AgentState:
    result = state.get("result", {})
    if not result or state.get("error"):
        return state

    # Get the main text content to validate
    content_to_check = (
        result.get("explanation") or
        result.get("summary") or
        str(result.get("questions", "")) or
        str(result.get("flashcards", ""))
    )

    if not content_to_check:
        return state

    validation = await validate_response(
        user_query=state["query"],
        ai_response=content_to_check,
        context=str(result.get("sources", ""))
    )

    # Attach validation metadata to result
    result["validated"] = validation["safe"]
    result["validation_issues"] = validation["issues"]

    return {**state, "result": result}

# ── Router ────────────────────────────────────────────────────────────────────
def route_intent(state: AgentState) -> Literal[
    "explainer", "quiz", "summarizer", "flashcards"
]:
    intent_map = {
        "explain": "explainer",
        "quiz": "quiz",
        "evaluate": "quiz",
        "summarize": "summarizer",
        "flashcards": "flashcards"
    }
    return intent_map.get(state["intent"], "explainer")

# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify",   classify_intent)
    graph.add_node("explainer",  explainer_node)
    graph.add_node("quiz",       quiz_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("flashcards", flashcard_node)
    graph.add_node("critic",     critic_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "explainer":  "explainer",
            "quiz":       "quiz",
            "summarizer": "summarizer",
            "flashcards": "flashcards"
        }
    )

    graph.add_edge("explainer",  "critic")
    graph.add_edge("quiz",       "critic")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("flashcards", "critic")
    graph.add_edge("critic",     END)

    return graph.compile()

# Compile once at module level
learniq_graph = build_graph()

# ── Public entry point ────────────────────────────────────────────────────────
async def run_agent(
    session_id: str,
    subject: str,
    query: str,
    eli5: bool = False,
    topic: str = "",
    num_items: int = 5,
    mode: str = "per_doc"
) -> dict:

    # Load session memory
    history = await get_session_history(session_id)

    initial_state: AgentState = {
        "session_id": session_id,
        "subject": subject,
        "query": query,
        "intent": "",
        "eli5": eli5,
        "topic": topic or query,
        "num_items": num_items,
        "mode": mode,
        "result": {},
        "error": ""
    }

    final_state = await learniq_graph.ainvoke(initial_state)

    # Save to session memory
    await append_to_session(session_id, "user", query)
    await append_to_session(
        session_id, "assistant",
        str(final_state.get("result", {}).get("explanation", ""))[:500]
    )

    return {
        "intent": final_state.get("intent"),
        "result": final_state.get("result"),
        "error": final_state.get("error"),
        "session_id": session_id
    }