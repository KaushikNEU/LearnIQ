from openai import AsyncOpenAI
from app.config import get_settings
from app.agents.retriever import run_retriever

settings = get_settings()

SYSTEM_PROMPT = """You are an expert concept explainer and tutor.

Your job is to explain concepts clearly using:
- Simple language first, then technical depth
- Real-world analogies the student can relate to
- Step-by-step reasoning (think out loud)
- Concrete examples grounded in the provided context

Always structure your response as:
1. Simple explanation (2-3 sentences anyone can understand)
2. Deeper explanation with the technical details
3. A real-world analogy
4. A concrete example from the course material

If the student asks for ELI5 mode, keep everything at the simple explanation level only.
Base your explanation on the provided context. If context is insufficient, say so clearly.
Never fabricate information not present in the context."""

async def run_explainer(query: str, subject: str, eli5: bool = False) -> dict:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Retrieve relevant context first
    retrieval = await run_retriever(query, subject)
    context = retrieval.get("context", "")
    sources = retrieval.get("sources", [])

    mode_instruction = "\n\nIMPORTANT: The student requested ELI5 mode. Keep the explanation extremely simple — no jargon, short sentences, maximum 150 words." if eli5 else ""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + mode_instruction},
            {"role": "user", "content": f"Context from course materials:\n{context}\n\nStudent question: {query}"}
        ],
        temperature=0.4
    )

    return {
        "explanation": response.choices[0].message.content,
        "sources": sources,
        "eli5_mode": eli5
    }