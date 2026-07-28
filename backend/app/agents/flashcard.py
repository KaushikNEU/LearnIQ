import json
from openai import AsyncOpenAI
from app.config import get_settings
from app.agents.retriever import run_retriever

settings = get_settings()

SYSTEM_PROMPT = """You are a flashcard creator specializing in active recall and spaced repetition.

Create flashcards from the provided content. Each card should:
- Have a clear, specific question on the front
- Have a concise, complete answer on the back
- Focus on one concept per card
- Use simple language
- Be suitable for spaced repetition study

Return ONLY valid JSON, no other text:
{{
  "flashcards": [
    {{
      "id": 1,
      "front": "Question or prompt",
      "back": "Answer",
      "tags": ["topic_tag"],
      "difficulty": "easy|medium|hard"
    }}
  ]
}}"""

async def generate_flashcards(subject: str, topic: str, num_cards: int = 10) -> dict:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    retrieval = await run_retriever(topic, subject, n_results=10)
    context = retrieval.get("context", "")
    sources = retrieval.get("sources", [])

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Content:\n{context}\n\nCreate {num_cards} flashcards about: {topic}"
            }
        ],
        temperature=0.5,
        response_format={"type": "json_object"}
    )

    raw = json.loads(response.choices[0].message.content)
    flashcards = raw.get("flashcards", [])

    # Build Anki-compatible export format
    anki_export = "\n".join([
        f"{card['front']}\t{card['back']}"
        for card in flashcards
    ])

    return {
        "subject": subject,
        "topic": topic,
        "flashcards": flashcards,
        "anki_export": anki_export,
        "sources": sources
    }