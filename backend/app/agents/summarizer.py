from openai import AsyncOpenAI
from app.config import get_settings
from app.agents.retriever import run_retriever

settings = get_settings()

PER_DOC_PROMPT = """You are an academic summarizer. Create a structured summary of the provided content.

Format your response as:
## Overview
2-3 sentence high-level summary

## Key Concepts
- Bullet points of the most important concepts

## Main Arguments / Findings
- Key arguments or findings presented

## Important Definitions
- Term: definition (only if definitions are present)

## What to Remember
2-3 most critical takeaways for a student

Keep it concise but comprehensive. Base everything strictly on the provided context."""

CROSS_DOC_PROMPT = """You are an expert at synthesizing knowledge across multiple sources.

Compare and synthesize the provided content from multiple documents.

Format your response as:
## Synthesis Overview
High-level synthesis of all materials

## Common Themes
Themes that appear across multiple documents

## Contrasting Perspectives
Where documents differ or present different angles (if any)

## Combined Key Takeaways
The most important points drawing from all sources

Base everything strictly on the provided context."""

async def summarize(subject: str, topic: str, mode: str = "per_doc") -> dict:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    retrieval = await run_retriever(topic, subject, n_results=10)
    context = retrieval.get("context", "")
    sources = retrieval.get("sources", [])

    system_prompt = CROSS_DOC_PROMPT if mode == "cross_doc" else PER_DOC_PROMPT

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Content to summarize:\n{context}\n\nTopic: {topic}"}
        ],
        temperature=0.3
    )

    return {
        "subject": subject,
        "topic": topic,
        "mode": mode,
        "summary": response.choices[0].message.content,
        "sources": sources
    }