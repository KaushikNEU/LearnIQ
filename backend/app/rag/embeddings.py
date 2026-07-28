from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()

def get_client():
    return AsyncOpenAI(api_key=settings.openai_api_key)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_client()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

async def embed_query(query: str) -> list[float]:
    client = get_client()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding