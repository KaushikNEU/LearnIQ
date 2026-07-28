import json
import redis.asyncio as redis
from app.config import get_settings

settings = get_settings()
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

async def get_session_history(session_id: str) -> list:
    raw = await redis_client.get(f"session:{session_id}")
    return json.loads(raw) if raw else []

async def append_to_session(session_id: str, role: str, content: str):
    history = await get_session_history(session_id)
    history.append({"role": role, "content": content})
    # Keep last 20 messages only
    history = history[-20:]
    await redis_client.setex(
        f"session:{session_id}",
        3600,  # 1 hour TTL
        json.dumps(history)
    )

async def clear_session(session_id: str):
    await redis_client.delete(f"session:{session_id}")