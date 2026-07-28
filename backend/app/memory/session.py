import json
import logging
import redis.asyncio as redis
from app.config import get_settings

settings = get_settings()
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

logger = logging.getLogger("learniq.session")

async def get_session_history(session_id: str) -> list:
    try:
        raw = await redis_client.get(f"session:{session_id}")
        return json.loads(raw) if raw else []
    except (redis.ConnectionError, redis.TimeoutError) as e:
        # Redis unreachable (e.g. no Redis provisioned in this environment) —
        # degrade to "no history" instead of crashing the whole request.
        logger.warning(f"Redis unavailable, returning empty history for session {session_id}: {e}")
        return []

async def append_to_session(session_id: str, role: str, content: str):
    try:
        history = await get_session_history(session_id)
        history.append({"role": role, "content": content})
        # Keep last 20 messages only
        history = history[-20:]
        await redis_client.setex(
            f"session:{session_id}",
            3600,  # 1 hour TTL
            json.dumps(history)
        )
    except (redis.ConnectionError, redis.TimeoutError) as e:
        # Same degradation here — skip persisting this turn rather than
        # failing the request. The response still gets returned to the user;
        # it just won't be remembered for the next turn.
        logger.warning(f"Redis unavailable, skipping session write for {session_id}: {e}")

async def clear_session(session_id: str):
    try:
        await redis_client.delete(f"session:{session_id}")
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Redis unavailable, could not clear session {session_id}: {e}")