from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routes import upload
from app.agents.orchestrator import run_agent

settings = get_settings()

app = FastAPI(title="LearnIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/upload", tags=["upload"])

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/test-agent")
async def test_agent(subject: str, query: str):
    result = await run_agent(
        session_id="test-session",
        subject=subject,
        query=query
    )
    return result