from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    postgres_url: str
    redis_url: str
    secret_key: str
    access_token_expire_minutes: int = 60
    chroma_persist_dir: str = "./chroma_db"
    langchain_tracing_v2: bool = False
    environment: str = "development"
    cors_origins: str = "http://localhost:5173"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()