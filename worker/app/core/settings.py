import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("worker/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:15432/rag_learning"
    redis_url: str = "redis://127.0.0.1:6379/0"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    storage_backend: str = "local"
    local_storage_root: str = "data/uploads"
    supabase_storage_bucket: str = "datasets"
    
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"
    gemini_embed_model: str = "gemini-embedding-001"
    
    worker_mode: str = "active"
    poll_interval_seconds: int = 5
    job_batch_size: int = 1
    gemini_api_key: str = ""
    gemini_vision_api_key: str | None = None

    @property
    def storage_local_path(self) -> str:
        # If absolute, use as is, otherwise resolve relative to workspace root
        if os.path.isabs(self.local_storage_root):
            return self.local_storage_root
        # The worker/api usually run in a compose context where data/ is at root
        return self.local_storage_root


@lru_cache
def get_settings() -> Settings:
    return Settings()
