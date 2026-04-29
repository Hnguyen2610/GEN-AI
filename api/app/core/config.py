from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("api/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    api_prefix: str = "/v1"
    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:15432/rag_learning"
    redis_url: str = "redis://127.0.0.1:6379/0"
    cors_origins_raw: str = "http://localhost:3000,http://127.0.0.1:3000"
    auth_mode: str = "supabase"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    storage_backend: str = "local"
    local_storage_root: str = "data/uploads"
    supabase_storage_bucket: str = "datasets"
    max_upload_size_mb: float = 20
    allowed_dataset_extensions_raw: str = "xlsx,xls,csv"
    allowed_knowledge_extensions_raw: str = "pdf,docx,txt,md"
    auto_create_schema: bool = True
    
    rag_top_k: int = 5
    rag_max_distance: float = 0.6
    rag_candidate_k: int = 20
    rag_distance_margin: float = 0.08
    rag_max_citation_assets: int = 2
    gemini_api_key: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_num_ctx: int = 4096

    @property
    def storage_local_path(self) -> str:
        return self.local_storage_root

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    @property
    def allowed_dataset_extensions(self) -> set[str]:
        return {
            extension.strip().lower().lstrip(".")
            for extension in self.allowed_dataset_extensions_raw.split(",")
            if extension.strip()
        }

    @property
    def allowed_knowledge_extensions(self) -> set[str]:
        return {
            extension.strip().lower().lstrip(".")
            for extension in self.allowed_knowledge_extensions_raw.split(",")
            if extension.strip()
        }

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
