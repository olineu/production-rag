from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
