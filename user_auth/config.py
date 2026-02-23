from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    APP_NAME: str = "Auth Microservice"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    SECRET_KEY: str = "changeme-use-a-real-secret-in-production"

    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DATABASE_URL: str = "postgresql+asyncpg://postgres:balsem@localhost:5432/postgres"

    REDIS_URL: str = "redis://localhost:6379/0"

    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    RATE_LIMIT_LOGIN: int = 10
    RATE_LIMIT_REGISTER: int = 5
    RATE_LIMIT_REFRESH: int = 30

    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = False

    EMAIL_VERIFICATION_REQUIRED: bool = False
    EMAIL_TOKEN_EXPIRE_HOURS: int = 24


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()