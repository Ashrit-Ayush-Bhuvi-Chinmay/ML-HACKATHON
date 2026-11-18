
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    APP_NAME: str = "PathSnap Backend"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Only explicitly approved frontend origins
    ALLOWED_ORIGINS: str = "http://localhost:5173"

    # Request rate limits
    RATE_LIMIT: str = "60/minute"
    DISCOVERY_SECRET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]


settings = Settings()
