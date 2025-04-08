"""
Application configuration
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SPLADE Content Server"
    PORT: int = 3000

    # SPLADE model settings
    MODEL_DIR: str = "./fine_tuned_splade"
    MAX_LENGTH: int = 512

    # Data storage settings
    DATA_DIR: str = "app/data"

    # Search settings
    DEFAULT_TOP_K: int = 10
    MIN_SCORE_THRESHOLD: float = 0.3

    class Config:
        env_file = ".env"
        env_prefix = "SPLADE_"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
