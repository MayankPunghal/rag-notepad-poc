"""
RAG Brain Configuration Settings
Central configuration for the multimodal RAG system
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    APP_NAME: str = "RAG Brain"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Keys
    GLM_API_KEY: str = Field(default="", description="Z.AI GLM API key for chatbot")
    GLM_API_URL: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic Claude API key for chatbot")
    ANTHROPIC_API_URL: str = Field(default="https://api.anthropic.com/v1/messages", description="Custom API URL for Anthropic (e.g., Z.AI proxy)")
    CHAT_PROVIDER: str = Field(default="anthropic", description="Chat provider: 'glm' or 'anthropic'")

    # Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    EMBEDDINGS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "embeddings")
    FILES_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "files")
    METADATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "metadata")
    LOGS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    MLFLOW_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "mlflow")

    # Database
    SQLITE_DB_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "metadata" / "rag_brain.db")

    # FAISS
    FAISS_INDEX_TYPE: str = "IndexFlatIP"  # Inner product (cosine similarity with normalized vectors)
    FAISS_TEXT_INDEX_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "embeddings" / "text.index")
    FAISS_IMAGE_INDEX_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "embeddings" / "image.index")

    # Embedding Models
    TEXT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_EMBEDDING_DIM: int = 384
    IMAGE_CAPTIONING_MODEL: str = "microsoft/Florence-2-base"  # CPU-friendly

    # Text Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_SUMMARY_LENGTH: int = 150
    MIN_TAG_SCORE: float = 0.3

    # Retrieval
    TOP_K_RESULTS: int = 10  # Increased to get more chunks
    SIMILARITY_THRESHOLD: float = 0.5

    # RAG Pipeline
    RAG_CONTEXT_LENGTH: int = 8000  # Increased to allow more chunks through
    RAG_MAX_TOKENS: int = 4096
    RAG_TEMPERATURE: float = 0.7

    # MLflow
    MLFLOW_TRACKING_URI: str = Field(default="", description="MLflow tracking server URI")
    MLFLOW_EXPERIMENT_NAME: str = "rag_brain"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # Upload limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_TEXT_EXTENSIONS: List[str] = [".txt", ".md", ".pdf", ".doc", ".docx"]
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

    def ensure_directories(self):
        """Create all necessary directories"""
        dirs = [
            self.DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.FILES_DIR,
            self.METADATA_DIR,
            self.LOGS_DIR,
            self.MLFLOW_DIR,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
