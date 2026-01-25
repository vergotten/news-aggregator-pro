# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/infrastructure/config/settings.py
# =============================================================================
"""
Application Settings - Infrastructure Layer.

Загружает настройки из переменных окружения и .env файла.
Поддерживает все переменные из docker-compose и .env.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Настройки приложения.
    
    Все переменные загружаются из .env файла или переменных окружения.
    """
    
    # ==========================================================================
    # Database (PostgreSQL)
    # ==========================================================================
    database_url: str = "postgresql://newsaggregator:changeme123@postgres:5432/news_aggregator"
    postgres_user: str = "newsaggregator"
    postgres_password: str = "changeme123"
    postgres_db: str = "news_aggregator"
    postgres_port: str = "5433"
    
    # ==========================================================================
    # Redis
    # ==========================================================================
    redis_url: str = "redis://redis:6379/0"
    
    # ==========================================================================
    # Qdrant (Vector DB)
    # ==========================================================================
    qdrant_url: str = "http://qdrant:6333"
    qdrant_host: str = "qdrant"
    qdrant_port: str = "6333"
    
    # ==========================================================================
    # Embedding
    # ==========================================================================
    embedding_model: str = "nomic-embed-text"
    
    # ==========================================================================
    # Ollama (локальный LLM)
    # ==========================================================================
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "mistral:latest"
    
    # ==========================================================================
    # LLM Provider Settings (для LangChain миграции)
    # ==========================================================================
    llm_provider: str = "ollama"  # ollama или openrouter
    llm_profile: str = "balanced"  # balanced, fast, cloud_balanced, cloud_quality, free_openrouter
    openrouter_api_key: Optional[str] = None  # API ключ для OpenRouter
    
    # ==========================================================================
    # Telegram Userbot (парсинг каналов)
    # ==========================================================================
    telegram_api_id: str = "your_api_id_here"
    telegram_api_hash: str = "your_api_hash_here"
    telegram_phone: str = ""
    
    # ==========================================================================
    # Telegram Bot (публикация)
    # ==========================================================================
    telegram_bot_token: str = "your_bot_token_here"
    telegram_channel: str = "@your_channel"
    telegram_min_relevance: str = "7.0"
    telegram_include_images: str = "true"
    telegram_post_delay: str = "60"
    
    # ==========================================================================
    # Directus CMS
    # ==========================================================================
    directus_port: str = "8055"
    directus_admin_email: str = "admin@example.com"
    directus_admin_password: str = "admin"
    directus_key: str = "replace-with-random-value-in-production"
    directus_secret: str = "replace-with-another-random-value"
    
    # ==========================================================================
    # App Settings
    # ==========================================================================
    debug: bool = False
    log_level: str = "INFO"
    pythonunbuffered: str = "1"
    
    # ==========================================================================
    # Pydantic Settings Config
    # ==========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Игнорировать неизвестные переменные
        case_sensitive=False,  # Регистронезависимые имена
    )
    
    # ==========================================================================
    # Вспомогательные методы
    # ==========================================================================
    
    def get_ollama_url(self) -> str:
        """Получить URL Ollama с учётом переменных окружения."""
        return self.ollama_base_url
    
    def get_openrouter_key(self) -> Optional[str]:
        """Получить API ключ OpenRouter."""
        return self.openrouter_api_key
    
    def is_cloud_provider(self) -> bool:
        """Проверить, используется ли облачный провайдер."""
        return self.llm_provider.lower() in ("openrouter", "openai", "anthropic")
    
    def get_telegram_min_relevance_float(self) -> float:
        """Получить минимальную релевантность как float."""
        try:
            return float(self.telegram_min_relevance)
        except ValueError:
            return 5.0


@lru_cache()
def get_settings() -> Settings:
    """
    Получить закэшированные настройки.
    
    Использует lru_cache - настройки загружаются один раз при старте.
    
    Возвращает:
        Экземпляр Settings
    """
    return Settings()