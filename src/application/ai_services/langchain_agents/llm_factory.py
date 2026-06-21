# -*- coding: utf-8 -*-
"""
Фабрика LangChain chat-моделей поверх существующей конфигурации проекта.

Все провайдеры подключаются через их OpenAI-совместимые endpoints,
поэтому достаточно одной зависимости — langchain-openai:

    openrouter → https://openrouter.ai/api/v1
    groq       → https://api.groq.com/openai/v1
    google     → https://generativelanguage.googleapis.com/v1beta/openai/
    ollama     → {OLLAMA_BASE_URL}/v1

Конфигурация (модель, температура, max_tokens, ключи) берётся из того же
ModelsConfig / LLMConfig, что и у legacy-агентов — единый источник правды.
"""

import logging
import os
from typing import Optional

from src.infrastructure.ai.llm_provider import LLMConfig, LLMProviderType

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    ChatOpenAI = None
    LANGCHAIN_AVAILABLE = False

# Gemini публикует OpenAI-совместимый endpoint — отдельный SDK не нужен
GOOGLE_OPENAI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _resolve_endpoint(config: LLMConfig) -> tuple[str, str]:
    """Вернуть (base_url, api_key) для OpenAI-совместимого endpoint провайдера."""
    provider = config.provider

    if provider == LLMProviderType.OPENROUTER:
        base_url = (
            config.base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or LLMConfig.OPENROUTER_DEFAULT_URL
        )
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY") or ""
    elif provider == LLMProviderType.GROQ:
        base_url = config.base_url or LLMConfig.GROQ_DEFAULT_URL
        api_key = config.api_key or os.getenv("GROQ_API_KEY") or ""
    elif provider == LLMProviderType.GOOGLE:
        # config.base_url хранит нативный REST URL Gemini — для LangChain
        # всегда используем OpenAI-совместимый endpoint
        base_url = GOOGLE_OPENAI_COMPAT_URL
        api_key = (
            config.api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
    elif provider == LLMProviderType.OLLAMA:
        native_url = (
            config.base_url
            or os.getenv("OLLAMA_BASE_URL")
            or LLMConfig.OLLAMA_DEFAULT_URL
        )
        base_url = native_url.rstrip("/") + "/v1"
        api_key = "ollama"  # Ollama игнорирует ключ, но SDK требует непустой
    else:
        raise ValueError(f"Неизвестный провайдер для LangChain: {provider}")

    return base_url, api_key


def build_chat_model(config: LLMConfig, temperature: Optional[float] = None):
    """
    Построить LangChain ChatModel из LLMConfig проекта.

    Raises:
        ImportError: если langchain-openai не установлен
            (pip install -r requirements-ai.txt)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-openai не установлен. "
            "Установите AI-зависимости: pip install -r requirements-ai.txt"
        )

    base_url, api_key = _resolve_endpoint(config)

    model = ChatOpenAI(
        model=config.model,
        temperature=temperature if temperature is not None else config.temperature,
        max_tokens=config.max_tokens,
        api_key=api_key,
        base_url=base_url,
        timeout=180,
        max_retries=2,
    )

    logger.info(
        f"[LangChain] ChatModel: provider={config.provider.value}, "
        f"model={config.model}, base_url={base_url}"
    )
    return model
