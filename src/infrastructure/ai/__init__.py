# -*- coding: utf-8 -*-
"""
AI Infrastructure - абстракции для работы с LLM провайдерами.

Компоненты:
- LLMProvider: Базовый класс для LLM провайдеров
- OllamaProvider: Провайдер для локального Ollama
- OpenRouterProvider: Провайдер для облачного OpenRouter API
- LLMProviderFactory: Фабрика для создания провайдеров
"""

from src.infrastructure.ai.llm_provider import (
    LLMProvider,
    LLMProviderType,
    LLMConfig,
    OllamaProvider,
    OpenRouterProvider,
    LLMProviderFactory,
    get_llm_provider,
)

__all__ = [
    'LLMProvider',
    'LLMProviderType',
    'LLMConfig',
    'OllamaProvider',
    'OpenRouterProvider',
    'LLMProviderFactory',
    'get_llm_provider',
]
