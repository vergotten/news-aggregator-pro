# -*- coding: utf-8 -*-
"""
Фабрика агентов с выбором backend'а.

Backend задаётся переменной окружения AGENT_BACKEND:
    legacy    — собственные HTTP-провайдеры (по умолчанию)
    langchain — LangChain ChatModels (требует requirements-ai.txt)

Для агентов без LangChain-версии (или если LangChain не установлен)
автоматически используется legacy-реализация, поэтому переключение
backend'а безопасно в любом окружении.
"""

import logging
import os
from typing import Any

from src.application.ai_services.agents import (
    ClassifierAgent,
    RelevanceAgent,
    SummarizerAgent,
    RewriterAgent,
)

logger = logging.getLogger(__name__)

LEGACY_AGENTS = {
    "classifier": ClassifierAgent,
    "relevance": RelevanceAgent,
    "summarizer": SummarizerAgent,
    "rewriter": RewriterAgent,
}


def get_agent_backend() -> str:
    """Текущий backend агентов: 'legacy' или 'langchain'."""
    backend = os.getenv("AGENT_BACKEND", "legacy").strip().lower()
    if backend not in ("legacy", "langchain"):
        logger.warning(f"Неизвестный AGENT_BACKEND={backend}, используем legacy")
        return "legacy"
    return backend


def _langchain_agents() -> dict:
    """Карта LangChain-агентов (импорт ленивый — зависимость опциональна)."""
    from src.application.ai_services.langchain_agents import (
        LANGCHAIN_AVAILABLE,
        LCClassifierAgent,
        LCRelevanceAgent,
        LCSummarizerAgent,
        LCRewriterAgent,
    )

    if not LANGCHAIN_AVAILABLE:
        raise ImportError("langchain-openai не установлен")

    return {
        "classifier": LCClassifierAgent,
        "relevance": LCRelevanceAgent,
        "summarizer": LCSummarizerAgent,
        "rewriter": LCRewriterAgent,
    }


def create_agent(name: str, **kwargs: Any):
    """
    Создать агента по имени с учётом выбранного backend'а.

    Args:
        name: имя агента (classifier, relevance, summarizer, rewriter, ...)
        kwargs: пробрасываются в конструктор агента

    Returns:
        Экземпляр агента (legacy или LangChain — публичный API одинаков)
    """
    backend = get_agent_backend()

    if backend == "langchain":
        try:
            lc_agents = _langchain_agents()
            if name in lc_agents:
                logger.info(f"[AgentFactory] {name}: backend=langchain")
                return lc_agents[name](**kwargs)
            logger.info(
                f"[AgentFactory] {name}: нет LangChain-версии, используем legacy"
            )
        except ImportError as e:
            logger.warning(
                f"[AgentFactory] LangChain недоступен ({e}), fallback на legacy"
            )

    if name not in LEGACY_AGENTS:
        raise ValueError(
            f"Неизвестный агент: {name}. Доступны: {sorted(LEGACY_AGENTS)}"
        )

    return LEGACY_AGENTS[name](**kwargs)
