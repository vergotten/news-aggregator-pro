# -*- coding: utf-8 -*-
"""
LangChain-backend для AI-агентов.

Альтернативная реализация основных агентов поверх LangChain ChatModels.
Включается переменной окружения AGENT_BACKEND=langchain (см. agent_factory).

Требует: pip install -r requirements-ai.txt (langchain-core, langchain-openai)
"""

from src.application.ai_services.langchain_agents.llm_factory import (
    build_chat_model,
    LANGCHAIN_AVAILABLE,
)
from src.application.ai_services.langchain_agents.base import LangChainAgent
from src.application.ai_services.langchain_agents.agents import (
    LCClassifierAgent,
    LCRelevanceAgent,
    LCSummarizerAgent,
    LCRewriterAgent,
)

__all__ = [
    "LANGCHAIN_AVAILABLE",
    "build_chat_model",
    "LangChainAgent",
    "LCClassifierAgent",
    "LCRelevanceAgent",
    "LCSummarizerAgent",
    "LCRewriterAgent",
]
