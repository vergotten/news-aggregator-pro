# -*- coding: utf-8 -*-
"""
AI Агенты v9.1

Изменения:
- Интеграция с SkiplistService
- Chunking для длинных текстов
- Исключение слабых моделей
"""

from src.application.ai_services.agents.base_agent import (
    BaseAgent,
    TaskType,
)
from src.application.ai_services.agents.classifier_agent import ClassifierAgent
from src.application.ai_services.agents.relevance_agent import RelevanceAgent
from src.application.ai_services.agents.rewriter_agent import RewriterAgent
from src.application.ai_services.agents.summarizer_agent import SummarizerAgent
from src.application.ai_services.agents.style_normalizer_agent import StyleNormalizerAgent
from src.application.ai_services.agents.telegram_formatter_agent import TelegramFormatterAgent
from src.application.ai_services.agents.quality_validator_agent import QualityValidatorAgent

__version__ = "9.1"

__all__ = [
    'BaseAgent',
    'TaskType',
    'ClassifierAgent',
    'RelevanceAgent',
    'RewriterAgent',
    'SummarizerAgent',
    'StyleNormalizerAgent',
    'TelegramFormatterAgent',
    'QualityValidatorAgent',
]
