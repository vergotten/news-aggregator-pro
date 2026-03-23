# -*- coding: utf-8 -*-
"""
AI Агенты v9.2

Изменения:
- Интеграция с SkiplistService
- Chunking для длинных текстов
- Исключение слабых моделей
- TelegraphFormatterAgent для разметки Telegraph
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
from src.application.ai_services.agents.telegraph_formatter_agent import TelegraphFormatterAgent
from src.application.ai_services.agents.quality_validator_agent import QualityValidatorAgent
from src.application.ai_services.agents.image_prompt_agent import ImagePromptAgent
from src.application.ai_services.agents.image_transform_agent import ImageTransformAgent


__version__ = "9.2"

__all__ = [
    'BaseAgent',
    'TaskType',
    'ClassifierAgent',
    'RelevanceAgent',
    'RewriterAgent',
    'SummarizerAgent',
    'StyleNormalizerAgent',
    'TelegramFormatterAgent',
    'TelegraphFormatterAgent',
    'QualityValidatorAgent',
    'ImagePromptAgent',
    'ImageTransformAgent',
]