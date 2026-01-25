# -*- coding: utf-8 -*-
"""
AI Агенты для обработки контента.

Доступные агенты:
- BaseAgent: Базовый класс для всех агентов
- ClassifierAgent: Классификация НОВОСТЬ/СТАТЬЯ
- RelevanceAgent: Оценка релевантности для технической аудитории
- SummarizerAgent: Создание тизеров и резюме
- RewriterAgent: Улучшение заголовков
- StyleNormalizerAgent: Нормализация стиля (удаление личных местоимений)
- QualityValidatorAgent: Валидация качества обработки
"""

from src.application.ai_services.agents.base_agent import BaseAgent
from src.application.ai_services.agents.classifier_agent import ClassifierAgent
from src.application.ai_services.agents.relevance_agent import RelevanceAgent
from src.application.ai_services.agents.summarizer_agent import SummarizerAgent
from src.application.ai_services.agents.rewriter_agent import RewriterAgent
from src.application.ai_services.agents.style_normalizer_agent import StyleNormalizerAgent
from src.application.ai_services.agents.quality_validator_agent import QualityValidatorAgent

__all__ = [
    'BaseAgent',
    'ClassifierAgent',
    'RelevanceAgent',
    'SummarizerAgent',
    'RewriterAgent',
    'StyleNormalizerAgent',
    'QualityValidatorAgent',
]
