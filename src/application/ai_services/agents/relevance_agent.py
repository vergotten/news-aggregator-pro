# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/relevance_agent.py
# =============================================================================
"""
Агент оценки релевантности v8.0
"""

import logging
import re
from typing import Optional
from pydantic import BaseModel, Field

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class RelevanceResult(BaseModel):
    """Результат оценки релевантности."""

    score: int = Field(ge=0, le=10, description="Оценка 0-10")
    reason: str = Field(description="Обоснование")
    categories: list[str] = Field(default_factory=list)
    target_audience: str = Field(default="general")


class RelevanceAgent(BaseAgent):
    """Агент оценки релевантности для технической аудитории."""

    agent_name = "relevance"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 5
    
    EXCLUDED_MODELS = [
        "openai/gpt-oss-120b:free",
    ]

    SYSTEM_PROMPT = """Ты - оценщик контента для технического портала.
Аудитория: разработчики, DevOps, ML-инженеры.
Оценивай объективно. Отвечай на русском."""

    SCORING_PROMPT = """Оцени релевантность для технической аудитории (0-10).

ШКАЛА:
9-10: Критически важно (прорывные технологии, важные релизы)
7-8: Очень интересно (новые подходы, полезные инструменты)
5-6: Умеренно интересно (стандартные туториалы)
3-4: Мало интересно (базовые темы, маркетинг)
1-2: Нерелевантно
0: Полный офтопик

Заголовок: {title}
Теги: {tags}
Текст: {content}

Оцени и объясни."""

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            config: Optional[ModelsConfig] = None,
            **kwargs
    ):
        super().__init__(llm_provider=llm_provider, config=config)
        logger.info(f"[INIT] RelevanceAgent v8: model={self.model}")

    def score(self, title: str, content: str, tags: Optional[list[str]] = None) -> tuple[int, str]:
        """Оценить -> (score, reason)."""
        result = self.process(title, content, tags)
        return (result.score, result.reason)

    def score_with_details(self, title: str, content: str, tags: Optional[list[str]] = None) -> RelevanceResult:
        """Оценить с деталями."""
        return self.process(title, content, tags)

    def process(self, title: str, content: str, tags: Optional[list[str]] = None) -> RelevanceResult:
        """Главный метод оценки."""
        tags_str = ", ".join(tags[:5]) if tags else "нет тегов"

        prompt = self.SCORING_PROMPT.format(
            title=title,
            tags=tags_str,
            content=content[:600]
        )

        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=RelevanceResult,
                system_prompt=self.SYSTEM_PROMPT
            )

            logger.info(f"[Relevance] {result.score}/10 - {result.reason[:50]}...")
            return result

        except Exception as e:
            logger.error(f"[Relevance] Structured failed: {e}")
            return self._score_simple(title, content, tags_str)

    def _score_simple(self, title: str, content: str, tags_str: str) -> RelevanceResult:
        """Простой fallback."""
        prompt = f"""Оцени релевантность для технической аудитории.

Заголовок: {title}
Теги: {tags_str}
Текст: {content[:500]}

Ответь:
Score: [0-10]
Reason: [причина]"""

        try:
            response = self.generate(prompt=prompt, max_tokens=200, min_response_length=10)
            score, reason = self._parse_response(response)

            return RelevanceResult(
                score=score,
                reason=reason,
                categories=[],
                target_audience="general"
            )

        except Exception as e:
            logger.error(f"[Relevance] Simple failed: {e}")
            return RelevanceResult(
                score=5,
                reason="Ошибка оценки, значение по умолчанию",
                categories=[],
                target_audience="general"
            )

    def _parse_response(self, response: str) -> tuple[int, str]:
        """Парсинг ответа."""
        score = 5
        reason = "Умеренная релевантность"

        for line in response.split('\n'):
            line = line.strip()

            if any(kw in line.lower() for kw in ['score:', 'оценка:', 'балл:']):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    score = min(10, max(0, int(numbers[0])))

            elif any(kw in line.lower() for kw in ['reason:', 'причина:']):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    reason = parts[1].strip()

        return (score, reason)

    def filter_by_relevance(
            self,
            items: list[tuple[str, str, list[str]]],
            min_score: int = 5
    ) -> list[tuple[int, RelevanceResult]]:
        """Фильтрация по релевантности."""
        results = []
        for i, (title, content, tags) in enumerate(items):
            try:
                result = self.process(title, content, tags)
                if result.score >= min_score:
                    results.append((i, result))
            except Exception as e:
                logger.error(f"[Relevance] Filter error {i}: {e}")

        results.sort(key=lambda x: x[1].score, reverse=True)
        return results
