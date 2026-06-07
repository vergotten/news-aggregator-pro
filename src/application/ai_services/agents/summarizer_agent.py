# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/summarizer_agent.py
# =============================================================================
"""
Агент создания тизеров v8.0

Изменения v8:
- Retry при пустом ответе
- Исключение слабых моделей
- Улучшенная очистка тизера
"""

import logging
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class SummaryResult(BaseModel):
    """Результат суммаризации."""

    teaser: str = Field(description="Тизер 2-4 предложения")
    key_points: list[str] = Field(default_factory=list)
    main_topic: str = Field(default="")

    @field_validator('teaser')
    @classmethod
    def validate_teaser(cls, v: str) -> str:
        if len(v) < 50:
            raise ValueError("Тизер слишком короткий")
        if len(v) > 500:
            v = v[:500]
            last_period = v.rfind('.')
            if last_period > 300:
                v = v[:last_period + 1]
        return v


class SummarizerAgent(BaseAgent):
    """Агент создания тизеров и резюме."""

    agent_name = "summarizer"
    task_type = TaskType.MEDIUM
    MIN_RESPONSE_LENGTH = 50
    
    EXCLUDED_MODELS = [
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "openai/gpt-oss-120b:free",
    ]

    SYSTEM_PROMPT = """Ты - редактор технических новостей.
Создавай краткие информативные тизеры.
Пиши САМОСТОЯТЕЛЬНЫЙ текст, НЕ пересказ.
Язык: тот же что у статьи."""

    SUMMARY_PROMPT = """Создай тизер для ленты новостей.

ТРЕБОВАНИЯ:
- 2-4 предложения, 150-300 символов
- Самостоятельный текст (НЕ пересказ)
- БЕЗ "Автор рассказывает...", "В статье говорится..."
- БЕЗ местоимений "я", "мы", "мой"
- Безличные конструкции

ПРИМЕР:
Оригинал: "Привет! Я расскажу как мы внедрили микросервисы..."
Тизер: "Представлен опыт миграции на микросервисную архитектуру. Рассмотрены решения при переходе от монолита."

Заголовок: {title}
Текст: {content}

Напиши тизер:"""

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            config: Optional[ModelsConfig] = None,
            **kwargs
    ):
        super().__init__(
            llm_provider=llm_provider, 
            config=config,
            max_retries=3,
            retry_delay=2.0
        )
        logger.info(f"[INIT] SummarizerAgent v8: model={self.model}")

    def summarize(self, title: str, content: str) -> str:
        """Создать тизер -> строка."""
        result = self.process(title, content)
        return result.teaser

    def summarize_with_details(self, title: str, content: str) -> SummaryResult:
        """Создать тизер с деталями."""
        return self.process(title, content)

    def process(self, title: str, content: str) -> SummaryResult:
        """Главный метод."""
        prompt = self.SUMMARY_PROMPT.format(
            title=title,
            content=content[:1200]
        )

        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=SummaryResult,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            result.teaser = self._clean_teaser(result.teaser)

            logger.info(f"[Summarizer] Teaser: {len(result.teaser)} chars, {len(result.key_points)} points")
            return result

        except Exception as e:
            logger.error(f"[Summarizer] Structured failed: {e}")
            return self._summarize_simple(title, content)

    def _summarize_simple(self, title: str, content: str) -> SummaryResult:
        """Простой fallback."""
        prompt = f"""Напиши тизер (2-4 предложения) для статьи.
БЕЗ "автор рассказывает", БЕЗ "я", "мы".
Самостоятельный текст.

Заголовок: {title}
Текст: {content[:800]}

Тизер:"""

        try:
            response = self.generate(
                prompt=prompt, 
                max_tokens=250,
                min_response_length=50
            )
            teaser = self._clean_teaser(response)

            return SummaryResult(
                teaser=teaser,
                key_points=[],
                main_topic=""
            )

        except Exception as e:
            logger.error(f"[Summarizer] Simple failed: {e}")
            # Крайний fallback
            sentences = content.split('.')[:2]
            fallback = '. '.join(s.strip() for s in sentences if s.strip()) + '.'

            return SummaryResult(
                teaser=fallback[:300],
                key_points=[],
                main_topic=""
            )

    def _clean_teaser(self, teaser: str) -> str:
        """Очистка тизера."""
        if not teaser:
            return ""
            
        cleaned = teaser.strip()

        # Убрать префиксы
        prefixes = [
            'Teaser:', 'Summary:', 'Тизер:', 'Краткое описание:',
            '**Teaser:**', '**Тизер:**', 'Вот тизер:',
            'Краткое содержание:', 'Резюме:'
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # Убрать кавычки и markdown
        cleaned = cleaned.strip('"').strip("'").strip('`')
        cleaned = cleaned.replace('**', '')

        # Убрать запрещенные паттерны
        forbidden = [
            r'^Автор\s+(?:рассказывает|описывает|делится)',
            r'^В\s+(?:статье|материале)\s+(?:говорится|рассказывается)',
            r'^Статья\s+(?:описывает|рассказывает)',
        ]
        for pattern in forbidden:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        # Проверка длины
        if len(cleaned) < 50:
            logger.warning(f"[Summarizer] Teaser too short: {len(cleaned)}")
        elif len(cleaned) > 500:
            sentences = cleaned.split('.')
            cleaned = '. '.join(sentences[:4]) + '.'
            if len(cleaned) > 500:
                cleaned = cleaned[:497] + '...'

        return cleaned

    def batch_summarize(self, items: list[tuple[str, str]]) -> list[SummaryResult]:
        """Batch суммаризация."""
        results = []
        for title, content in items:
            try:
                result = self.process(title, content)
                results.append(result)
            except Exception as e:
                logger.error(f"[Summarizer] Batch error '{title[:30]}': {e}")
                results.append(SummaryResult(
                    teaser=f"Суммаризация недоступна: {title[:50]}",
                    key_points=[],
                    main_topic=""
                ))
        return results
