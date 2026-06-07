# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/rewriter_agent.py
# =============================================================================
"""
Агент переписывания заголовков v9.2
"""

import logging
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class TitleResult(BaseModel):
    """Результат переписывания."""
    improved_title: str = Field(description="Улучшенный заголовок")
    original_issues: list[str] = Field(default_factory=list)
    improvements_made: list[str] = Field(default_factory=list)

    # Добавляем дополнительные поля для совместимости с orchestrator
    reasoning: Optional[str] = Field(default=None, description="Объяснение изменений")
    original_length: Optional[int] = Field(default=None, description="Длина оригинального заголовка")
    new_length: Optional[int] = Field(default=None, description="Длина нового заголовка")

    @field_validator('improved_title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = v.strip().strip('"').strip("'").strip('`')
        if v.endswith('.'):
            v = v[:-1]
        v = v.replace('!', '').replace('**', '').replace('*', '')
        return v


class RewriterAgent(BaseAgent):
    """Агент улучшения заголовков v9.2"""

    agent_name = "rewriter"
    task_type = TaskType.MEDIUM
    MIN_RESPONSE_LENGTH = 15

    # Исключенные модели - возвращают пустые ответы
    EXCLUDED_MODELS = [
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron",
        "meta-llama/llama-3.2-3b",
        "meta-llama/llama-3.1-8b",
        "openai/gpt-oss-120b:free",
        "google/gemma-2-9b-it:free",
    ]

    SYSTEM_PROMPT = """Ты - редактор. Улучшай заголовки: информативно и профессионально.
Сохраняй язык. Отвечай ТОЛЬКО заголовком, без пояснений."""

    REWRITE_PROMPT = """Улучши заголовок.

ПРАВИЛА:
- 40-80 символов
- Информативный
- БЕЗ "Как я...", "Мой опыт..."
- БЕЗ восклицаний

ОРИГИНАЛ: {title}

Напиши ТОЛЬКО новый заголовок:"""

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            config: Optional[ModelsConfig] = None,
            **kwargs
    ):
        super().__init__(llm_provider=llm_provider, config=config, max_retries=3, retry_delay=2.0)
        logger.info(f"[INIT] RewriterAgent v9.2: model={self.model}")

    def rewrite_title(self, title: str, content: str = "") -> str:
        """Переписать заголовок, возвращая только текст."""
        result = self.process(title, content)
        return result.improved_title

    def rewrite_with_details(self, title: str, content: str = "", max_length: int = 100) -> TitleResult:
        """
        Переписать заголовок с подробной информацией.

        Args:
            title: Оригинальный заголовок
            content: Содержание статьи
            max_length: Максимальная длина нового заголовка

        Returns:
            Объект TitleResult с новым заголовком и дополнительной информацией
        """
        try:
            # Используем существующий метод process для получения результата
            result = self.process(title, content)

            # Анализируем различия между оригиналом и улучшенной версией
            reasoning = "Заголовок улучшен с учетом требований к информативности и читаемости"

            # Определяем конкретные улучшения
            if result.improved_title != title:
                if len(result.improved_title) > len(title):
                    reasoning = "Заголовок сделан более информативным и подробным"
                elif len(result.improved_title) < len(title):
                    reasoning = "Заголовок сделан более лаконичным и сфокусированным"
                else:
                    reasoning = "Заголовок переформулирован для лучшего восприятия"

            # Создаем новый объект TitleResult с дополнительными полями
            return TitleResult(
                improved_title=result.improved_title,
                original_issues=result.original_issues,
                improvements_made=result.improvements_made,
                reasoning=reasoning,
                original_length=len(title),
                new_length=len(result.improved_title)
            )
        except Exception as e:
            logger.error(f"Ошибка в методе rewrite_with_details: {e}")
            # Возвращаем базовый ответ в случае ошибки
            return TitleResult(
                improved_title=title,
                original_issues=[f"Произошла ошибка: {str(e)}"],
                improvements_made=[],
                reasoning=f"Произошла ошибка: {str(e)}",
                original_length=len(title),
                new_length=len(title)
            )

    def process(self, title: str, content: str = "") -> TitleResult:
        """Обработать заголовок и вернуть результат."""
        prompt = self.REWRITE_PROMPT.format(title=title)

        try:
            response = self.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=80,
                min_response_length=10
            )

            improved = self._extract_title(response, title)
            improved = self._validate_title(improved, title)

            # Анализируем различия между оригиналом и улучшенной версией
            improvements = []
            original_issues = []

            if improved and improved != title and len(improved) >= 15:
                # Определяем улучшения
                if len(improved) > len(title):
                    improvements.append("Более информативный")
                if not any(char in title for char in ['!', '?']) and any(char in improved for char in ['!', '?']):
                    improvements.append("Добавлена эмоциональная окраска")
                if any(word in improved.lower() for word in ['как', 'почему', 'что']):
                    improvements.append("Добавлен вопросительный элемент")
                if not improvements:
                    improvements.append("Улучшена структура")

                logger.info(f"[Rewriter] '{title[:30]}...' -> '{improved[:30]}...'")
                return TitleResult(
                    improved_title=improved,
                    improvements_made=improvements
                )
            else:
                if not improved:
                    original_issues.append("Не удалось сгенерировать улучшенный заголовок")
                elif len(improved) < 15:
                    original_issues.append("Слишком короткий заголовок")
                elif improved == title:
                    original_issues.append("Заголовок не был улучшен")

                logger.warning(f"[Rewriter] Using original: {', '.join(original_issues)}")
                return TitleResult(
                    improved_title=title,
                    original_issues=original_issues
                )

        except Exception as e:
            logger.error(f"[Rewriter] Error: {e}")
            return TitleResult(
                improved_title=title,
                original_issues=[str(e)]
            )

    def _extract_title(self, response: str, original: str) -> str:
        """Извлечь заголовок из ответа модели."""
        if not response:
            return ""

        text = response.strip()
        text = re.sub(r'^```.*?\n', '', text)
        text = re.sub(r'\n```$', '', text)

        prefixes = [
            r'^Improved\s*(?:title)?:\s*',
            r'^Улучшенный\s*(?:заголовок)?:\s*',
            r'^Заголовок:\s*',
            r'^Title:\s*',
            r'^->\s*',
        ]
        for pattern in prefixes:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            text = lines[0]

        for quote in ['"', "'", '`', '«', '»']:
            text = text.strip(quote)

        text = text.replace('**', '')
        return text.strip()

    def _validate_title(self, improved: str, original: str) -> str:
        """Валидировать и корректировать улучшенный заголовок."""
        if not improved:
            return original

        improved = improved.strip().strip('"').strip("'")
        improved = improved.replace('**', '').replace('*', '')

        if improved.endswith('.'):
            improved = improved[:-1]
        improved = improved.replace('!', '')

        if len(improved) < 15:
            return original

        if len(improved) > 120:
            improved = improved[:120].rsplit(' ', 1)[0]

        return improved