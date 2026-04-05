# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/cat_commentator_agent.py
# =============================================================================
"""
Агент кошачьего комментатора v1.0

НейроКотΔ — маскот канала. Читает заголовок и тизер статьи,
оставляет короткий саркастичный но добрый комментарий от лица кота.

Используется как финальный штрих в пайплайне — добавляет уникальный
голос канала в конец каждой статьи.
"""

import logging
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)


class CatCommentResult(BaseModel):
    """Результат кошачьего комментария."""

    comment: str = Field(description="Комментарий НейроКота 2-3 предложения")

    @field_validator('comment')
    @classmethod
    def validate_comment(cls, v: str) -> str:
        if len(v) < 20:
            raise ValueError("Комментарий слишком короткий")
        if len(v) > 400:
            v = v[:400]
            last_period = v.rfind('.')
            if last_period > 200:
                v = v[:last_period + 1]
        return v


class CatCommentatorAgent(BaseAgent):
    """
    Агент НейроКотΔ — добавляет кошачий комментарий к статье.

    Характер персонажа:
    - Технически грамотный кот
    - Саркастичный но добрый, без злобы
    - Иногда уходит в кошачьи метафоры (мышки, клубки, дрёма)
    - Говорит от первого лица ("Я, НейроКот...")
    - Короткий и ёмкий, не пересказывает статью
    - Может выразить скепсис, восторг или иронию
    """

    agent_name = "cat_commentator"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 20

    SYSTEM_PROMPT = """Ты — НейроКотΔ, технически подкованный кот-редактор.
Ты читаешь технические статьи и оставляешь короткие комментарии от своего имени.

Твой характер:
- Саркастичный но добрый, без злобы и агрессии
- Технически грамотный — понимаешь о чём пишешь
- Иногда используешь кошачьи метафоры (мышки, клубки, дрёма, мурчание)
- Говоришь от первого лица
- Никогда не пересказываешь статью — только своё мнение

Формат: 2-3 предложения, не больше. Без хэштегов, без эмодзи в тексте."""

    COMMENT_PROMPT = """Прочитай заголовок и тизер статьи, оставь короткий комментарий от лица НейроКота.

Заголовок: {title}
Тизер: {teaser}

ТРЕБОВАНИЯ:
- 2-3 предложения максимум
- От первого лица ("Я, НейроКот...", "Признаю...", "Мяукну честно...")
- Своё мнение — восторг, скепсис, ирония или удивление
- Можно одну кошачью метафору
- НЕ пересказывай статью
- НЕ используй слова "интересно", "познавательно", "полезно"

ПРИМЕРЫ хорошего комментария:
"Признаю — когда читал про FPGA, хотел свернуться клубком и подремать. Но потом дошло зачем это нужно, и стало любопытно. Слежу за развитием."

"Я, НейроКот, видел немало статей про LLM-агентов. Но вот чтобы один заменил целого тестировщика — это уже что-то новенькое. Мышку им точно не доверю."

"Мяукну честно: хэш-таблицы с постоянным временем поиска — это миф который я давно подозревал. Автор наконец-то расставил всё по местам."

Напиши комментарий НейроКота:"""

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            config: Optional[ModelsConfig] = None,
            **kwargs
    ):
        super().__init__(
            llm_provider=llm_provider,
            config=config,
            max_retries=2,
            retry_delay=1.0
        )
        logger.info(f"[INIT] CatCommentatorAgent v1.0: model={self.model}")

    def comment(self, title: str, teaser: str) -> str:
        """Получить комментарий НейроКота -> строка."""
        result = self.process(title, teaser)
        return result.comment

    def process(self, title: str, teaser: str) -> CatCommentResult:
        """Главный метод."""
        prompt = self.COMMENT_PROMPT.format(
            title=title,
            teaser=teaser[:300] if teaser else ""
        )

        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=CatCommentResult,
                system_prompt=self.SYSTEM_PROMPT
            )
            result.comment = self._clean_comment(result.comment)
            logger.info(f"[CatCommentator] Comment: {len(result.comment)} chars")
            return result

        except Exception as e:
            logger.error(f"[CatCommentator] Structured failed: {e}")
            return self._comment_simple(title, teaser)

    def _comment_simple(self, title: str, teaser: str) -> CatCommentResult:
        """Простой fallback."""
        prompt = (
            f"Ты — НейроКотΔ, саркастичный кот-редактор.\n"
            f"Оставь короткий комментарий (2-3 предложения) к статье от своего имени.\n"
            f"Не пересказывай, выскажи своё мнение с иронией.\n\n"
            f"Заголовок: {title}\n"
            f"Тизер: {teaser[:200]}\n\n"
            f"Комментарий НейроКота:"
        )

        try:
            response = self.generate(
                prompt=prompt,
                max_tokens=150,
                min_response_length=20
            )
            comment = self._clean_comment(response)
            return CatCommentResult(comment=comment)

        except Exception as e:
            logger.error(f"[CatCommentator] Simple failed: {e}")
            return CatCommentResult(
                comment="Мяу. Прочитал. Думаю."
            )

    def _clean_comment(self, comment: str) -> str:
        """Очистка комментария."""
        if not comment:
            return ""

        cleaned = comment.strip()

        # Убрать префиксы
        prefixes = [
            'Комментарий:', 'НейроКот:', 'НейроКотΔ:',
            'Comment:', 'Ответ:', 'Мнение:'
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # Убрать markdown
        cleaned = cleaned.strip('"').strip("'").strip('`')
        cleaned = cleaned.replace('**', '').replace('__', '')

        # Убрать запрещённые слова
        forbidden_patterns = [
            r'\bинтересно\b',
            r'\bпознавательно\b',
            r'\bполезно\b',
            r'\bзамечательно\b',
        ]
        for pattern in forbidden_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        # Длина
        if len(cleaned) > 400:
            sentences = cleaned.split('.')
            cleaned = '. '.join(sentences[:3]).strip()
            if not cleaned.endswith('.'):
                cleaned += '.'

        return cleaned

    def format_for_article(self, comment: str) -> str:
        """
        Форматировать комментарий для вставки в статью.
        Возвращает HTML блок для сайта и RSS.
        """
        if not comment:
            return ""
        return (
            '\n\n---\n\n'
            '🐱 **НейроКот говорит:**\n\n'
            f'*{comment}*'
        )

    def format_for_dzen(self, comment: str) -> str:
        """
        Форматировать для Дзен RSS (только разрешённые теги).
        """
        if not comment:
            return ""
        return (
            '<p>—</p>'
            '<p><b>🐱 НейроКот говорит:</b></p>'
            f'<p><i>{comment}</i></p>'
        )