# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/cat_commentator_agent.py
# =============================================================================
"""
Агент кошачьего комментатора v2.0

Два режима:
- comment_short() — 1-2 предложения, для Telegram поста (цепляющий)
- comment_long()  — 3-5 предложений, для статьи (развёрнутый)

Сохраняется в:
- article.cat_comment_short → добавляется в telegram_post_text
- article.cat_comment       → добавляется в editorial_rewritten + telegraph_content_html
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
    comment: str = Field(description="Комментарий НейроКота")

    @field_validator('comment')
    @classmethod
    def validate_comment(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("Комментарий слишком короткий")
        if len(v) > 800:
            v = v[:800]
            last_period = v.rfind('.')
            if last_period > 400:
                v = v[:last_period + 1]
        return v


class CatCommentatorAgent(BaseAgent):
    """
    Агент НейроКотΔ v2.0

    Характер:
    - Технически грамотный кот
    - Саркастичный но добрый, без злобы
    - Кошачьи метафоры (мышки, клубки, дрёма, мурчание)
    - Говорит от первого лица
    - Никогда не пересказывает статью
    """

    agent_name = "cat_commentator"
    task_type = TaskType.LIGHT
    MIN_RESPONSE_LENGTH = 10

    # ── Короткий (для Telegram) ──
    SHORT_SYSTEM_PROMPT = """Ты — НейроКотΔ, технически подкованный кот-редактор.
Оставляешь ОЧЕНЬ короткие цепляющие комментарии к статьям.
Цель: заинтриговать читателя, чтобы он кликнул читать дальше.
Формат: строго 1-2 предложения. Без хэштегов, без эмодзи в тексте."""

    SHORT_PROMPT = """Оставь короткий цепляющий комментарий от НейроКота для Telegram поста.

Заголовок: {title}
Тизер: {teaser}

ТРЕБОВАНИЯ:
- СТРОГО 1-2 предложения
- Цепляющий — должен захотеться читать дальше
- От первого лица ("Я, НейроКот...", "Мяукну честно...", "Признаю...")
- Скепсис, ирония или восторг — на выбор
- НЕ пересказывай, НЕ используй "интересно/познавательно/полезно"

ПРИМЕРЫ:
"Я, НейроКот, давно ждал когда кто-то наконец это объяснит нормально."
"Мяукну честно: не думал что FPGA и клавиатура могут быть в одном предложении."
"Признаю — после этого моя мышка смотрит на меня с уважением."

Комментарий НейроКота (1-2 предложения):"""

    # ── Длинный (для статьи) ──
    LONG_SYSTEM_PROMPT = """Ты — НейроКотΔ, технически подкованный кот-редактор канала.
Ты пишешь развёрнутые комментарии к прочитанным статьям.
Читатель уже прочитал статью — дай ему пищу для размышлений.
Без хэштегов, без эмодзи в тексте."""

    LONG_PROMPT = """Напиши развёрнутый комментарий НейроКота к статье.
Читатель уже прочитал статью — прокомментируй по существу.

Заголовок: {title}
Тизер: {teaser}
Начало текста: {content_preview}

ТРЕБОВАНИЯ:
- 3-5 предложений
- От первого лица
- Технически по существу — можешь добавить свои мысли, провести аналогию
- Можно одну кошачью метафору максимум
- Финальная мысль — вывод или открытый вопрос читателю
- НЕ пересказывай, НЕ используй "интересно/познавательно/полезно"

ПРИМЕРЫ:
"Признаю — когда читал про FPGA, хотел свернуться клубком и подремать. Но потом дошло зачем это нужно: полный контроль над железом без прослойки ОС. Это как разница между охотой на мышь через стекло и напрямую. Главный вопрос который остался: когда это станет доступно обычным разработчикам без трёх степеней по электронике?"

Комментарий НейроКота (3-5 предложений):"""

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
        logger.info(f"[INIT] CatCommentatorAgent v2.0: model={self.model}")

    # ─────────────────────────────────────────────
    # Публичные методы
    # ─────────────────────────────────────────────

    def comment_short(self, title: str, teaser: str) -> str:
        """Короткий комментарий для Telegram (1-2 предложения)."""
        prompt = self.SHORT_PROMPT.format(
            title=title,
            teaser=teaser[:300] if teaser else ""
        )
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=CatCommentResult,
                system_prompt=self.SHORT_SYSTEM_PROMPT
            )
            cleaned = self._clean_comment(result.comment)
            logger.info(f"[CatCommentator] Short: {len(cleaned)} chars")
            return cleaned
        except Exception as e:
            logger.error(f"[CatCommentator] Short structured failed: {e}")
            return self._simple_comment(title, teaser, short=True)

    def comment_long(self, title: str, teaser: str, content: str = "") -> str:
        """Развёрнутый комментарий для статьи (3-5 предложений)."""
        content_preview = content[:500] if content else teaser[:300]
        prompt = self.LONG_PROMPT.format(
            title=title,
            teaser=teaser[:300] if teaser else "",
            content_preview=content_preview
        )
        try:
            result = self.generate_structured(
                prompt=prompt,
                output_schema=CatCommentResult,
                system_prompt=self.LONG_SYSTEM_PROMPT
            )
            cleaned = self._clean_comment(result.comment)
            logger.info(f"[CatCommentator] Long: {len(cleaned)} chars")
            return cleaned
        except Exception as e:
            logger.error(f"[CatCommentator] Long structured failed: {e}")
            return self._simple_comment(title, teaser, short=False)

    # обратная совместимость
    def comment(self, title: str, teaser: str) -> str:
        return self.comment_short(title, teaser)

    def process(self, title: str, teaser: str) -> CatCommentResult:
        return CatCommentResult(comment=self.comment_short(title, teaser))

    # ─────────────────────────────────────────────
    # Форматирование
    # ─────────────────────────────────────────────

    def format_for_telegram(self, comment: str) -> str:
        """Для Telegram поста — курсив."""
        if not comment:
            return ""
        return "\n\n🐱 <i>" + comment + "</i>"

    def format_for_article(self, comment: str) -> str:
        """Для editorial_rewritten — markdown курсив."""
        if not comment:
            return ""
        return (
            '\n\n---\n\n'
            '🐱 *НейроКот говорит:*\n\n'
            '*' + comment + '*'
        )

    def format_for_html(self, comment: str) -> str:
        """Для telegraph_content_html — HTML курсив."""
        if not comment:
            return ""
        return (
            '<hr>'
            '<p><em>🐱 НейроКот говорит:</em></p>'
            '<p><em>' + comment + '</em></p>'
        )

    def format_for_dzen(self, comment: str) -> str:
        """Для Дзен RSS — только разрешённые теги."""
        if not comment:
            return ""
        return (
            '<p>—</p>'
            '<p><b>🐱 НейроКот говорит:</b></p>'
            '<p><i>' + comment + '</i></p>'
        )

    # ─────────────────────────────────────────────
    # Внутренние методы
    # ─────────────────────────────────────────────

    def _simple_comment(self, title: str, teaser: str, short: bool = True) -> str:
        """Простой fallback без structured output."""
        if short:
            prompt = (
                "Ты НейроКотΔ, кот-редактор. Напиши 1-2 предложения цепляющего комментария.\n"
                "От первого лица, с иронией. НЕ пересказывай.\n\n"
                f"Заголовок: {title}\nТизер: {teaser[:200]}\n\nКомментарий:"
            )
            max_tokens = 100
        else:
            prompt = (
                "Ты НейроКотΔ, кот-редактор. Напиши 3-5 предложений развёрнутого комментария.\n"
                "От первого лица, технически по существу, с одной аналогией.\n\n"
                f"Заголовок: {title}\nТизер: {teaser[:200]}\n\nКомментарий:"
            )
            max_tokens = 250

        try:
            response = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                min_response_length=10
            )
            return self._clean_comment(response)
        except Exception as e:
            logger.error(f"[CatCommentator] Simple failed: {e}")
            return "Мяу. Прочитал. Думаю." if short else "Прочитал внимательно. Технически грамотно. Буду следить за развитием темы."

    def _clean_comment(self, comment: str) -> str:
        """Очистка комментария от артефактов."""
        if not comment:
            return ""
        cleaned = comment.strip()
        prefixes = ['Комментарий:', 'НейроКот:', 'НейроКотΔ:', 'Comment:', 'Ответ:', 'Мнение:']
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        cleaned = cleaned.strip('"\'`').replace('**', '').replace('__', '')
        for pattern in [r'\bинтересно\b', r'\bпознавательно\b', r'\bполезно\b', r'\bзамечательно\b']:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned