# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/agents/telegram_formatter_agent.py
# =============================================================================
"""
Агент форматирования для Telegram v9.0

Формирует короткий пост для Telegram-канала:
- Заголовок
- Тизер (выжимка важного, 2-4 предложения)
- Ссылка "Читать полностью" → Telegraph
- Ссылка на оригинал
- Хештеги

Полная версия статьи (editorial_rewritten) публикуется на Telegraph
через TelegraphPublisher. Этот агент готовит только Telegram-пост.
"""

import logging
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from src.application.ai_services.agents.base_agent import BaseAgent, TaskType
from src.infrastructure.ai.llm_provider import LLMProvider
from src.config.models_config import ModelsConfig

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LENGTH = 4096
TELEGRAM_OPTIMAL_LENGTH = 1500  # Короткие посты — тизер + ссылка


class TelegramPost(BaseModel):
    """Пост для Telegram-канала."""
    text: str = Field(description="Текст поста (HTML)")
    format_type: Literal["html"] = Field(default="html")
    preview_mode: bool = Field(default=True, description="Это превью, полная версия на Telegraph")
    telegraph_needed: bool = Field(default=True, description="Нужна публикация на Telegraph")
    telegraph_content: Optional[str] = Field(
        default=None,
        description="Полный контент для Telegraph (plain text)"
    )
    telegraph_url: Optional[str] = Field(
        default=None,
        description="URL страницы на Telegraph (заполняется позже)"
    )
    hashtags: list[str] = Field(default_factory=list)
    cover_image: Optional[str] = Field(default=None)
    all_images: list[str] = Field(default_factory=list)

    @field_validator('text')
    @classmethod
    def validate_length(cls, v: str) -> str:
        if len(v) > TELEGRAM_MAX_LENGTH:
            raise ValueError(f"Пост слишком длинный: {len(v)}")
        return v


class TelegramFormatterAgent(BaseAgent):
    """
    Агент форматирования для Telegram v9.0

    Всегда создаёт короткий пост-тизер для Telegram.
    Полная версия (editorial_rewritten) идёт на Telegraph.

    Вызов из оркестратора:
        format_for_telegram(title, content, source_url, tags, images)
        - content используется для извлечения тизера (первые абзацы)
        - telegraph_content = content (полная версия для Telegraph)

    Вызов из pipeline/publisher:
        format_for_telegram(title, content, teaser=..., telegraph_url=...)
        - teaser подставляется напрямую
        - telegraph_url уже готов
    """

    agent_name = "telegram_formatter"
    task_type = TaskType.LIGHT

    def __init__(
            self,
            llm_provider: Optional[LLMProvider] = None,
            config: Optional[ModelsConfig] = None,
            default_author: str = "TechNews",
            add_source_link: bool = True
    ):
        super().__init__(llm_provider=llm_provider, config=config)
        self.default_author = default_author
        self.add_source_link = add_source_link
        logger.info("[INIT] TelegramFormatterAgent v9")

    def format_for_telegram(
            self,
            title: str,
            content: str,
            source_url: Optional[str] = None,
            tags: Optional[list[str]] = None,
            author: Optional[str] = None,
            images: Optional[list[str]] = None,
            teaser: Optional[str] = None,
            telegraph_url: Optional[str] = None,
            source_name: Optional[str] = None,
    ) -> TelegramPost:
        """
        Сформировать пост для Telegram.

        Всегда создаёт короткий тизер + ссылку на Telegraph.
        Полная версия (content) сохраняется в telegraph_content.

        Args:
            title: Заголовок (editorial_title или title)
            content: Полный текст (editorial_rewritten) — идёт на Telegraph
            source_url: Ссылка на оригинал
            tags: Теги/хабы
            author: Автор
            images: Изображения
            teaser: Готовый тизер (от SummarizerAgent). Если нет — берём из content
            telegraph_url: URL Telegraph (если уже создан)
            source_name: Название источника

        Returns:
            TelegramPost
        """
        hashtags = self._make_hashtags(tags or [])
        cover_image = images[0] if images else None

        # Тизер: из параметра или извлекаем из content
        post_teaser = teaser or self._extract_teaser(content)

        # Telegram пост: заголовок + тизер + ссылка
        post_text = self._build_telegram_post(
            title=title,
            teaser=post_teaser,
            telegraph_url=telegraph_url,
            source_url=source_url,
            source_name=source_name,
            hashtags=hashtags,
        )

        # Полный контент для Telegraph (plain text)
        telegraph_content = self._make_telegraph_text(content)

        logger.info(
            f"[Formatter] Telegram: {len(post_text)} chars, "
            f"Telegraph content: {len(telegraph_content)} chars"
        )

        return TelegramPost(
            text=post_text,
            format_type="html",
            preview_mode=True,
            telegraph_needed=True,
            telegraph_content=telegraph_content,
            telegraph_url=telegraph_url,
            hashtags=hashtags,
            cover_image=cover_image,
            all_images=images or []
        )

    def process(self, title: str, content: str, **kwargs) -> TelegramPost:
        """Основной метод для orchestrator."""
        return self.format_for_telegram(title, content, **kwargs)

    # -----------------------------------------------------------------
    # Формирование Telegram-поста
    # -----------------------------------------------------------------

    def _build_telegram_post(
            self,
            title: str,
            teaser: str,
            telegraph_url: Optional[str],
            source_url: Optional[str],
            source_name: Optional[str],
            hashtags: list[str],
    ) -> str:
        """
        Построить короткий пост для Telegram.

        Формат:
            📰 <b>Заголовок</b>

            Тизер (2-4 предложения)

            📖 Читать полностью → Telegraph

            🔗 Источник

            #теги
        """
        parts = []

        # Заголовок
        parts.append(f"📰 <b>{self._escape(title)}</b>")

        # Тизер
        if teaser:
            parts.append(f"\n{self._escape(teaser)}")

        # Ссылка на Telegraph (полная версия)
        if telegraph_url:
            parts.append(f'\n📖 <a href="{telegraph_url}">Читать полностью</a>')
        else:
            # Плейсхолдер — заменится при публикации
            parts.append('\n📖 Читать полностью: {TELEGRAPH_URL}')

        # Ссылка на оригинал
        if source_url and self.add_source_link:
            label = source_name or "Источник"
            parts.append(f'🔗 <a href="{source_url}">{self._escape(label)}</a>')

        # Хештеги
        if hashtags:
            parts.append('\n' + ' '.join(hashtags[:5]))

        return '\n'.join(parts)

    # -----------------------------------------------------------------
    # Извлечение тизера из контента
    # -----------------------------------------------------------------

    def _extract_teaser(self, content: str, max_length: int = 400) -> str:
        """
        Извлечь тизер из контента (первые 2-3 абзаца).

        Используется если нет готового editorial_teaser.
        """
        if not content:
            return ""

        paragraphs = content.split('\n\n')
        teaser_parts = []
        total_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 30:
                continue

            # Пропускаем заголовки
            if para.startswith('#'):
                para = re.sub(r'^#+\s*', '', para)

            if total_len + len(para) > max_length:
                break

            teaser_parts.append(para)
            total_len += len(para)

            if len(teaser_parts) >= 3:
                break

        teaser = ' '.join(teaser_parts)

        # Обрезаем по последнему предложению если слишком длинный
        if len(teaser) > max_length:
            teaser = teaser[:max_length]
            last_period = teaser.rfind('.')
            if last_period > max_length * 0.5:
                teaser = teaser[:last_period + 1]
            else:
                teaser = teaser.rstrip() + '...'

        return teaser

    # -----------------------------------------------------------------
    # Подготовка контента для Telegraph
    # -----------------------------------------------------------------

    def _make_telegraph_text(self, content: str) -> str:
        """
        Plain text для Telegraph.

        Убирает HTML и Markdown разметку, оставляя чистый текст.
        """
        if not content:
            return ""

        text = content

        # Убираем HTML-теги
        text = re.sub(r'<[^>]+>', '', text)

        # Убираем markdown заголовки
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Убираем markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Убираем markdown ссылки, оставляя текст
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Убираем inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Убираем code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Чистим пустые строки
        paragraphs = text.split('\n\n')
        clean = []
        for para in paragraphs:
            para = para.strip()
            if para:
                para = re.sub(r'\s+', ' ', para)
                clean.append(para)

        return '\n\n'.join(clean)

    # -----------------------------------------------------------------
    # Вспомогательные
    # -----------------------------------------------------------------

    def inject_telegraph_url(self, post: TelegramPost, telegraph_url: str) -> TelegramPost:
        """
        Подставить Telegraph URL в готовый пост.

        Вызывается после создания Telegraph-страницы.
        """
        updated_text = post.text.replace(
            '{TELEGRAPH_URL}',
            f'<a href="{telegraph_url}">Читать полностью</a>'
        )
        updated_text = updated_text.replace(
            '📖 Читать полностью: <a href=',
            '📖 <a href='
        )

        return TelegramPost(
            text=updated_text,
            format_type=post.format_type,
            preview_mode=True,
            telegraph_needed=False,
            telegraph_content=post.telegraph_content,
            telegraph_url=telegraph_url,
            hashtags=post.hashtags,
            cover_image=post.cover_image,
            all_images=post.all_images
        )

    def _escape(self, text: str) -> str:
        """Экранирование HTML."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))

    def _make_hashtags(self, tags: list[str], max_count: int = 5) -> list[str]:
        """Создать хештеги."""
        hashtags = []
        for tag in tags[:max_count]:
            clean = re.sub(r'[^\w\s-]', '', tag)
            clean = clean.replace(' ', '_').replace('-', '_')
            if clean and len(clean) > 1:
                hashtags.append(f"#{clean}")
        return hashtags