# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/infrastructure/telegram/telegram_publisher.py
# =============================================================================
"""
Telegram Publisher Service v2.0

Только отправка сообщений в Telegram-канал.
Telegraph-публикация — в telegraph_publisher.py.

Зависимости:
    pip install aiohttp

Использование:
    from src.infrastructure.telegram.telegram_publisher import TelegramPublisher

    publisher = TelegramPublisher()

    # Отправить пост с тизером и ссылкой на Telegraph
    msg_id = await publisher.send_article_post(
        title="Заголовок",
        teaser="Краткое описание...",
        telegraph_url="https://telegra.ph/...",
        tags=["python", "backend"],
        source_url="https://habr.com/..."
    )

    # Отправить произвольное сообщение
    msg_id = await publisher.send_message("Текст сообщения")
"""

import os
import re
import logging
from typing import Optional, List
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# Конфигурация
# =============================================================================

@dataclass
class TelegramConfig:
    """Конфигурация Telegram."""
    bot_token: str = ""
    chat_id: str = ""  # @channel_name или -100xxxxxxxxxx
    disable_notification: bool = True  # Без звука
    max_message_length: int = 4096


@dataclass
class TelegramSendResult:
    """Результат отправки."""
    success: bool
    message_id: Optional[int] = None
    error: Optional[str] = None


# =============================================================================
# Telegram Publisher
# =============================================================================

class TelegramPublisher:
    """
    Сервис отправки сообщений в Telegram-канал.

    Формирует красивые посты с тизером и ссылкой на Telegraph.
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        config: Optional[TelegramConfig] = None,
    ):
        self.config = config or TelegramConfig()

        # Переопределяем из параметров или env
        self.config.bot_token = (
            bot_token
            or os.getenv("TELEGRAM_BOT_TOKEN", self.config.bot_token)
        )
        self.config.chat_id = (
            chat_id
            or os.getenv("TELEGRAM_CHAT_ID", self.config.chat_id)
        )

        logger.info("[Telegram] TelegramPublisher v2.0 initialized")

    # -----------------------------------------------------------------
    # Отправка поста со статьёй
    # -----------------------------------------------------------------

    async def send_article_post(
        self,
        title: str,
        telegraph_url: str,
        teaser: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> TelegramSendResult:
        """
        Отправить пост со статьёй в Telegram-канал.

        Формат поста:
            📰 Заголовок статьи

            Краткий тизер (2-4 предложения)...

            📖 Читать полностью (ссылка на Telegraph)

            #python #backend #habr

        Args:
            title: Заголовок статьи
            telegraph_url: URL страницы на Telegraph
            teaser: Тизер (короткое описание)
            tags: Теги для хештегов
            source_url: Ссылка на оригинал
            source_name: Название источника (habr, reddit, etc.)

        Returns:
            TelegramSendResult
        """
        text = self._format_article_post(
            title=title,
            telegraph_url=telegraph_url,
            teaser=teaser,
            tags=tags,
            source_url=source_url,
            source_name=source_name,
        )

        return await self.send_message(text)

    # -----------------------------------------------------------------
    # Форматирование поста
    # -----------------------------------------------------------------

    @staticmethod
    def _format_article_post(
        title: str,
        telegraph_url: str,
        teaser: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> str:
        """
        Формирует пост для Telegram-канала.

        Пример:
            📰 Как мигрировать монолит на микросервисы

            Представлен опыт миграции крупного проекта
            на микросервисную архитектуру. Рассмотрены
            ключевые решения и подводные камни.

            📖 Читать полностью

            🔗 Источник

            #python #backend #microservices
        """
        parts = []

        # Заголовок
        parts.append(f"📰 <b>{_escape_html(title)}</b>")

        # Тизер
        if teaser:
            parts.append(f"\n{_escape_html(teaser)}")

        # Ссылка на Telegraph (полная версия)
        parts.append(f'\n📖 <a href="{telegraph_url}">Читать полностью</a>')

        # Ссылка на оригинал
        if source_url:
            source_label = source_name or "Источник"
            parts.append(f'🔗 <a href="{source_url}">{_escape_html(source_label)}</a>')

        # Хештеги
        if tags:
            hashtags = " ".join(
                f"#{_sanitize_tag(t)}" for t in tags[:5] if t
            )
            if hashtags:
                parts.append(f"\n{hashtags}")

        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Отправка сообщения
    # -----------------------------------------------------------------

    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: Optional[bool] = None,
        disable_web_page_preview: bool = False,
    ) -> TelegramSendResult:
        """
        Отправляет сообщение в Telegram-канал.

        Args:
            text: Текст сообщения (HTML)
            parse_mode: Режим парсинга ("HTML" или "MarkdownV2")
            disable_notification: Без звука
            disable_web_page_preview: Отключить превью ссылок

        Returns:
            TelegramSendResult
        """
        if not self.config.bot_token or not self.config.chat_id:
            logger.error("[Telegram] Не настроены bot_token или chat_id")
            return TelegramSendResult(
                success=False,
                error="Не настроены bot_token или chat_id"
            )

        # Обрезаем если слишком длинный
        if len(text) > self.config.max_message_length:
            text = text[:self.config.max_message_length - 20] + "\n\n<i>...</i>"

        url = self.TELEGRAM_API.format(
            token=self.config.bot_token,
            method="sendMessage",
        )

        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": (
                disable_notification
                if disable_notification is not None
                else self.config.disable_notification
            ),
            "disable_web_page_preview": disable_web_page_preview,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if data.get("ok"):
                        msg_id = data["result"]["message_id"]
                        logger.info("[Telegram] Отправлено, message_id=%s", msg_id)
                        return TelegramSendResult(
                            success=True,
                            message_id=msg_id
                        )
                    else:
                        error = data.get("description", "unknown error")
                        logger.error("[Telegram] Ошибка API: %s", error)
                        return TelegramSendResult(
                            success=False,
                            error=error
                        )

        except Exception as e:
            logger.error("[Telegram] Ошибка отправки: %s", e)
            return TelegramSendResult(
                success=False,
                error=str(e)
            )

    # -----------------------------------------------------------------
    # Пакетная отправка
    # -----------------------------------------------------------------

    async def send_batch(
        self,
        posts: List[dict],
        delay_seconds: float = 2.0,
    ) -> List[TelegramSendResult]:
        """
        Отправка нескольких постов с задержкой.

        Telegram ограничивает ~20 сообщений в минуту.

        Args:
            posts: Список словарей с параметрами для send_article_post
            delay_seconds: Задержка между отправками

        Returns:
            Список результатов
        """
        import asyncio

        results = []
        total = len(posts)

        for i, post in enumerate(posts, 1):
            logger.info("[Telegram] Пакет: %d/%d", i, total)

            result = await self.send_article_post(**post)
            results.append(result)

            if i < total:
                await asyncio.sleep(delay_seconds)

        success = sum(1 for r in results if r.success)
        logger.info(
            "[Telegram] Пакет завершён: %d/%d успешно",
            success, total,
        )

        return results


# =============================================================================
# Утилиты
# =============================================================================

def _escape_html(text: str) -> str:
    """Экранирование для Telegram HTML."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _sanitize_tag(tag: str) -> str:
    """Очистка тега для хештега."""
    tag = tag.strip().lower()
    tag = re.sub(r"[^\w\s]", "", tag)
    tag = re.sub(r"\s+", "_", tag)
    return tag