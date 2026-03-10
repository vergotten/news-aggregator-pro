# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/infrastructure/telegram/telegram_publisher.py
# =============================================================================
"""
Telegram Publisher Service v2.1

Изменения v2.1:
- Подробное логирование на каждом шаге
- Логирование env vars при инициализации
- Без звука (disable_notification=True)
"""

import os
import re
import logging
from typing import Optional, List
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""
    disable_notification: bool = True
    max_message_length: int = 4096


@dataclass
class TelegramSendResult:
    success: bool
    message_id: Optional[int] = None
    error: Optional[str] = None


class TelegramPublisher:
    TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token=None, chat_id=None, config=None):
        self.config = config or TelegramConfig()
        self.config.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", self.config.bot_token)
        self.config.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", self.config.chat_id)

        has_token = bool(self.config.bot_token)
        has_chat = bool(self.config.chat_id)
        token_preview = self.config.bot_token[:10] + "..." if has_token else "EMPTY"
        logger.info(
            f"[Telegram] v2.1 init: token={token_preview}, "
            f"chat_id={self.config.chat_id or 'EMPTY'}, "
            f"silent={self.config.disable_notification}, "
            f"ready={'YES' if has_token and has_chat else 'NO'}"
        )
        if not has_token:
            logger.warning("[Telegram] TELEGRAM_BOT_TOKEN не задан!")
        if not has_chat:
            logger.warning("[Telegram] TELEGRAM_CHAT_ID не задан!")

    async def send_article_post(self, title, telegraph_url, teaser=None, tags=None, source_url=None, source_name=None):
        logger.info(f"[Telegram] send_article_post: title='{title[:40]}...'")
        text = self._format_article_post(title=title, telegraph_url=telegraph_url, teaser=teaser, tags=tags)
        logger.info(f"[Telegram] Formatted post: {len(text)} chars")
        return await self.send_message(text)

    @staticmethod
    def _format_article_post(title, telegraph_url, teaser=None, tags=None, **kwargs):
        parts = []
        parts.append(f"📰 <b>{_escape_html(title)}</b>")
        if teaser:
            parts.append(f"\n{_escape_html(teaser)}")
        parts.append(f'\n📖 <a href="{telegraph_url}">Читать полностью →</a>')
        if tags:
            hashtags = " ".join(f"#{_sanitize_tag(t)}" for t in tags[:5] if t)
            if hashtags:
                parts.append(f"\n{hashtags}")
        return "\n".join(parts)

    async def send_message(self, text, parse_mode="HTML", disable_notification=None, disable_web_page_preview=False):
        if not self.config.bot_token:
            logger.error("[Telegram] bot_token пустой! Проверьте TELEGRAM_BOT_TOKEN в .env и docker-compose.yml")
            return TelegramSendResult(success=False, error="bot_token пустой")
        if not self.config.chat_id:
            logger.error("[Telegram] chat_id пустой! Проверьте TELEGRAM_CHAT_ID в .env и docker-compose.yml")
            return TelegramSendResult(success=False, error="chat_id пустой")

        original_len = len(text)
        if original_len > self.config.max_message_length:
            text = text[:self.config.max_message_length - 20] + "\n\n<i>...</i>"
            logger.warning(f"[Telegram] Обрезка: {original_len} -> {len(text)}")

        silent = disable_notification if disable_notification is not None else self.config.disable_notification
        url = self.TELEGRAM_API.format(token=self.config.bot_token, method="sendMessage")
        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": silent,
            "disable_web_page_preview": disable_web_page_preview,
        }

        logger.info(f"[Telegram] Отправка: chat_id={self.config.chat_id}, len={len(text)}, silent={silent}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    status = resp.status
                    data = await resp.json()
                    logger.info(f"[Telegram] HTTP {status}, ok={data.get('ok')}")

                    if data.get("ok"):
                        msg_id = data["result"]["message_id"]
                        logger.info(f"[Telegram] ✅ Отправлено! message_id={msg_id}")
                        return TelegramSendResult(success=True, message_id=msg_id)
                    else:
                        error = data.get("description", "unknown")
                        code = data.get("error_code", "?")
                        logger.error(f"[Telegram] API ошибка {code}: {error}")
                        logger.error(f"[Telegram] Начало текста: {text[:200]}...")
                        return TelegramSendResult(success=False, error=f"{code}: {error}")

        except aiohttp.ClientError as e:
            logger.error(f"[Telegram] Сетевая ошибка: {type(e).__name__}: {e}")
            return TelegramSendResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"[Telegram] Ошибка: {type(e).__name__}: {e}", exc_info=True)
            return TelegramSendResult(success=False, error=str(e))

    async def send_batch(self, posts, delay_seconds=2.0):
        import asyncio
        results = []
        for i, post in enumerate(posts, 1):
            logger.info(f"[Telegram] Пакет: {i}/{len(posts)}")
            result = await self.send_article_post(**post)
            results.append(result)
            if i < len(posts):
                await asyncio.sleep(delay_seconds)
        success = sum(1 for r in results if r.success)
        logger.info(f"[Telegram] Пакет: {success}/{len(posts)} успешно")
        return results


def _escape_html(text):
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _sanitize_tag(tag):
    tag = tag.strip().lower()
    tag = re.sub(r"[^\w\s]", "", tag)
    tag = re.sub(r"\s+", "_", tag)
    return tag