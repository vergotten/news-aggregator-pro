#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_pending.py v1.0

Публикация обработанных статей из Supabase → Telegraph → Telegram.

Берёт статьи где:
  - status = 'processed'
  - relevance_score >= min_score
  - telegraph_url IS NULL (ещё не опубликованы)

Для каждой:
  1. Создаёт Telegraph страницу
  2. Отправляет пост в Telegram канал
  3. Обновляет статус → 'published'

Использование:
    # Опубликовать все обработанные (score >= 7)
    python publish_pending.py

    # С другим порогом
    python publish_pending.py --min-score 5

    # Только Telegraph (без Telegram)
    python publish_pending.py --no-telegram

    # Лимит публикаций
    python publish_pending.py --limit 5

    # Dry run (показать что будет опубликовано)
    python publish_pending.py --dry-run

Зависимости:
    pip install psycopg2-binary requests aiohttp
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import psycopg2
import psycopg2.extras
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Database
# =============================================================================

def get_db_url() -> str:
    """Получить DATABASE_URL."""
    url = os.getenv("DATABASE_URL", "")
    if not url:
        logger.error("DATABASE_URL не задан!")
        sys.exit(1)
    return url


def get_pending_articles(min_score: int = 7, limit: int = 20) -> List[Dict]:
    """
    Получить статьи для публикации.

    Условия:
    - status = 'processed'
    - relevance_score >= min_score
    - telegraph_url IS NULL
    """
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT id, title, url, content,
               editorial_title, editorial_teaser, editorial_rewritten,
               telegram_post_text, telegram_cover_image,
               tags, hubs, images,
               relevance_score, author, source
        FROM articles
        WHERE status = 'processed'
          AND relevance_score >= %s
          AND (telegraph_url IS NULL OR telegraph_url = '')
        ORDER BY relevance_score DESC, created_at ASC
        LIMIT %s
    """, (min_score, limit))

    articles = cur.fetchall()
    cur.close()
    conn.close()

    logger.info(f"Найдено {len(articles)} статей для публикации (score >= {min_score})")
    return [dict(a) for a in articles]


def update_article_published(article_id: str, telegraph_url: str, telegram_msg_id: Optional[int] = None):
    """Обновить статью после публикации."""
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()

    cur.execute("""
        UPDATE articles SET
            telegraph_url = %s,
            status = 'published',
            updated_at = %s
        WHERE id = %s
    """, (telegraph_url, datetime.now(timezone.utc), str(article_id)))

    conn.commit()
    cur.close()
    conn.close()


# =============================================================================
# Telegraph Publisher (standalone, requests-based)
# =============================================================================

TELEGRAPH_API = "https://api.telegra.ph"

_telegraph_token: Optional[str] = None


def telegraph_ensure_account() -> str:
    """Создать аккаунт Telegraph."""
    global _telegraph_token
    if _telegraph_token:
        return _telegraph_token

    resp = requests.post(f"{TELEGRAPH_API}/createAccount", data={
        "short_name": "NewsBot",
        "author_name": os.getenv("TELEGRAPH_AUTHOR", "News Aggregator"),
    })
    resp.raise_for_status()
    result = resp.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegraph createAccount failed: {result}")

    _telegraph_token = result["result"]["access_token"]
    logger.info("Telegraph: аккаунт создан")
    return _telegraph_token


def telegraph_publish(title: str, content: str, images: List[str] = None,
                      author: str = None) -> Optional[str]:
    """
    Опубликовать страницу на Telegraph.

    Returns:
        URL страницы или None
    """
    token = telegraph_ensure_account()
    nodes = content_to_telegraph_nodes(content, images)

    resp = requests.post(f"{TELEGRAPH_API}/createPage", data={
        "access_token": token,
        "title": title[:256],
        "author_name": author or os.getenv("TELEGRAPH_AUTHOR", "News Aggregator"),
        "content": json.dumps(nodes, ensure_ascii=False),
        "return_content": "false",
    })
    resp.raise_for_status()
    result = resp.json()

    if result.get("ok"):
        url = result["result"].get("url")
        logger.info(f"Telegraph: {url}")
        return url
    else:
        logger.error(f"Telegraph error: {result.get('error', 'unknown')}")
        return None


def content_to_telegraph_nodes(content: str, images: List[str] = None) -> List[Dict]:
    """Конвертировать текст в Telegraph JSON nodes."""
    if not content:
        return [{"tag": "p", "children": ["Контент отсутствует"]}]

    images = images or []
    remaining = list(images)
    nodes = []

    # Обложка
    if remaining:
        nodes.append({"tag": "figure", "children": [
            {"tag": "img", "attrs": {"src": remaining.pop(0)}}
        ]})

    # Разбиваем на абзацы
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    img_every = 3  # картинка каждые 3 абзаца

    for i, para in enumerate(paragraphs):
        # Короткая строка без пунктуации → подзаголовок
        if len(para) < 80 and not para.endswith((".", ":", "!", "?", ",")):
            nodes.append({"tag": "h4", "children": [para]})
        else:
            nodes.append({"tag": "p", "children": [para]})

        # Вставляем картинку
        if remaining and (i + 1) % img_every == 0:
            nodes.append({"tag": "figure", "children": [
                {"tag": "img", "attrs": {"src": remaining.pop(0)}}
            ]})

    # Оставшиеся картинки
    if remaining:
        nodes.append({"tag": "hr"})
        for img in remaining[:5]:
            nodes.append({"tag": "figure", "children": [
                {"tag": "img", "attrs": {"src": img}}
            ]})

    return nodes


# =============================================================================
# Telegram Publisher (standalone, aiohttp-based)
# =============================================================================

async def telegram_send(text: str) -> Optional[int]:
    """
    Отправить пост в Telegram канал.

    Returns:
        message_id или None
    """
    import aiohttp

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        logger.error(f"Telegram: не настроено (token={'YES' if bot_token else 'NO'}, chat={'YES' if chat_id else 'NO'})")
        return None

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:4096],
        "parse_mode": "HTML",
        "disable_notification": True,
        "disable_web_page_preview": False,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if data.get("ok"):
                    msg_id = data["result"]["message_id"]
                    logger.info(f"Telegram: отправлено (msg_id={msg_id})")
                    return msg_id
                else:
                    logger.error(f"Telegram error: {data.get('description', 'unknown')}")
                    return None
    except Exception as e:
        logger.error(f"Telegram: {type(e).__name__}: {e}")
        return None


def build_telegram_post(article: Dict, telegraph_url: str) -> str:
    """
    Собрать пост для Telegram.

    Приоритет: telegram_post_text (от AI) → fallback (title + teaser).
    """
    # Вариант 1: полный пост от TelegramFormatterAgent
    tg_text = article.get("telegram_post_text")
    if tg_text:
        # Подставляем Telegraph URL
        tg_text = tg_text.replace(
            "📖 Читать полностью → {TELEGRAPH_URL}",
            f'📖 <a href="{telegraph_url}">Читать полностью →</a>'
        )
        return tg_text

    # Вариант 2: fallback
    title = article.get("editorial_title") or article.get("title", "")
    teaser = article.get("editorial_teaser") or ""

    tags = article.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []

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


def _escape_html(text: str) -> str:
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _sanitize_tag(tag: str) -> str:
    import re
    tag = tag.strip().lower()
    tag = re.sub(r"[^\w\s]", "", tag)
    tag = re.sub(r"\s+", "_", tag)
    return tag


# =============================================================================
# Main
# =============================================================================

async def publish_articles(
    min_score: int = 7,
    limit: int = 20,
    no_telegram: bool = False,
    dry_run: bool = False,
    delay: float = 3.0,
):
    """Основной цикл публикации."""
    articles = get_pending_articles(min_score=min_score, limit=limit)

    if not articles:
        logger.info("Нет статей для публикации")
        return

    if dry_run:
        logger.info("=== DRY RUN ===")
        for i, a in enumerate(articles, 1):
            title = a.get("editorial_title") or a.get("title", "?")
            score = a.get("relevance_score", 0)
            has_tg = "✓" if a.get("telegram_post_text") else "✗"
            logger.info(f"  {i}. [{score}/10] {title[:60]}... (tg_post={has_tg})")
        return

    published = 0
    errors = 0

    for i, article in enumerate(articles, 1):
        title = article.get("editorial_title") or article.get("title", "?")
        score = article.get("relevance_score", 0)
        logger.info(f"\n[{i}/{len(articles)}] {title[:60]}... (score={score})")

        # --- Telegraph ---
        try:
            t_content = article.get("editorial_rewritten") or article.get("content") or ""
            t_images = article.get("images") or []
            if isinstance(t_images, str):
                try:
                    t_images = json.loads(t_images)
                except Exception:
                    t_images = []

            telegraph_url = telegraph_publish(
                title=title,
                content=t_content,
                images=t_images,
                author=article.get("author"),
            )

            if not telegraph_url:
                logger.error(f"  Telegraph FAIL, пропуск")
                errors += 1
                continue

        except Exception as e:
            logger.error(f"  Telegraph ERROR: {e}")
            errors += 1
            continue

        # --- Telegram ---
        msg_id = None
        if not no_telegram:
            try:
                tg_text = build_telegram_post(article, telegraph_url)
                msg_id = await telegram_send(tg_text)

                if msg_id:
                    logger.info(f"  ✅ Telegram OK (msg_id={msg_id})")
                else:
                    logger.warning(f"  ⚠️ Telegram не отправлено")
            except Exception as e:
                logger.warning(f"  ⚠️ Telegram ERROR: {e}")

        # --- Update DB ---
        try:
            update_article_published(
                article_id=article["id"],
                telegraph_url=telegraph_url,
                telegram_msg_id=msg_id,
            )
            published += 1
            logger.info(f"  ✅ Опубликовано")
        except Exception as e:
            logger.error(f"  DB update ERROR: {e}")
            errors += 1

        # Задержка между публикациями
        if i < len(articles):
            await asyncio.sleep(delay)

    logger.info(f"\n{'='*50}")
    logger.info(f"Итого: {published} опубликовано, {errors} ошибок из {len(articles)}")


def main():
    parser = argparse.ArgumentParser(description="Публикация обработанных статей")
    parser.add_argument("--min-score", type=int, default=7, help="Мин. score для публикации (default: 7)")
    parser.add_argument("--limit", type=int, default=20, help="Макс. количество (default: 20)")
    parser.add_argument("--no-telegram", action="store_true", help="Только Telegraph, без Telegram")
    parser.add_argument("--dry-run", action="store_true", help="Показать что будет опубликовано")
    parser.add_argument("--delay", type=float, default=3.0, help="Задержка между публикациями (сек)")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"publish_pending.py v1.0 | score >= {args.min_score} | limit={args.limit}")
    asyncio.run(publish_articles(
        min_score=args.min_score,
        limit=args.limit,
        no_telegram=args.no_telegram,
        dry_run=args.dry_run,
        delay=args.delay,
    ))


if __name__ == "__main__":
    main()