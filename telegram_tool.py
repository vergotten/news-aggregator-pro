#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telegram_tool.py v1.0

Утилита для работы с Telegram ботом.

Использование:
    # Проверить подключение
    python telegram_tool.py status

    # Отправить тестовое сообщение
    python telegram_tool.py test

    # Отправить форматированный пост
    python telegram_tool.py post --title "Заголовок" --body "Текст поста"

    # Показать статьи из БД
    python telegram_tool.py list [--status processed] [--limit 10]

    # Отправить статью из БД
    python telegram_tool.py send --id <article_id>

    # Отправить все processed статьи
    python telegram_tool.py send-all [--min-score 7] [--limit 5]

Зависимости:
    pip install requests psycopg2-binary
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TelegramTool:
    """Утилита для Telegram бота."""

    API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.db_url = os.getenv("DATABASE_URL", "")

        if not self.token:
            logger.error("TELEGRAM_BOT_TOKEN не задан! Добавьте в .env")
        if not self.chat_id:
            logger.error("TELEGRAM_CHAT_ID не задан! Добавьте в .env")

    # =================================================================
    # API
    # =================================================================

    def _call(self, method: str, **params) -> Dict:
        """Вызов Telegram Bot API."""
        url = self.API.format(token=self.token, method=method)
        resp = requests.post(url, json=params, timeout=30)
        return resp.json()

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        silent: bool = True,
    ) -> Dict:
        """Отправить сообщение в канал."""
        result = self._call(
            "sendMessage",
            chat_id=self.chat_id,
            text=text[:4096],
            parse_mode=parse_mode,
            disable_notification=silent,
        )

        if result.get("ok"):
            msg_id = result["result"]["message_id"]
            logger.info(f"✅ Отправлено! message_id={msg_id}")
        else:
            error = result.get("description", "unknown")
            code = result.get("error_code", "?")
            logger.error(f"❌ Ошибка {code}: {error}")

        return result

    # =================================================================
    # Команды
    # =================================================================

    def status(self):
        """Проверить подключение бота."""
        print("=" * 50)
        print("🤖 TELEGRAM BOT STATUS")
        print("=" * 50)

        # Токен
        if self.token:
            print(f"  Token:   {self.token[:10]}...{self.token[-4:]}")
        else:
            print("  Token:   ❌ НЕ ЗАДАН")
            return

        # Chat ID
        if self.chat_id:
            print(f"  Chat ID: {self.chat_id}")
        else:
            print("  Chat ID: ❌ НЕ ЗАДАН")
            return

        # getMe
        me = self._call("getMe")
        if me.get("ok"):
            bot = me["result"]
            print(f"  Bot:     @{bot.get('username', '?')}")
            print(f"  Name:    {bot.get('first_name', '?')}")
            print(f"  Status:  ✅ Подключён")
        else:
            print(f"  Status:  ❌ {me.get('description', 'error')}")
            return

        # getChat
        chat = self._call("getChat", chat_id=self.chat_id)
        if chat.get("ok"):
            ch = chat["result"]
            chat_type = ch.get("type", "?")
            chat_title = ch.get("title", ch.get("username", "?"))
            print(f"  Channel: {chat_title} ({chat_type})")
            print(f"  Status:  ✅ Бот имеет доступ")
        else:
            print(f"  Channel: ❌ {chat.get('description', 'нет доступа')}")

        # DB
        if self.db_url:
            try:
                import psycopg2
                conn = psycopg2.connect(self.db_url)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM articles")
                total = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM articles WHERE status = 'processed'")
                processed = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM articles WHERE status = 'published'")
                published = cur.fetchone()[0]
                cur.close()
                conn.close()
                print(f"  DB:      ✅ {total} статей (processed={processed}, published={published})")
            except Exception as e:
                print(f"  DB:      ❌ {e}")
        else:
            print(f"  DB:      ⚠️ DATABASE_URL не задан")

        print("=" * 50)

    def test(self):
        """Отправить тестовое сообщение."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"🧪 <b>Тестовое сообщение</b>\n\n"
            f"Бот работает корректно.\n"
            f"Время: {now}\n\n"
            f"<i>telegram_tool.py v1.0</i>"
        )
        self.send_message(text)

    def post(self, title: str, body: str, link: Optional[str] = None, tags: List[str] = None):
        """Отправить форматированный пост."""
        parts = []
        parts.append(f"📰 <b>{_escape(title)}</b>")
        parts.append(f"\n{_escape(body)}")

        if link:
            parts.append(f'\n📖 <a href="{link}">Читать полностью →</a>')

        if tags:
            hashtags = " ".join(f"#{t.strip()}" for t in tags if t.strip())
            if hashtags:
                parts.append(f"\n{hashtags}")

        self.send_message("\n".join(parts))

    def list_articles(self, status: str = "processed", limit: int = 10):
        """Показать статьи из БД."""
        if not self.db_url:
            logger.error("DATABASE_URL не задан!")
            return

        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT id, title, editorial_title, relevance_score, status,
                   telegraph_url, telegram_post_text IS NOT NULL as has_tg_post,
                   created_at
            FROM articles
            WHERE status = %s
            ORDER BY relevance_score DESC NULLS LAST, created_at DESC
            LIMIT %s
        """, (status, limit))

        articles = cur.fetchall()
        cur.close()
        conn.close()

        if not articles:
            print(f"Нет статей со статусом '{status}'")
            return

        print(f"\n{'='*70}")
        print(f"📋 Статьи [{status}] — {len(articles)} шт.")
        print(f"{'='*70}")

        for i, a in enumerate(articles, 1):
            title = a.get("editorial_title") or a.get("title", "?")
            score = a.get("relevance_score", "?")
            has_tg = "✓" if a.get("has_tg_post") else "✗"
            has_telegraph = "✓" if a.get("telegraph_url") else "✗"
            aid = str(a["id"])[:8]

            print(f"  {i}. [{score}/10] {title[:55]}...")
            print(f"     ID: {aid}  TG_post:{has_tg}  Telegraph:{has_telegraph}")

        print(f"{'='*70}\n")

    def send_article(self, article_id: str):
        """Отправить конкретную статью из БД."""
        if not self.db_url:
            logger.error("DATABASE_URL не задан!")
            return

        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT id, title, editorial_title, editorial_teaser,
                   telegram_post_text, telegraph_url, tags, relevance_score
            FROM articles WHERE id::text LIKE %s
            LIMIT 1
        """, (f"{article_id}%",))

        article = cur.fetchone()
        cur.close()
        conn.close()

        if not article:
            logger.error(f"Статья {article_id} не найдена")
            return

        title = article.get("editorial_title") or article.get("title", "?")
        logger.info(f"Отправка: {title[:50]}...")

        # Приоритет: telegram_post_text → fallback
        tg_text = article.get("telegram_post_text")
        telegraph_url = article.get("telegraph_url")

        if tg_text:
            if telegraph_url:
                tg_text = tg_text.replace(
                    "📖 Читать полностью → {TELEGRAPH_URL}",
                    f'📖 <a href="{telegraph_url}">Читать полностью →</a>'
                )
            logger.info(f"Отправка полного поста ({len(tg_text)} chars)")
            self.send_message(tg_text)
        else:
            # Fallback
            teaser = article.get("editorial_teaser") or ""
            tags = article.get("tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []

            self.post(
                title=title,
                body=teaser,
                link=telegraph_url,
                tags=tags,
            )

    def send_all(self, min_score: int = 7, limit: int = 5):
        """Отправить все processed статьи."""
        if not self.db_url:
            logger.error("DATABASE_URL не задан!")
            return

        import time
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT id::text as id FROM articles
            WHERE status = 'processed'
              AND relevance_score >= %s
              AND (telegraph_url IS NULL OR telegraph_url = '')
            ORDER BY relevance_score DESC
            LIMIT %s
        """, (min_score, limit))

        articles = cur.fetchall()
        cur.close()
        conn.close()

        if not articles:
            logger.info(f"Нет статей для отправки (score >= {min_score})")
            return

        logger.info(f"Отправка {len(articles)} статей...")

        for i, a in enumerate(articles, 1):
            logger.info(f"[{i}/{len(articles)}]")
            self.send_article(a["id"])
            if i < len(articles):
                time.sleep(3)


# =================================================================
# Утилиты
# =================================================================

def _escape(text: str) -> str:
    if not text:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# =================================================================
# CLI
# =================================================================

def main():
    # Загружаем .env если есть
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)

    parser = argparse.ArgumentParser(description="Telegram Bot Tool v1.0")
    sub = parser.add_subparsers(dest="command", help="Команда")

    # status
    sub.add_parser("status", help="Проверить подключение")

    # test
    sub.add_parser("test", help="Отправить тестовое сообщение")

    # post
    p_post = sub.add_parser("post", help="Отправить форматированный пост")
    p_post.add_argument("--title", required=True, help="Заголовок")
    p_post.add_argument("--body", required=True, help="Текст")
    p_post.add_argument("--link", help="Ссылка")
    p_post.add_argument("--tags", help="Теги через запятую")

    # list
    p_list = sub.add_parser("list", help="Показать статьи из БД")
    p_list.add_argument("--status", default="processed", help="Статус (default: processed)")
    p_list.add_argument("--limit", type=int, default=10)

    # send
    p_send = sub.add_parser("send", help="Отправить статью из БД")
    p_send.add_argument("--id", required=True, help="ID статьи (можно первые символы)")

    # send-all
    p_all = sub.add_parser("send-all", help="Отправить все processed статьи")
    p_all.add_argument("--min-score", type=int, default=7)
    p_all.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()
    tool = TelegramTool()

    if args.command == "status":
        tool.status()
    elif args.command == "test":
        tool.test()
    elif args.command == "post":
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        tool.post(title=args.title, body=args.body, link=args.link, tags=tags)
    elif args.command == "list":
        tool.list_articles(status=args.status, limit=args.limit)
    elif args.command == "send":
        tool.send_article(args.id)
    elif args.command == "send-all":
        tool.send_all(min_score=args.min_score, limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# =================================================================
# Примеры использования:
# =================================================================
#
# python telegram_tool.py status
#   → Проверить бота, канал, БД (токен, chat_id, доступ, кол-во статей)
#
# python telegram_tool.py test
#   → Отправить тестовое сообщение в канал
#
# python telegram_tool.py post --title "AI новости" --body "OpenAI выпустил GPT-5" --link "https://example.com" --tags "ai,openai"
#   → Отправить форматированный пост с заголовком, текстом, ссылкой и тегами
#
# python telegram_tool.py list
#   → Показать processed статьи из БД
#
# python telegram_tool.py list --status pending --limit 20
#   → Показать 20 pending статей
#
# python telegram_tool.py send --id 0929dd03
#   → Отправить конкретную статью (достаточно первых символов ID)
#
# python telegram_tool.py send-all --min-score 5 --limit 3
#   → Отправить до 3 статей со скором >= 5
#
# С Supabase:
# DATABASE_URL="postgresql://..." python telegram_tool.py list
# =================================================================