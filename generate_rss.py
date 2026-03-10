#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rss.py v1.0

Генерация RSS ленты из опубликованных статей.

Читает из Supabase все статьи со статусом 'published'
и telegraph_url IS NOT NULL, генерирует feed.xml.

Использование:
    python generate_rss.py
    python generate_rss.py --output docs/feed.xml --limit 50

Зависимости:
    pip install psycopg2-binary
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from typing import List, Dict

import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

FEED_TITLE = os.getenv("RSS_FEED_TITLE", "Tech News Aggregator")
FEED_DESCRIPTION = os.getenv("RSS_FEED_DESCRIPTION", "Curated tech news from Habr and more")
FEED_LINK = os.getenv("RSS_FEED_LINK", "https://github.com")
FEED_LANGUAGE = os.getenv("RSS_FEED_LANGUAGE", "ru")


# =============================================================================
# Database
# =============================================================================

def get_published_articles(limit: int = 50) -> List[Dict]:
    """Получить опубликованные статьи."""
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL не задан!")
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT id, title, url,
               editorial_title, editorial_teaser,
               telegraph_url, tags,
               relevance_score, published_at, updated_at
        FROM articles
        WHERE status = 'published'
          AND telegraph_url IS NOT NULL
          AND telegraph_url != ''
        ORDER BY updated_at DESC
        LIMIT %s
    """, (limit,))

    articles = cur.fetchall()
    cur.close()
    conn.close()

    logger.info(f"Найдено {len(articles)} опубликованных статей")
    return [dict(a) for a in articles]


# =============================================================================
# RSS Generation
# =============================================================================

def generate_rss(articles: List[Dict]) -> str:
    """Сгенерировать RSS XML из списка статей."""
    rss = Element("rss", version="2.0")
    rss.set("xmlns:atom", "http://www.w3.org/2005/Atom")

    channel = SubElement(rss, "channel")

    # Channel metadata
    SubElement(channel, "title").text = FEED_TITLE
    SubElement(channel, "description").text = FEED_DESCRIPTION
    SubElement(channel, "link").text = FEED_LINK
    SubElement(channel, "language").text = FEED_LANGUAGE
    SubElement(channel, "lastBuildDate").text = _rfc822_now()
    SubElement(channel, "generator").text = "news-aggregator-pro/generate_rss.py"

    # Items
    for article in articles:
        item = SubElement(channel, "item")

        title = article.get("editorial_title") or article.get("title", "Untitled")
        SubElement(item, "title").text = title

        # Ссылка на Telegraph (полная версия)
        telegraph_url = article.get("telegraph_url", "")
        SubElement(item, "link").text = telegraph_url

        # GUID
        guid = SubElement(item, "guid", isPermaLink="false")
        guid.text = str(article.get("id", telegraph_url))

        # Описание (тизер)
        teaser = article.get("editorial_teaser") or ""
        if teaser:
            SubElement(item, "description").text = teaser

        # Дата
        pub_date = article.get("published_at") or article.get("updated_at")
        if pub_date:
            SubElement(item, "pubDate").text = _to_rfc822(pub_date)

        # Категории (теги)
        tags = article.get("tags") or []
        if isinstance(tags, str):
            import json
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []

        for tag in tags[:5]:
            if tag:
                SubElement(item, "category").text = str(tag)

    # Pretty print
    xml_str = tostring(rss, encoding="unicode")
    dom = parseString(xml_str)
    pretty = dom.toprettyxml(indent="  ", encoding="utf-8")

    # Убираем лишнюю XML декларацию от minidom, ставим свою
    lines = pretty.decode("utf-8").split("\n")
    if lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'

    return "\n".join(lines)


def _rfc822_now() -> str:
    """Текущее время в RFC 822."""
    now = datetime.now(timezone.utc)
    return now.strftime("%a, %d %b %Y %H:%M:%S +0000")


def _to_rfc822(dt) -> str:
    """Конвертировать datetime в RFC 822."""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return _rfc822_now()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Генерация RSS ленты")
    parser.add_argument("--output", "-o", default="docs/feed.xml", help="Путь к выходному файлу")
    parser.add_argument("--limit", type=int, default=50, help="Макс. статей в ленте")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"generate_rss.py v1.0 | output={args.output} | limit={args.limit}")

    articles = get_published_articles(limit=args.limit)

    if not articles:
        logger.warning("Нет опубликованных статей, генерируем пустую ленту")

    rss_xml = generate_rss(articles)

    # Создаём директорию если нужно
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(rss_xml)

    logger.info(f"RSS: {len(articles)} статей → {args.output}")


if __name__ == "__main__":
    main()