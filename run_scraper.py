#!/usr/bin/env python3
"""
Простой запуск парсера (без AI обработки) v4.2

Два режима работы:
  1) Массовый сбор (по умолчанию) — использует src.scrapers + async DB (требует asyncpg)
  2) --url режим — standalone парсинг + psycopg2 для сохранения (без asyncpg)

Изменения v4.2:
- Добавлен --url для парсинга конкретных статей по ссылке
- --url режим: standalone парсер + psycopg2 (без async стека)
- Поддержка нескольких URL через запятую или повтор --url
- --no-save для парсинга без записи в БД
"""

import asyncio
import sys
import os
import argparse
import json
import re
import uuid
from typing import List, Optional, Dict
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup


# =========================================================================
# Standalone Habr парсер (для --url режима, без зависимости от src.*)
#
# Повторяет логику HabrScraperService._parse_full_article(), но не
# импортирует src.* и, значит, не тянет asyncpg через database.py.
# =========================================================================

HABR_BASE_URL = "https://habr.com"
HABR_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def _extract_content_text(article_body) -> str:
    """Извлечение текста из body статьи."""
    if not article_body:
        return ""
    for tag in article_body.find_all(['script', 'style']):
        tag.decompose()
    text = article_body.get_text(separator='\n', strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _normalize_image_url(url: str) -> Optional[str]:
    """Нормализация URL изображения."""
    if not url:
        return None
    url = url.strip()
    if url.startswith('//'):
        url = 'https:' + url
    elif url.startswith('/'):
        url = HABR_BASE_URL + url
    if not url.startswith('http'):
        return None
    return url


def _get_best_image_url(img_tag) -> Optional[str]:
    """Лучший URL изображения из тега (data-src → srcset → src)."""
    if not img_tag:
        return None
    data_src = img_tag.get('data-src')
    if data_src:
        return _normalize_image_url(data_src)
    srcset = img_tag.get('srcset')
    if srcset:
        parts = srcset.split(',')
        if parts:
            return _normalize_image_url(parts[-1].strip().split()[0])
    src = img_tag.get('src')
    if src:
        return _normalize_image_url(src)
    return None


def _extract_all_images(article_body) -> List[str]:
    """Извлечение всех изображений из body."""
    if not article_body:
        return []
    images = []
    seen = set()

    for figure in article_body.find_all('figure'):
        img = figure.find('img')
        if img:
            url = _get_best_image_url(img)
            if url and url not in seen:
                images.append(url)
                seen.add(url)

    for img in article_body.find_all('img'):
        if img.find_parent('figure'):
            continue
        url = _get_best_image_url(img)
        if url and url not in seen:
            images.append(url)
            seen.add(url)

    return images


async def parse_habr_article(url: str) -> Optional[Dict]:
    """
    Спарсить одну статью с Habr по URL.

    Standalone-реализация — повторяет логику HabrScraperService._parse_full_article(),
    но без зависимости от src.* (и asyncpg).
    """
    try:
        async with aiohttp.ClientSession(headers=HABR_HEADERS) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    print(f"  ✗ HTTP {response.status}: {url}")
                    return None
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')

        # Заголовок
        title_elem = soup.find('h1', class_='tm-title')
        if not title_elem:
            title_elem = soup.find('h1', class_='tm-article-snippet__title')
        title = title_elem.find('span').text.strip() if title_elem else "Untitled"

        # Автор
        author_elem = soup.find('a', class_='tm-user-info__username')
        author = author_elem.text.strip() if author_elem else None

        # Дата
        time_elem = soup.find('time')
        published_at = None
        if time_elem and time_elem.get('datetime'):
            try:
                published_at = datetime.fromisoformat(time_elem['datetime'])
            except Exception:
                published_at = datetime.utcnow()
        else:
            published_at = datetime.utcnow()

        # Хабы
        hubs = []
        for hub_elem in soup.find_all('a', class_='tm-publication-hub__link'):
            hub_span = hub_elem.find('span')
            if hub_span:
                hubs.append(hub_span.text.strip())

        # Контент
        article_body = soup.find('div', class_='tm-article-body')
        if not article_body:
            article_body = soup.find('div', class_='article-formatted-body')
        if not article_body:
            print(f"  ✗ Контент не найден: {url}")
            return None

        content = _extract_content_text(article_body)
        images = _extract_all_images(article_body)

        return {
            'title': title,
            'content': content,
            'url': url,
            'author': author,
            'published_at': published_at,
            'tags': hubs.copy(),
            'hubs': hubs,
            'images': images,
        }

    except Exception as e:
        print(f"  ✗ Ошибка парсинга {url}: {e}")
        return None


# =========================================================================
# Сохранение через psycopg2 (синхронное, без asyncpg)
#
# Читает connection string из .env / переменных окружения тем же способом,
# что и Settings, но не импортирует Settings (чтобы не тянуть database.py).
# =========================================================================

def get_db_connection_string() -> str:
    """
    Получить строку подключения PostgreSQL из .env / переменных окружения.

    Приоритет: DATABASE_URL env → построение из POSTGRES_* компонентов → дефолт.
    """
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        # psycopg2 понимает postgresql://, убираем +asyncpg если попало
        return db_url.replace('postgresql+asyncpg://', 'postgresql://')

    user = os.getenv('POSTGRES_USER', 'newsaggregator')
    password = os.getenv('POSTGRES_PASSWORD', 'changeme123')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5433')
    db = os.getenv('POSTGRES_DB', 'news_aggregator')

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def save_articles_to_db_sync(articles: List[Dict]) -> Dict[str, int]:
    """
    Сохранить статьи в PostgreSQL через psycopg2 (синхронно).

    Использует INSERT ... ON CONFLICT (url) DO NOTHING для пропуска дубликатов.
    """
    import psycopg2

    stats = {'saved': 0, 'duplicates': 0, 'errors': 0}
    conn_str = get_db_connection_string()

    try:
        conn = psycopg2.connect(conn_str)
        conn.autocommit = False
        cur = conn.cursor()

        for article in articles:
            try:
                article_id = str(uuid.uuid4())
                now = datetime.utcnow()

                pub_at = article.get('published_at')
                if isinstance(pub_at, datetime):
                    pub_at = pub_at.isoformat()

                cur.execute("""
                    INSERT INTO articles (
                        id, title, content, url, source,
                        author, published_at, tags, hubs, images,
                        status, created_at, updated_at, article_metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (url) DO NOTHING
                """, (
                    article_id,
                    article['title'],
                    article.get('content', ''),
                    article['url'],
                    'habr',
                    article.get('author'),
                    pub_at,
                    article.get('tags', []),
                    article.get('hubs', []),
                    article.get('images', []),
                    'pending',
                    now,
                    now,
                    json.dumps({}, ensure_ascii=False),
                ))

                if cur.rowcount > 0:
                    stats['saved'] += 1
                else:
                    stats['duplicates'] += 1

            except Exception as e:
                conn.rollback()
                stats['errors'] += 1
                print(f"  ✗ DB ошибка для '{article.get('title', '')[:40]}': {e}")
                continue

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print(f"\n❌ Не удалось подключиться к БД: {e}")
        print(f"   Строка подключения: {conn_str}")
        print(f"   Проверьте, запущен ли PostgreSQL и правильны ли настройки в .env")
        stats['errors'] = len(articles)

    return stats


# =========================================================================
# Режим --url: парсинг + сохранение (standalone, без asyncpg)
# =========================================================================

async def run_habr_scraper_by_urls(urls: List[str], verbose: bool = False, save: bool = True):
    """
    Спарсить конкретные статьи по URL и (опционально) сохранить в БД.

    Работает без asyncpg — использует aiohttp для парсинга и psycopg2 для БД.
    """
    print(f"\n{'=' * 60}")
    print(f"🔗 HABR SCRAPER v4.2 (по URL)")
    print(f"{'=' * 60}")
    print(f"  Режим:            парсинг по URL")
    print(f"  Статей:           {len(urls)}")
    print(f"  Сохранение в БД:  {'да' if save else 'нет'}")
    for i, url in enumerate(urls, 1):
        print(f"  [{i}] {url}")
    print(f"{'=' * 60}\n")

    # Фаза 1: Парсинг
    parsed_articles = []
    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url:
            continue

        print(f"  [{i}/{len(urls)}] Парсинг: {url[:70]}...")

        article_data = await parse_habr_article(url)
        if article_data:
            parsed_articles.append(article_data)
            print(f"  ✓ {article_data['title'][:60]}")
            if verbose:
                print(f"    {len(article_data.get('content', ''))} chars, "
                      f"{len(article_data.get('images', []))} images, "
                      f"hubs: {', '.join(article_data.get('hubs', []))}")
        else:
            print(f"  ✗ Не удалось спарсить")

    # Фаза 2: Сохранение
    stats = {'scraped': len(parsed_articles), 'saved': 0, 'duplicates': 0, 'errors': 0}

    if parsed_articles and save:
        print(f"\n  Сохранение в БД ({len(parsed_articles)} статей)...")
        db_stats = save_articles_to_db_sync(parsed_articles)
        stats.update(db_stats)

    # Результаты
    print(f"\n{'=' * 60}")
    print(f"✅ РЕЗУЛЬТАТЫ")
    print(f"{'=' * 60}")
    print(f"  Собрано:     {stats['scraped']}")
    if save:
        print(f"  Сохранено:   {stats['saved']}")
        print(f"  Дубликатов:  {stats['duplicates']}")
        print(f"  Ошибок:      {stats['errors']}")
    print(f"{'=' * 60}")

    if stats.get('saved', 0) > 0 or (not save and stats['scraped'] > 0):
        print(f"\n💡 Для AI обработки:")
        print(f"   python run_full_pipeline.py --url {','.join(urls[:3])}")

    print()
    return stats


# =========================================================================
# Массовый сбор (оригинальный режим, импортирует src.* — требует asyncpg)
# =========================================================================

async def run_habr_scraper(limit: int = 10, hubs: str = "", verbose: bool = False):
    """Запустить Habr парсер (массовый сбор + async DB). Требует asyncpg."""
    from src.scrapers.habr.scraper_service import HabrScraperService

    print(f"\n{'=' * 60}")
    print(f"🚀 HABR SCRAPER v4.2 (без AI обработки)")
    print(f"{'=' * 60}")
    print(f"  Режим:        массовый сбор")
    print(f"  Лимит статей: {limit}")
    print(f"  Хабы:         {hubs if hubs else 'все'}")
    print(f"{'=' * 60}\n")

    service = HabrScraperService()
    hubs_list = [h.strip() for h in hubs.split(',')] if hubs else []

    def progress_callback():
        if verbose:
            print(".", end="", flush=True)

    results = await service.scrape_and_save(
        limit=limit,
        hubs=hubs_list,
        progress_callback=progress_callback if verbose else None
    )

    if verbose:
        print()

    print(f"\n{'=' * 60}")
    print(f"✅ РЕЗУЛЬТАТЫ")
    print(f"{'=' * 60}")
    print(f"  Собрано:     {results['scraped']}")
    print(f"  Сохранено:   {results['saved']}")
    print(f"  Дубликатов:  {results['duplicates']}")
    print(f"  Ошибок:      {results['errors']}")
    print(f"{'=' * 60}")

    if results['saved'] > 0:
        print(f"\n💡 Для AI обработки:")
        print(f"   python run_full_pipeline.py {results['saved']} --provider groq")

    print()


# =========================================================================
# CLI
# =========================================================================

def parse_urls(raw_urls: List[str]) -> List[str]:
    """Нормализовать список URL: разделение по запятым."""
    result = []
    for item in raw_urls:
        for url in item.split(','):
            url = url.strip()
            if url:
                result.append(url)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Habr парсер v4.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Собрать 50 статей из ленты (требует asyncpg + PostgreSQL)
  python %(prog)s 50

  # Спарсить конкретную статью (работает без asyncpg!)
  python %(prog)s --url https://habr.com/ru/articles/123456/

  # Несколько статей
  python %(prog)s --url https://habr.com/ru/articles/111/,https://habr.com/ru/articles/222/

  # Повтор --url
  python %(prog)s --url https://habr.com/ru/articles/111/ --url https://habr.com/ru/articles/222/

  # Только парсинг, без сохранения в БД
  python %(prog)s --url https://habr.com/ru/articles/123456/ --no-save

Для AI обработки:
  python run_full_pipeline.py --url https://habr.com/ru/articles/123456/
        """
    )

    parser.add_argument('limit', type=int, nargs='?', default=10,
                        help='Количество статей (default: 10). Игнорируется при --url')
    parser.add_argument('hubs', type=str, nargs='?', default="",
                        help='Хабы через запятую. Игнорируется при --url')
    parser.add_argument('--url', '-u', action='append', default=None,
                        help='URL статьи (можно указать несколько раз или через запятую)')
    parser.add_argument('--no-save', action='store_true', default=False,
                        help='Только парсинг, без сохранения в БД')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')

    args = parser.parse_args()

    try:
        if args.url:
            # --url режим: standalone парсер + psycopg2 (без asyncpg)
            urls = parse_urls(args.url)
            if not urls:
                print("❌ Не указаны URL")
                sys.exit(1)
            asyncio.run(run_habr_scraper_by_urls(
                urls=urls,
                verbose=args.verbose,
                save=not args.no_save
            ))
        else:
            # Массовый режим: src.scrapers + async DB (asyncpg)
            asyncio.run(run_habr_scraper(args.limit, args.hubs, args.verbose))
    except KeyboardInterrupt:
        print("\n⚠️  Прервано")
        sys.exit(1)