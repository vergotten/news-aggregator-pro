#!/usr/bin/env python3
"""
Process Existing Articles v4.2

AI обработка статей, уже находящихся в базе данных.

Способы выбора статей:
  1) По умолчанию — все необработанные (relevance_score IS NULL)
  2) --url       — конкретная статья по URL
  3) --id        — конкретная статья по UUID
  4) --reprocess-all — все статьи, включая обработанные
  5) --days N    — только за последние N дней
  6) --limit N   — ограничить количество

Изменения v4.2:
- --url / --id для выбора конкретных статей
- psycopg2 fallback для --url/--id (работает без asyncpg)
- Ленивые импорты src.* (не падает без asyncpg)
"""

import asyncio
import sys
import os
import time
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# НЕ импортируем src.* на верхнем уровне — чтобы не тянуть asyncpg
# Все src.* импорты ленивые, внутри функций

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================
# Форматирование
# =========================================================================

def fmt_header(title, char="=", w=80):
    return f"\n{char * w}\n{title}\n{char * w}"


def fmt_sub(title, w=80):
    return f"\n{'-' * w}\n{title}\n{'-' * w}"


def fmt_row(label, value, w=80):
    l = f"  {label}:"
    v = str(value)
    return f"{l}{' ' * max(1, w - len(l) - len(v))}{v}"


# =========================================================================
# psycopg2: загрузка статей из БД (для --url / --id режима)
# =========================================================================

def _get_db_connection_string() -> str:
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url.replace('postgresql+asyncpg://', 'postgresql://')
    user = os.getenv('POSTGRES_USER', 'newsaggregator')
    password = os.getenv('POSTGRES_PASSWORD', 'changeme123')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5433')
    db = os.getenv('POSTGRES_DB', 'news_aggregator')
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _row_to_dict(row, columns) -> Dict[str, Any]:
    """Конвертировать строку psycopg2 в словарь."""
    return dict(zip(columns, row))


def _load_articles_sync(
        urls: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        days: Optional[int] = None,
        reprocess_all: bool = False,
) -> List[Dict[str, Any]]:
    """
    Загрузить статьи из БД через psycopg2.

    Returns:
        Список словарей с полями статьи
    """
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(_get_db_connection_string())
    cur = conn.cursor()

    # Базовые колонки
    columns = [
        'id', 'title', 'content', 'url', 'source',
        'author', 'published_at', 'created_at', 'updated_at',
        'status', 'is_news', 'relevance_score', 'relevance_reason',
        'editorial_title', 'editorial_teaser', 'editorial_rewritten',
        'tags', 'hubs', 'images',
    ]
    select_cols = ', '.join(columns)

    conditions = []
    params = []

    # Фильтр по URL
    if urls:
        conditions.append("url = ANY(%s)")
        params.append(urls)

    # Фильтр по ID
    elif ids:
        conditions.append("id = ANY(%s::uuid[])")
        params.append(ids)

    else:
        # Стандартные фильтры
        if not reprocess_all:
            conditions.append("relevance_score IS NULL")

        if days:
            conditions.append("created_at >= %s")
            params.append(datetime.utcnow() - timedelta(days=days))

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = "ORDER BY created_at DESC"
    limit_clause = f"LIMIT {limit}" if limit else ""

    sql = f"SELECT {select_cols} FROM articles {where} {order} {limit_clause}"

    cur.execute(sql, params)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [_row_to_dict(row, columns) for row in rows]


def _update_article_after_ai(article_id: str, processed_article) -> bool:
    """
    Обновить статью в БД после AI обработки (psycopg2).

    Записывает все AI-поля: relevance_score, editorial_title, и т.д.
    """
    import psycopg2

    try:
        conn = psycopg2.connect(_get_db_connection_string())
        cur = conn.cursor()

        cur.execute("""
            UPDATE articles SET
                is_news = %s,
                relevance_score = %s,
                relevance_reason = %s,
                editorial_title = %s,
                editorial_teaser = %s,
                editorial_rewritten = %s,
                telegram_post_text = %s,
                telegram_cover_image = %s,
                telegraph_content_html = %s,
                status = %s,
                updated_at = %s,
                article_metadata = %s
            WHERE id = %s
        """, (
            getattr(processed_article, 'is_news', False),
            processed_article.relevance_score,
            getattr(processed_article, 'relevance_reason', None),
            getattr(processed_article, 'editorial_title', None),
            getattr(processed_article, 'editorial_teaser', None),
            getattr(processed_article, 'editorial_rewritten', None),
            getattr(processed_article, 'telegram_post_text', None),
            getattr(processed_article, 'telegram_cover_image', None),
            getattr(processed_article, 'telegraph_content_html', None),
            'processed',
            datetime.utcnow(),
            json.dumps(getattr(processed_article, 'metadata', {}) or {}, ensure_ascii=False),
            str(article_id),
        ))

        updated = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()
        return updated

    except Exception as e:
        logger.error(f"DB update ошибка для {article_id}: {e}")
        return False


# =========================================================================
# Основной конвейер
# =========================================================================

async def process_existing_articles(
        limit: Optional[int] = None,
        days: Optional[int] = None,
        reprocess_all: bool = False,
        min_relevance: int = 5,
        debug: bool = False,
        provider: Optional[str] = None,
        strategy: Optional[str] = None,
        no_fallback: bool = False,
        urls: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
):
    """
    AI обработка существующих статей из БД.

    Args:
        limit: Макс. количество статей
        days: Только статьи за последние N дней
        reprocess_all: Переобработать все (включая уже обработанные)
        min_relevance: Мин. score для Qdrant
        debug: Debug режим
        provider: LLM провайдер
        strategy: Стратегия выбора моделей
        no_fallback: Отключить fallback
        urls: Список URL для обработки конкретных статей
        ids: Список UUID для обработки конкретных статей
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline_start = time.time()

    # Определяем режим
    if urls:
        mode_label = f"По URL ({len(urls)} шт.)"
    elif ids:
        mode_label = f"По ID ({len(ids)} шт.)"
    elif reprocess_all:
        mode_label = "Переобработать всё"
    else:
        mode_label = "Только необработанные"

    # Header
    print(fmt_header("AI PROCESSING PIPELINE v4.2"))
    print(fmt_row("Версия", "4.2.0"))
    print(fmt_row("Запущен", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(fmt_row("Режим", mode_label))
    print(fmt_row("Лимит", limit if limit else "Без лимита"))
    print(fmt_row("Период", f"Последние {days} дней" if days else "Всё время"))
    print(fmt_row("Мин. релевантность", f"{min_relevance}/10"))

    if urls:
        for i, u in enumerate(urls, 1):
            print(fmt_row(f"  URL [{i}]", u[:70]))
    if ids:
        for i, aid in enumerate(ids, 1):
            print(fmt_row(f"  ID [{i}]", aid))

    # --- LLM конфигурация ---
    print(fmt_sub("КОНФИГУРАЦИЯ LLM"))

    from src.config.models_config import get_models_config, reset_models_config

    if provider:
        reset_models_config()

    config = get_models_config(
        provider=provider,
        strategy=strategy,
        enable_fallback=not no_fallback if no_fallback else None,
        force_new=bool(provider)
    )

    print(fmt_row("Провайдер", config.provider_name.upper()))
    print(fmt_row("Стратегия", config.strategy))
    print(fmt_row("Fallback", "ВЫКЛЮЧЕН ⚠️" if not config.enable_fallback else "ВКЛЮЧЁН ✓"))

    if config.enable_fallback:
        chain = config.get_fallback_providers()
        print(fmt_row("Цепочка fallback", " → ".join(chain)))

    # --- Инициализация AI сервисов ---
    print(fmt_sub("ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ"))

    try:
        from src.application.ai_services.orchestrator import AIOrchestrator
        orchestrator = AIOrchestrator(
            provider=provider,
            strategy=strategy,
            enable_fallback=not no_fallback if no_fallback else None,
            enable_validation=True,
            max_retries=2
        )
        logger.info("✓ AIOrchestrator")
    except Exception as e:
        logger.error(f"Ошибка инициализации AIOrchestrator: {e}")
        return

    qdrant = None
    try:
        from src.infrastructure.ai.qdrant_client import QdrantService
        qdrant = QdrantService()
        logger.info("✓ QdrantService")
    except Exception as e:
        logger.info(f"ⓘ Qdrant недоступен (нормально вне Docker)")

    # --- Загрузка статей ---
    print(fmt_header("ЗАГРУЗКА СТАТЕЙ"))

    try:
        article_rows = _load_articles_sync(
            urls=urls, ids=ids, limit=limit,
            days=days, reprocess_all=reprocess_all,
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки из БД: {e}")
        return

    print(fmt_row("Найдено статей", len(article_rows)))

    if not article_rows:
        print(fmt_row("Статус", "Нет статей для обработки"))
        if not urls and not ids and not reprocess_all:
            print(fmt_row("Совет", "Попробуйте --reprocess-all или --url <link>"))
        return

    # Список статей
    print(fmt_sub("ОЧЕРЕДЬ СТАТЕЙ"))
    for idx, row in enumerate(article_rows[:10], 1):
        title = row.get('title', '')
        short = title[:55] + "..." if len(title) > 55 else title
        score = row.get('relevance_score')
        score_str = f" [score={score}]" if score is not None else ""
        print(f"  {idx:2d}. {short}{score_str}")
    if len(article_rows) > 10:
        print(f"  ... и ещё {len(article_rows) - 10}")

    # --- Конвертация в Article entity ---
    from src.domain.entities.article import Article
    from src.domain.value_objects.source_type import SourceType

    def row_to_article(row: Dict) -> Article:
        """Конвертировать строку БД в Article entity."""
        source_str = row.get('source', 'habr')
        try:
            source = SourceType(source_str)
        except ValueError:
            source = SourceType.HABR

        return Article(
            id=row['id'] if isinstance(row['id'], uuid.UUID) else uuid.UUID(str(row['id'])),
            title=row.get('title', ''),
            content=row.get('content', ''),
            url=row.get('url'),
            source=source,
            author=row.get('author'),
            published_at=row.get('published_at'),
            created_at=row.get('created_at', datetime.utcnow()),
            updated_at=row.get('updated_at', datetime.utcnow()),
            is_news=row.get('is_news', False),
            relevance_score=row.get('relevance_score'),
            relevance_reason=row.get('relevance_reason'),
            editorial_title=row.get('editorial_title'),
            editorial_teaser=row.get('editorial_teaser'),
            editorial_rewritten=row.get('editorial_rewritten'),
            tags=row.get('tags') or [],
            hubs=row.get('hubs') or [],
            images=row.get('images') or [],
        )

    articles = [row_to_article(row) for row in article_rows]

    # --- Статистика ---
    stats = {
        'total': len(articles),
        'processed': 0,
        'saved_to_db': 0,
        'saved_to_qdrant': 0,
        'low_relevance': 0,
        'errors': 0,
        'article_times': [],
    }

    # --- AI обработка ---
    print(fmt_header("AI ОБРАБОТКА"))

    pbar = tqdm(total=len(articles), desc="Обработка") if HAS_TQDM and not debug else None

    for i, article in enumerate(articles, 1):
        article_start = time.time()
        short_title = article.title[:40] + "..." if len(article.title) > 40 else article.title

        try:
            logger.info(f"[{i}/{len(articles)}] {short_title}")

            # AI
            ai_start = time.time()
            processed = orchestrator.process_article(
                article,
                verbose=debug,
                min_relevance=min_relevance
            )
            ai_time = time.time() - ai_start

            if processed is None:
                logger.warning(f"[{i}] AI вернул None")
                stats['errors'] += 1
                if pbar:
                    pbar.update(1)
                continue

            score = processed.relevance_score or 0
            stats['processed'] += 1

            # DB save (psycopg2)
            db_start = time.time()
            if _update_article_after_ai(str(article.id), processed):
                stats['saved_to_db'] += 1
                db_time = time.time() - db_start
                logger.info(f"[{i}] ✅ DB OK ({db_time:.2f}s)")
            else:
                logger.warning(f"[{i}] ⚠️ DB: не обновлено")

            # Qdrant
            if qdrant and score >= min_relevance:
                try:
                    qdrant.add_article(
                        str(processed.id), processed.title, processed.content or ""
                    )
                    stats['saved_to_qdrant'] += 1
                except Exception as e:
                    logger.warning(f"[{i}] Qdrant: {e}")
            elif score < min_relevance:
                stats['low_relevance'] += 1

            article_time = time.time() - article_start
            stats['article_times'].append(article_time)

            logger.info(
                f"[{i}] Score: {score}/10 | "
                f"{'НОВОСТЬ' if getattr(processed, 'is_news', False) else 'СТАТЬЯ'} | "
                f"{article_time:.1f}s"
            )

            if pbar:
                pbar.update(1)
                pbar.set_postfix({'score': f"{score}/10", 'time': f"{article_time:.1f}s"})

        except Exception as e:
            stats['errors'] += 1
            logger.error(f"[{i}] Ошибка: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    # --- Результаты ---
    pipeline_time = time.time() - pipeline_start

    print(fmt_header("РЕЗУЛЬТАТЫ"))

    print(fmt_sub("СТАТИСТИКА"))
    print(fmt_row("Провайдер", config.provider_name.upper()))
    print(fmt_row("Режим", mode_label))
    print(fmt_row("Всего статей", stats['total']))
    print(fmt_row("AI обработано", stats['processed']))
    print(fmt_row("Сохранено в БД", stats['saved_to_db']))
    print(fmt_row("В Qdrant", stats['saved_to_qdrant']))
    print(fmt_row("Низкая релевантность", stats['low_relevance']))
    print(fmt_row("Ошибок", stats['errors']))

    print(fmt_sub("ПРОИЗВОДИТЕЛЬНОСТЬ"))
    if stats['article_times']:
        avg_t = sum(stats['article_times']) / len(stats['article_times'])
        print(fmt_row("Среднее время", f"{avg_t:.2f}с"))
        print(fmt_row("Мин. время", f"{min(stats['article_times']):.2f}с"))
        print(fmt_row("Макс. время", f"{max(stats['article_times']):.2f}с"))

    print(fmt_row("Общее время", f"{pipeline_time:.2f}с ({pipeline_time / 60:.1f} мин)"))

    if stats['processed'] > 0:
        throughput = stats['processed'] / pipeline_time
        print(fmt_row("Throughput", f"{throughput:.2f} статей/сек"))

    print(fmt_sub("СТАТУС"))
    if stats['errors'] == 0:
        print(fmt_row("Статус", "✅ УСПЕХ"))
    elif stats['errors'] < stats['total'] * 0.1:
        print(fmt_row("Статус", "⚠️  Незначительные ошибки (<10%)"))
    else:
        print(fmt_row("Статус", "❌ Значительные ошибки"))

    print(fmt_row("Завершён", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 80 + "\n")


# =========================================================================
# CLI
# =========================================================================

def parse_urls(raw: List[str]) -> List[str]:
    result = []
    for item in raw:
        for url in item.split(','):
            url = url.strip()
            if url:
                result.append(url)
    return result


def parse_ids(raw: List[str]) -> List[str]:
    result = []
    for item in raw:
        for aid in item.split(','):
            aid = aid.strip()
            if aid:
                result.append(aid)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='AI обработка существующих статей v4.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Обработать все необработанные статьи
  python %(prog)s

  # Обработать 10 необработанных с Groq
  python %(prog)s --limit 10 --provider groq

  # Конкретная статья по URL
  python %(prog)s --url https://habr.com/ru/news/1004288/

  # Конкретная статья по ID
  python %(prog)s --id 550e8400-e29b-41d4-a716-446655440000

  # Несколько статей по URL
  python %(prog)s --url https://habr.com/ru/articles/111/,https://habr.com/ru/articles/222/

  # Переобработать статьи за неделю
  python %(prog)s --days 7 --reprocess-all --provider google

  # Локальный Ollama
  python %(prog)s --limit 5 --provider ollama --no-fallback
        """
    )

    # Выбор статей
    parser.add_argument('--url', '-u', action='append', default=None,
                        help='URL статьи из БД (можно несколько раз или через запятую)')
    parser.add_argument('--id', action='append', default=None,
                        help='UUID статьи из БД (можно несколько раз или через запятую)')
    parser.add_argument('--limit', type=int, metavar='N',
                        help='Макс. количество статей')
    parser.add_argument('--days', type=int, metavar='N',
                        help='Только статьи за последние N дней')
    parser.add_argument('--reprocess-all', action='store_true',
                        help='Переобработать все статьи (включая уже обработанные)')

    # AI параметры
    parser.add_argument('--min-relevance', type=int, default=5, metavar='N',
                        help='Мин. score для Qdrant (default: 5)')
    parser.add_argument('--provider', '-p',
                        choices=['groq', 'openrouter', 'google', 'ollama'],
                        help='LLM провайдер')
    parser.add_argument('--no-fallback', action='store_true',
                        help='Отключить fallback')
    parser.add_argument('--strategy', '-s',
                        choices=['cost_optimized', 'balanced', 'quality_focused', 'speed_focused'],
                        help='Стратегия выбора моделей')

    # Подключение
    parser.add_argument('--db', help='DATABASE_URL (Supabase pooler URL)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')

    parser.add_argument('--debug', action='store_true',
                        help='Debug режим')

    args = parser.parse_args()

    # --db перезаписывает DATABASE_URL
    if args.db:
        os.environ["DATABASE_URL"] = args.db

    # --verbose = --debug
    if args.verbose:
        args.debug = True

    # Нормализация
    urls = parse_urls(args.url) if args.url else None
    ids = parse_ids(args.id) if args.id else None

    try:
        asyncio.run(process_existing_articles(
            limit=args.limit,
            days=args.days,
            reprocess_all=args.reprocess_all,
            min_relevance=args.min_relevance,
            debug=args.debug,
            provider=args.provider,
            strategy=args.strategy,
            no_fallback=args.no_fallback,
            urls=urls,
            ids=ids,
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Прервано")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# =================================================================
# Примеры:
# =================================================================
#
# Обработать 2 статьи из Supabase через Ollama:
# python process_existing_articles.py --db "postgresql://postgres.xxx:PASS@aws-1-eu-west-1.pooler.supabase.com:6543/postgres" -p ollama --limit 2
#
# Или через env:
# DATABASE_URL="postgresql://..." python process_existing_articles.py -p ollama --limit 3
#
# Через Docker (локальная БД):
# docker compose exec api python process_existing_articles.py -p ollama --limit 5
#
# Конкретная статья:
# python process_existing_articles.py --db "..." -p ollama --url https://habr.com/ru/articles/1006098/
#
# Переобработать всё:
# python process_existing_articles.py --db "..." -p ollama --reprocess-all --limit 10
# =================================================================