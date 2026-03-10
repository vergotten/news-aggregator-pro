#!/usr/bin/env python3
"""
Cleanup Articles v1.0

Очистка и архивация старых статей в Supabase.

Логика:
  1. Опубликованные (status=processed, score>=7) старше N дней → archive + delete
  2. Необработанные (relevance_score IS NULL) старше M дней → delete
  3. Низкий score (score<5) старше M дней → delete
  4. articles_archive хранит URL для проверки дубликатов при парсинге

Запуск:
  python cleanup_articles.py --published-days 30 --unprocessed-days 7
  python cleanup_articles.py --dry-run  # показать что будет удалено
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Подключение к БД из DATABASE_URL."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        # Fallback на компоненты
        user = os.getenv('POSTGRES_USER', 'newsaggregator')
        password = os.getenv('POSTGRES_PASSWORD', 'changeme123')
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5433')
        db = os.getenv('POSTGRES_DB', 'news_aggregator')
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    else:
        db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')

    return psycopg2.connect(db_url)


def archive_and_delete_published(conn, days: int, dry_run: bool = False) -> int:
    """
    Архивировать и удалить опубликованные статьи старше N дней.

    Перемещает в articles_archive (сохраняет URL для проверки дубликатов),
    затем удаляет из articles.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    cur = conn.cursor()

    # Найти кандидатов
    cur.execute("""
        SELECT id, title, url, source, relevance_score, published_at
        FROM articles
        WHERE status = 'processed'
          AND relevance_score >= 5
          AND created_at < %s
    """, (cutoff,))
    rows = cur.fetchall()

    if not rows:
        logger.info(f"Архивация: нет статей старше {days} дней")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Архивация: {len(rows)} статей")
        for row in rows[:5]:
            logger.info(f"  → {row[1][:50]}... (score={row[4]})")
        if len(rows) > 5:
            logger.info(f"  ... и ещё {len(rows) - 5}")
        return len(rows)

    # Архивировать
    archived = 0
    for row in rows:
        try:
            cur.execute("""
                INSERT INTO articles_archive (id, title, url, source, relevance_score, published_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING
            """, row)
            archived += 1
        except Exception as e:
            logger.warning(f"Ошибка архивации {row[0]}: {e}")

    # Удалить архивированные
    ids = [row[0] for row in rows]
    cur.execute("DELETE FROM articles WHERE id = ANY(%s)", (ids,))
    deleted = cur.rowcount

    conn.commit()
    logger.info(f"Архивация: {archived} архивировано, {deleted} удалено (старше {days} дней)")
    return deleted


def delete_unprocessed(conn, days: int, dry_run: bool = False) -> int:
    """Удалить необработанные статьи старше N дней."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    cur = conn.cursor()

    # Посмотреть сколько
    cur.execute("""
        SELECT COUNT(*) FROM articles
        WHERE relevance_score IS NULL
          AND created_at < %s
    """, (cutoff,))
    count = cur.fetchone()[0]

    if count == 0:
        logger.info(f"Необработанные: нет статей старше {days} дней")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Удаление необработанных: {count} статей старше {days} дней")
        return count

    cur.execute("""
        DELETE FROM articles
        WHERE relevance_score IS NULL
          AND created_at < %s
    """, (cutoff,))
    deleted = cur.rowcount
    conn.commit()

    logger.info(f"Необработанные: удалено {deleted} (старше {days} дней)")
    return deleted


def delete_low_score(conn, days: int, min_score: float = 5.0, dry_run: bool = False) -> int:
    """Удалить статьи с низким score старше N дней."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM articles
        WHERE relevance_score IS NOT NULL
          AND relevance_score < %s
          AND created_at < %s
    """, (min_score, cutoff))
    count = cur.fetchone()[0]

    if count == 0:
        logger.info(f"Низкий score: нет статей (score<{min_score}) старше {days} дней")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Удаление низкий score: {count} статей (score<{min_score}, старше {days} дней)")
        return count

    cur.execute("""
        DELETE FROM articles
        WHERE relevance_score IS NOT NULL
          AND relevance_score < %s
          AND created_at < %s
    """, (min_score, cutoff))
    deleted = cur.rowcount
    conn.commit()

    logger.info(f"Низкий score: удалено {deleted} (score<{min_score}, старше {days} дней)")
    return deleted


def get_stats(conn) -> dict:
    """Текущая статистика БД."""
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM articles")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM articles WHERE relevance_score IS NULL")
    pending = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM articles WHERE status = 'processed'")
    processed = cur.fetchone()[0]

    archive_count = 0
    try:
        cur.execute("SELECT COUNT(*) FROM articles_archive")
        archive_count = cur.fetchone()[0]
    except Exception:
        conn.rollback()

    return {
        'total': total,
        'pending': pending,
        'processed': processed,
        'archived': archive_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Очистка и архивация старых статей v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Стандартная очистка
  python %(prog)s --published-days 30 --unprocessed-days 7

  # Посмотреть что будет удалено (без удаления)
  python %(prog)s --dry-run

  # Агрессивная очистка
  python %(prog)s --published-days 14 --unprocessed-days 3

  # Только статистика
  python %(prog)s --stats-only
        """
    )

    parser.add_argument('--published-days', type=int, default=30,
                        help='Удалить опубликованные старше N дней (default: 30)')
    parser.add_argument('--unprocessed-days', type=int, default=7,
                        help='Удалить необработанные старше N дней (default: 7)')
    parser.add_argument('--min-score', type=float, default=5.0,
                        help='Порог "низкого" score (default: 5.0)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Показать что будет удалено, без удаления')
    parser.add_argument('--stats-only', action='store_true',
                        help='Только показать статистику')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        conn = get_db_connection()
    except Exception as e:
        logger.error(f"Не удалось подключиться к БД: {e}")
        sys.exit(1)

    # Статистика до
    stats_before = get_stats(conn)
    print(f"\n{'=' * 60}")
    print(f"🧹 CLEANUP ARTICLES v1.0")
    print(f"{'=' * 60}")
    print(f"  Статей в БД:        {stats_before['total']}")
    print(f"  Ожидают AI:         {stats_before['pending']}")
    print(f"  Обработано:         {stats_before['processed']}")
    print(f"  В архиве:           {stats_before['archived']}")
    if args.dry_run:
        print(f"  Режим:              DRY RUN (без удаления)")
    print(f"{'=' * 60}\n")

    if args.stats_only:
        conn.close()
        return

    # Очистка
    total_deleted = 0

    total_deleted += archive_and_delete_published(
        conn, days=args.published_days, dry_run=args.dry_run
    )
    total_deleted += delete_unprocessed(
        conn, days=args.unprocessed_days, dry_run=args.dry_run
    )
    total_deleted += delete_low_score(
        conn, days=args.unprocessed_days, min_score=args.min_score, dry_run=args.dry_run
    )

    # Статистика после
    if not args.dry_run:
        stats_after = get_stats(conn)
        print(f"\n{'=' * 60}")
        print(f"✅ РЕЗУЛЬТАТЫ")
        print(f"{'=' * 60}")
        print(f"  Удалено:            {total_deleted}")
        print(f"  Осталось в БД:      {stats_after['total']}")
        print(f"  В архиве:           {stats_after['archived']}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n  [DRY RUN] Было бы удалено: {total_deleted}")
        print(f"  Запустите без --dry-run для удаления\n")

    conn.close()


if __name__ == '__main__':
    main()