# -*- coding: utf-8 -*-
"""
Добавление telegram, telegraph, seo колонок в таблицу articles.

Revision ID: add_telegram_seo_fields
Revises: (предыдущая миграция)
Create Date: 2026-02-01

Новые колонки:
- telegram_post_text: Готовый HTML пост для Telegram
- telegram_cover_image: URL обложки для Telegram
- telegraph_url: URL статьи в Telegraph
- telegraph_content_html: HTML контент для Telegraph
- seo_title: SEO заголовок
- seo_description: Meta description
- seo_slug: URL-friendly slug
- seo_keywords: Массив ключевых слов
- seo_focus_keyword: Главное ключевое слово
- article_metadata: JSON метаданные (если ещё нет)
- images: Массив URL изображений (если ещё нет)
"""

# ============================================================================
# ВАРИАНТ 1: Через Alembic (если Alembic настроен)
# ============================================================================
# Положить этот файл в: alembic/versions/xxxx_add_telegram_seo_fields.py
# И запустить: alembic upgrade head

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision = 'add_telegram_seo_v3'
down_revision = None  # ← ЗАМЕНИ на ID последней миграции!
branch_labels = None
depends_on = None


def upgrade():
    # Telegram поля
    op.add_column('articles', sa.Column('telegram_post_text', sa.Text(), nullable=True,
                  comment='Ready-to-post Telegram text with HTML formatting'))
    op.add_column('articles', sa.Column('telegram_cover_image', sa.String(2048), nullable=True,
                  comment='Cover image URL for Telegram post'))
    op.add_column('articles', sa.Column('telegraph_url', sa.String(2048), nullable=True,
                  comment='URL of full article in Telegraph'))
    op.add_column('articles', sa.Column('telegraph_content_html', sa.Text(), nullable=True,
                  comment='HTML content for Telegraph publication'))

    # SEO поля
    op.add_column('articles', sa.Column('seo_title', sa.String(200), nullable=True,
                  comment='SEO optimized title (50-60 chars)'))
    op.add_column('articles', sa.Column('seo_description', sa.Text(), nullable=True,
                  comment='Meta description (150-160 chars)'))
    op.add_column('articles', sa.Column('seo_slug', sa.String(500), nullable=True,
                  comment='URL-friendly slug'))
    op.add_column('articles', sa.Column('seo_keywords', ARRAY(sa.String()), nullable=True,
                  comment='SEO keywords array'))
    op.add_column('articles', sa.Column('seo_focus_keyword', sa.String(200), nullable=True,
                  comment='Primary focus keyword'))

    # Metadata JSON (если ещё нет)
    op.add_column('articles', sa.Column('article_metadata', sa.JSON(), nullable=True,
                  comment='Additional metadata as JSON'))

    # Images (если ещё нет)
    op.add_column('articles', sa.Column('images', ARRAY(sa.String()), nullable=True,
                  comment='URLs of images from article'))


def downgrade():
    op.drop_column('articles', 'images')
    op.drop_column('articles', 'article_metadata')
    op.drop_column('articles', 'seo_focus_keyword')
    op.drop_column('articles', 'seo_keywords')
    op.drop_column('articles', 'seo_slug')
    op.drop_column('articles', 'seo_description')
    op.drop_column('articles', 'seo_title')
    op.drop_column('articles', 'telegraph_content_html')
    op.drop_column('articles', 'telegraph_url')
    op.drop_column('articles', 'telegram_cover_image')
    op.drop_column('articles', 'telegram_post_text')
"""

# ============================================================================
# ВАРИАНТ 2: Прямой SQL (если Alembic НЕ настроен)
# ============================================================================
# Запустить внутри Docker:
#   docker-compose exec db psql -U postgres -d news_aggregator -f /tmp/migrate.sql
# Или скопировать SQL и выполнить вручную.

MIGRATION_SQL = """
-- ==========================================================================
-- Миграция: добавление telegram, telegraph, seo, metadata, images полей
-- Дата: 2026-02-01
-- ==========================================================================

-- Telegram поля
ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegram_post_text TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegram_cover_image VARCHAR(2048);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegraph_url VARCHAR(2048);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegraph_content_html TEXT;

-- SEO поля
ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_title VARCHAR(200);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_description TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_slug VARCHAR(500);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_keywords VARCHAR[];
ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_focus_keyword VARCHAR(200);

-- Metadata JSON
ALTER TABLE articles ADD COLUMN IF NOT EXISTS article_metadata JSON;

-- Images (массив URL)
ALTER TABLE articles ADD COLUMN IF NOT EXISTS images VARCHAR[];

-- Комментарии для документации
COMMENT ON COLUMN articles.telegram_post_text IS 'Ready-to-post Telegram text with HTML formatting';
COMMENT ON COLUMN articles.telegram_cover_image IS 'Cover image URL for Telegram post';
COMMENT ON COLUMN articles.telegraph_url IS 'URL of full article in Telegraph (for long articles)';
COMMENT ON COLUMN articles.telegraph_content_html IS 'HTML content for Telegraph publication';
COMMENT ON COLUMN articles.seo_title IS 'SEO optimized title (50-60 chars)';
COMMENT ON COLUMN articles.seo_description IS 'Meta description (150-160 chars)';
COMMENT ON COLUMN articles.seo_slug IS 'URL-friendly slug';
COMMENT ON COLUMN articles.seo_keywords IS 'SEO keywords array';
COMMENT ON COLUMN articles.seo_focus_keyword IS 'Primary focus keyword';
COMMENT ON COLUMN articles.article_metadata IS 'Additional metadata as JSON';
COMMENT ON COLUMN articles.images IS 'URLs of images from article';

-- Проверка
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'articles' 
ORDER BY ordinal_position;
"""

# ============================================================================
# ВАРИАНТ 3: Автоматическое выполнение через Python (рекомендуется)
# ============================================================================

import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)


async def run_migration():
    """
    Выполнить миграцию автоматически через asyncpg.

    Запуск:
        docker-compose exec api python -m src.infrastructure.persistence.migration_add_fields
    Или:
        docker-compose exec api python migrate_db.py
    """
    try:
        import asyncpg
    except ImportError:
        print("❌ asyncpg не установлен. Установите: pip install asyncpg")
        return False

    # Получить URL из ENV или дефолт
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/news_aggregator")

    # asyncpg хочет формат без +asyncpg
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    statements = [
        # Telegram
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegram_post_text TEXT",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegram_cover_image VARCHAR(2048)",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegraph_url VARCHAR(2048)",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS telegraph_content_html TEXT",
        # SEO
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_title VARCHAR(200)",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_description TEXT",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_slug VARCHAR(500)",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_keywords VARCHAR[]",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_focus_keyword VARCHAR(200)",
        # Metadata & Images
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS article_metadata JSON",
        "ALTER TABLE articles ADD COLUMN IF NOT EXISTS images VARCHAR[]",
    ]

    try:
        conn = await asyncpg.connect(db_url)

        print("=" * 60)
        print("🔄 МИГРАЦИЯ БД: добавление telegram/seo/images полей")
        print("=" * 60)

        for stmt in statements:
            try:
                await conn.execute(stmt)
                col_name = stmt.split("IF NOT EXISTS ")[-1].split(" ")[0]
                print(f"  ✅ {col_name}")
            except Exception as e:
                col_name = stmt.split("IF NOT EXISTS ")[-1].split(" ")[0]
                if "already exists" in str(e).lower():
                    print(f"  ⏭️  {col_name} (уже существует)")
                else:
                    print(f"  ❌ {col_name}: {e}")

        # Проверить результат
        columns = await conn.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'articles' 
            ORDER BY ordinal_position
        """)

        print(f"\n{'─' * 60}")
        print(f"📋 Колонки таблицы articles ({len(columns)} всего):")
        print(f"{'─' * 60}")
        for col in columns:
            print(f"  {col['column_name']:<30} {col['data_type']}")

        await conn.close()
        print(f"\n✅ Миграция завершена успешно!")
        return True

    except Exception as e:
        print(f"\n❌ Ошибка миграции: {e}")
        return False


def main():
    """Точка входа для CLI."""
    asyncio.run(run_migration())


if __name__ == "__main__":
    main()