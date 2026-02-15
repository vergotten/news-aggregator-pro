-- init-db.sql
-- Инициализация базы данных
-- v3.1: Все поля включая telegram, telegraph, seo

-- Подключаемся к основной БД
\c news_aggregator;

-- Создаём расширения
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Создаём таблицу articles
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    content TEXT,
    url VARCHAR(2048) UNIQUE,
    source VARCHAR(100) NOT NULL,

    -- Метаданные
    author VARCHAR(255),
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Статус обработки
    status VARCHAR(50) DEFAULT 'pending',
    embedding_status VARCHAR(50) DEFAULT 'pending',
    vector_id VARCHAR(255),

    -- Результаты AI обработки
    is_news BOOLEAN DEFAULT FALSE,
    relevance_score FLOAT,
    relevance_reason TEXT,
    editorial_title VARCHAR(500),
    editorial_teaser TEXT,
    editorial_rewritten TEXT,

    -- Коллекции
    tags TEXT[],
    hubs TEXT[],
    images TEXT[],

    -- Telegram интеграция
    telegram_post_text TEXT,
    telegram_cover_image VARCHAR(2048),
    telegraph_url VARCHAR(2048),
    telegraph_content_html TEXT,

    -- SEO оптимизация
    seo_title VARCHAR(200),
    seo_description TEXT,
    seo_slug VARCHAR(500),
    seo_keywords TEXT[],
    seo_focus_keyword VARCHAR(200),

    -- Метаданные JSON
    article_metadata JSONB DEFAULT '{}'::jsonb
);

-- Индексы
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_is_news ON articles(is_news);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);

-- Комментарии
COMMENT ON TABLE articles IS 'Таблица статей из различных источников';
COMMENT ON COLUMN articles.id IS 'Уникальный идентификатор';
COMMENT ON COLUMN articles.url IS 'URL статьи (уникальный)';
COMMENT ON COLUMN articles.source IS 'Источник: habr, telegram, reddit, medium';
COMMENT ON COLUMN articles.is_news IS 'Флаг новостной статьи';
COMMENT ON COLUMN articles.relevance_score IS 'Оценка релевантности 0-10';
COMMENT ON COLUMN articles.telegram_post_text IS 'Готовый HTML пост для Telegram';
COMMENT ON COLUMN articles.telegram_cover_image IS 'URL обложки для Telegram';
COMMENT ON COLUMN articles.telegraph_url IS 'URL статьи в Telegraph';
COMMENT ON COLUMN articles.telegraph_content_html IS 'HTML контент для Telegraph';
COMMENT ON COLUMN articles.seo_title IS 'SEO заголовок (50-60 символов)';
COMMENT ON COLUMN articles.seo_description IS 'Meta description (150-160 символов)';
COMMENT ON COLUMN articles.seo_slug IS 'URL-friendly slug';
COMMENT ON COLUMN articles.seo_keywords IS 'Массив ключевых слов';
COMMENT ON COLUMN articles.seo_focus_keyword IS 'Главное ключевое слово';
COMMENT ON COLUMN articles.article_metadata IS 'Дополнительные метаданные в JSON';