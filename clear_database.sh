#!/bin/bash

echo "================================================================================"
echo "                    ОЧИСТКА БАЗЫ ДАННЫХ"
echo "================================================================================"
echo ""

# Подтверждение
read -p "Удалить ВСЕ данные (PostgreSQL + Qdrant)? Введите 'DELETE': " -r
echo ""

if [[ $REPLY != "DELETE" ]]; then
    echo "Операция отменена"
    exit 0
fi

# ===============================================================================
# ПРОВЕРКА И ЗАПУСК СЕРВИСОВ
# ===============================================================================

echo "================================================================================"
echo "ПРОВЕРКА СЕРВИСОВ"
echo "================================================================================"

# Проверка PostgreSQL
if docker-compose ps postgres | grep -q "Up"; then
    echo "  PostgreSQL: запущен"
else
    echo "  PostgreSQL: не запущен, запускаем..."
    docker-compose up -d postgres
    sleep 5
    echo "  PostgreSQL: запущен"
fi

# Проверка Qdrant
if docker-compose ps qdrant | grep -q "Up"; then
    echo "  Qdrant: запущен"
else
    echo "  Qdrant: не запущен, запускаем..."
    docker-compose up -d qdrant
    sleep 3
    echo "  Qdrant: запущен"
fi

# Проверка API
if docker-compose ps api | grep -q "Up"; then
    echo "  API: запущен"
else
    echo "  API: не запущен, запускаем..."
    docker-compose up -d api
    sleep 5
    echo "  API: запущен"
fi

echo ""

# ===============================================================================
# ШАГ 1: POSTGRESQL - УДАЛЕНИЕ И ПЕРЕСОЗДАНИЕ БД
# ===============================================================================

echo "================================================================================"
echo "ШАГ 1/2: POSTGRESQL"
echo "================================================================================"

echo "  Удаление старой базы данных..."
docker-compose exec -T postgres psql -U newsaggregator -d postgres << 'SQL' > /dev/null 2>&1
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'newsaggregator' AND pid <> pg_backend_pid();
DROP DATABASE IF EXISTS newsaggregator;
CREATE DATABASE newsaggregator;
SQL

echo "  База данных пересоздана"

echo "  Создание таблиц..."
docker-compose exec -T postgres psql -U newsaggregator -d newsaggregator << 'SQL' > /dev/null 2>&1
-- Таблица статей
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    hub TEXT,
    author TEXT,
    published_at TIMESTAMP,

    -- AI поля
    category TEXT,
    relevance_score FLOAT,
    editorial_intro TEXT,
    editorial_rewritten TEXT,
    editorial_title TEXT,

    -- Метаданные
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Таблица embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    embedding_vector BYTEA,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Индексы
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_articles_hub ON articles(hub);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_relevance ON articles(relevance_score);
CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_article_id ON embeddings(article_id);

-- Триггер для updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_articles_updated_at ON articles;
CREATE TRIGGER update_articles_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
SQL

if [ $? -eq 0 ]; then
    echo "  Таблицы созданы успешно"
else
    echo "  ОШИБКА: Не удалось создать таблицы"
    exit 1
fi

echo ""

# ===============================================================================
# ШАГ 2: QDRANT - УДАЛЕНИЕ ВСЕХ КОЛЛЕКЦИЙ
# ===============================================================================

echo "================================================================================"
echo "ШАГ 2/2: QDRANT"
echo "================================================================================"

docker-compose exec -T api python << 'PYTHON'
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

try:
    client = QdrantClient(host="qdrant", port=6333)

    # Удалить все коллекции
    collections = client.get_collections().collections
    if len(collections) > 0:
        print(f"  Найдено коллекций: {len(collections)}")
        for c in collections:
            print(f"  Удаление: {c.name}")
            client.delete_collection(c.name)
    else:
        print("  Коллекций не найдено")

    # Создать коллекцию articles
    print("  Создание коллекции 'articles'...")
    client.create_collection(
        collection_name="articles",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    print("  Qdrant очищен успешно")

except Exception as e:
    print(f"  ОШИБКА: {e}")
    exit(1)
PYTHON

echo ""

# ===============================================================================
# ПРОВЕРКА РЕЗУЛЬТАТА
# ===============================================================================

echo "================================================================================"
echo "ПРОВЕРКА РЕЗУЛЬТАТА"
echo "================================================================================"

# Количество статей в PostgreSQL
PG_COUNT=$(docker-compose exec -T postgres psql -U newsaggregator -d newsaggregator -t -c "SELECT COUNT(*) FROM articles;" 2>/dev/null | tr -d ' ')

# Количество таблиц
TABLES=$(docker-compose exec -T postgres psql -U newsaggregator -d newsaggregator -t -c "\dt" 2>/dev/null | grep -c "articles\|embeddings")

# Количество векторов в Qdrant
QDRANT_COUNT=$(docker-compose exec -T api python -c "
from qdrant_client import QdrantClient
try:
    client = QdrantClient(host='qdrant', port=6333)
    info = client.get_collection('articles')
    print(info.points_count, end='')
except Exception as e:
    print('0', end='')
" 2>/dev/null)

# Убрать пробелы
QDRANT_COUNT=$(echo "$QDRANT_COUNT" | tr -d ' ')
if [ -z "$QDRANT_COUNT" ]; then
    QDRANT_COUNT=0
fi

echo "  PostgreSQL:"
echo "    Таблиц: $TABLES (articles, embeddings)"
echo "    Статей: $PG_COUNT"
echo ""
echo "  Qdrant:"
echo "    Векторов: $QDRANT_COUNT"
echo ""

if [ "$PG_COUNT" = "0" ] && [ "$QDRANT_COUNT" = "0" ] && [ "$TABLES" -ge 2 ]; then
    echo "================================================================================"
    echo "                    УСПЕХ: БАЗА ДАННЫХ ОЧИЩЕНА"
    echo "================================================================================"
    echo ""
    echo "Готово к обработке статей:"
    echo "  docker-compose exec api python run_full_pipeline.py 10"
    echo ""
else
    echo "================================================================================"
    echo "                    ВНИМАНИЕ: ПРОВЕРКА НЕ ПРОШЛА"
    echo "================================================================================"
    echo ""
    echo "Детали:"
    echo "  PostgreSQL статей: $PG_COUNT (ожидалось: 0)"
    echo "  Qdrant векторов: $QDRANT_COUNT (ожидалось: 0)"
    echo "  PostgreSQL таблиц: $TABLES (ожидалось: >= 2)"
    echo ""
fi