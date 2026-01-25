# News Aggregator Pro - Профессиональная Система Агрегации Новостей

**Архитектура**: Hexagonal Architecture + CQRS  
**Язык**: Python 3.11+  
**Фреймворк**: FastAPI

---

## Содержание

1. [Обзор](#обзор)
2. [Ключевые Возможности](#ключевые-возможности)
3. [Multi-Provider LLM](#multi-provider-llm)
4. [API Ключи (Бесплатные)](#api-ключи-бесплатные)
5. [Архитектура](#архитектура)
6. [Технологический Стек](#технологический-стек)
7. [Установка](#установка)
8. [Быстрый Старт](#быстрый-старт)
9. [Конфигурация LLM Профилей](#конфигурация-llm-профилей)
10. [Режимы Обработки Статей](#режимы-обработки-статей)
11. [Ускорение с GPU](#ускорение-с-gpu)
12. [Использование API](#использование-api)
13. [Структура Проекта](#структура-проекта)
14. [Разработка](#разработка)
15. [Мониторинг и Диагностика](#мониторинг-и-диагностика)
16. [Production Deployment](#production-deployment)
17. [Troubleshooting](#troubleshooting)
18. [Roadmap](#roadmap)

---

## Обзор

**News Aggregator Pro** — профессиональная система для сбора, обработки и управления новостным контентом из различных источников с использованием AI-технологий для классификации, оценки релевантности и семантического поиска.

Система построена на принципах **чистой архитектуры** (Hexagonal Architecture) с применением **CQRS паттерна**, что обеспечивает:
- Полную изоляцию бизнес-логики от внешних зависимостей
- Высокую тестируемость всех компонентов
- Легкую расширяемость и масштабируемость
- Независимость от конкретных фреймворков и библиотек

---

## Ключевые Возможности

### Multi-Provider LLM с Auto-Fallback

Автоматическое переключение между провайдерами при rate limit (429):

```
Groq (30 req/min) -> Google Gemini (60 req/min) -> OpenRouter (50 req/day) -> HuggingFace
```

- **Groq** — самый быстрый inference (Llama 3.1 70B)
- **Google Gemini** — высокое качество (Gemini 1.5 Flash)
- **OpenRouter** — доступ к 200+ моделям
- **HuggingFace** — fallback провайдер
- **Ollama** — локальные модели (без лимитов)

**Суммарные бесплатные лимиты: ~16,000 запросов/день**

### Сбор и Обработка Контента

**Автоматический парсинг**
- Сбор статей с Habr.com с поддержкой фильтрации по хабам
- RSS feeds поддержка
- Telegram каналы (в разработке)
- Проверка дубликатов по URL в базе данных
- Расширяемая архитектура для добавления новых источников

**AI-обработка (6-агентный пайплайн)**
1. **Classifier** — классификация контента (новости, статьи, обзоры)
2. **Relevance** — оценка релевантности (0-10 баллов)
3. **Summarizer** — генерация редакторских тизеров (2-3 предложения)
4. **Rewriter** — улучшение заголовков для повышения кликабельности
5. **Style Normalizer** — нормализация стиля текста для единообразия
6. **Quality Validator** — финальная проверка качества

**Интеллектуальное хранение**
- PostgreSQL для реляционного хранения всех статей
- Qdrant для векторного семантического поиска
- Условное сохранение: все статьи в БД, только качественные (score >= 5) в Qdrant
- Redis для кэширования и очередей

### Архитектурные Преимущества

**Hexagonal Architecture**
- Полная изоляция бизнес-логики от infrastructure
- Легкая замена компонентов (смена БД, AI-моделей)
- 100% тестируемость через dependency injection

**CQRS Pattern**
- Разделение команд (запись) и запросов (чтение)
- Оптимизация производительности для каждого типа операций
- Независимое масштабирование компонентов

### API и Интеграции

**RESTful API**
- FastAPI с автоматической документацией (Swagger/OpenAPI)
- Асинхронные endpoints для высокой производительности
- Валидация данных через Pydantic v2

---

## Multi-Provider LLM

### Архитектура

```
+-------------------------------------------------------------------+
|                    MultiProviderWrapper                           |
|                                                                   |
|   Request -> Groq --429?--> Google --429?--> OpenRouter --> HF   |
|               |               |                |            |     |
|               v               v                v            v     |
|          Llama 3.1      Gemini 1.5      Llama 3.3     Mistral    |
|            70B            Flash          70B:free       7B       |
|                                                                   |
|   +---------------------------------------------------------+    |
|   | Cooldown Tracking | Rate Limit Detection | Auto-Retry   |    |
|   +---------------------------------------------------------+    |
+-------------------------------------------------------------------+
```

### Таблица Провайдеров

| Провайдер | Лимит (бесплатно) | Модели | Скорость |
|-----------|-------------------|--------|----------|
| **Groq** | 30 req/min, 14K/day | Llama 3.1 70B, Llama 3.1 8B, Mixtral 8x7B | Fastest |
| **Google** | 60 req/min, 1.5K/day | Gemini 1.5 Flash, Gemini 1.5 Pro | Fast |
| **OpenRouter** | 50 req/day (free models) | Llama 3.3 70B:free, Gemma 3 27B:free | Medium |
| **HuggingFace** | Limited | Mistral 7B, модели <10GB | Slow |
| **Ollama** | Unlimited (local) | Qwen 2.5, Llama, Mistral | Depends on HW |

### Логика Fallback

При получении ошибки 429 (rate limit) или других ошибок:
1. Провайдер помечается в cooldown (65 сек для Groq/Google, 120 сек для OpenRouter)
2. Запрос автоматически перенаправляется на следующий доступный провайдер
3. При дневном лимите — cooldown 1 час
4. Статистика по каждому провайдеру отслеживается

---

## API Ключи (Бесплатные)

**Минимум нужен ОДИН ключ. Рекомендуется получить все для максимальных лимитов.**

### 1. Groq (Рекомендуется)

Самый быстрый inference, щедрые лимиты.

**Лимиты:** 30 req/min, 14,400 req/day  
**Модели:** Llama 3.1 70B, Llama 3.1 8B, Mixtral 8x7B

```
1. Регистрация: https://console.groq.com
2. Перейти в API Keys
3. Create API Key
4. Скопировать ключ (начинается с gsk_...)
```

### 2. Google AI Studio

Высокое качество, большой контекст.

**Лимиты:** 60 req/min, 1,500 req/day  
**Модели:** Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0 Flash

```
1. Перейти: https://aistudio.google.com/apikey
2. Войти в Google аккаунт
3. Create API Key
4. Скопировать ключ (начинается с AIzaSy...)
```

### 3. OpenRouter

Доступ к 200+ моделям, включая бесплатные.

**Лимиты:** 50 req/day (бесплатные модели, общий лимит)  
**Модели:** Llama 3.3 70B:free, Gemma 3 27B:free, Mistral 24B:free

```
1. Регистрация: https://openrouter.ai
2. Перейти в Keys
3. Create Key
4. Скопировать ключ (начинается с sk-or-v1-...)
```

**Важно:** Для бесплатных моделей нужно включить "Free model publication" в настройках приватности: https://openrouter.ai/settings/privacy

### 4. HuggingFace (Fallback)

Используется как последний резерв.

**Лимиты:** Ограниченный бесплатный tier  
**Модели:** Mistral 7B, модели размером <10GB

```
1. Регистрация: https://huggingface.co
2. Settings -> Access Tokens
3. New Token (выбрать Read)
4. Скопировать токен (начинается с hf_...)
```

---

## Архитектура

### Hexagonal Architecture (Ports & Adapters)

```
+-------------------------------------------------------------------+
|                   PRESENTATION LAYER (API)                        |
|                      FastAPI REST API                             |
|                   Middleware, Routes, Schemas                     |
+-----------------------------+-------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                   APPLICATION LAYER                               |
|  +----------------+  +-----------------+  +-----------------+     |
|  |   Commands     |  |    Queries      |  |   AI Agents     |     |
|  |   (CQRS)       |  |    (CQRS)       |  |   Pipeline      |     |
|  +----------------+  +-----------------+  +-----------------+     |
|  +---------------------------------------------------------------+|
|  |            Application Services                               ||
|  |     (Координация между Domain и Infrastructure)               ||
|  +---------------------------------------------------------------+|
+-----------------------------+-------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                      DOMAIN LAYER                                 |
|                   (Чистая бизнес-логика)                          |
|  +-------------+  +----------------+  +--------------------+      |
|  |  Entities   |  | Value Objects  |  |  Domain Services   |      |
|  |  (Article)  |  | (ArticleStatus)|  | (DuplicationCheck) |      |
|  +-------------+  +----------------+  +--------------------+      |
|  +---------------------------------------------------------------+|
|  |           Repository Interfaces (Ports)                       ||
|  |           Domain Events, Specifications                       ||
|  +---------------------------------------------------------------+|
+-----------------------------+-------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                  INFRASTRUCTURE LAYER                             |
|                     (Adapters)                                    |
|  +--------------+  +------------+  +------------------------+     |
|  |  PostgreSQL  |  |   Redis    |  |   Multi-Provider LLM   |     |
|  |  Repository  |  |   Cache    |  | (Groq/Google/OpenRouter)|    |
|  +--------------+  +------------+  +------------------------+     |
|  +---------------------------------------------------------------+|
|  |    Qdrant, Message Queue, Config, Scrapers, Parsers           ||
|  +---------------------------------------------------------------+|
+-------------------------------------------------------------------+
```

### AI Pipeline

```
Article -> Classifier -> Relevance -> Summarizer -> Rewriter -> Normalizer -> Validator -> DB/Qdrant
              |             |            |             |            |             |
              v             v            v             v            v             v
           is_news?    score 0-10    teaser      new_title    style fix    quality OK?
```

### CQRS (Command Query Responsibility Segregation)

**Commands (изменяют состояние)**
- `CreateArticleCommand` - создание новой статьи
- `UpdateArticleCommand` - обновление существующей статьи
- `DeleteArticleCommand` - удаление статьи

**Queries (только чтение)**
- `GetArticleQuery` - получение статьи по ID
- `ListArticlesQuery` - список статей с фильтрацией и пагинацией
- `SearchArticlesQuery` - семантический поиск через Qdrant

---

## Технологический Стек

### Backend
- **Python 3.11+** - основной язык программирования
- **FastAPI** - современный async веб-фреймворк
- **SQLAlchemy 2.0** - ORM с полной поддержкой async
- **Pydantic v2** - валидация и сериализация данных
- **LangChain** - интеграция с LLM провайдерами
- **Alembic** - миграции базы данных

### Базы Данных
- **PostgreSQL 15** - основное реляционное хранилище
- **Redis 7** - кэш и очереди сообщений
- **Qdrant** - векторная база данных для семантического поиска

### AI и LLM
- **Groq** - облачный inference (Llama 3.1 70B)
- **Google Gemini** - облачный inference (Gemini 1.5 Flash)
- **OpenRouter** - доступ к 200+ моделям
- **HuggingFace** - fallback провайдер
- **Ollama** - локальный сервер для запуска LLM
- **nomic-embed-text** - модель для создания эмбеддингов

### Инфраструктура
- **Docker & Docker Compose** - контейнеризация всех компонентов
- **GitHub Actions** - CI/CD пайплайн
- **Nginx** - reverse proxy и load balancer (production)

### Инструменты Разработки
- **pytest** - фреймворк для тестирования
- **black** - автоматическое форматирование кода
- **ruff** - быстрый линтер для Python
- **pre-commit** - git hooks для контроля качества

---

## Установка

### Системные Требования

**Минимальные**
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10/11 + WSL2
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Диск**: 20 GB свободного пространства
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

**Рекомендуемые**
- **CPU**: 8+ cores
- **RAM**: 16-32 GB
- **GPU**: NVIDIA GPU с 8+ GB VRAM (для локального Ollama)
- **Диск**: 100 GB SSD

### Клонирование репозитория

```bash
git clone https://github.com/vergotten/news-aggregator-pro.git
cd news-aggregator-pro
```

### Настройка окружения

```bash
# Создать файл с переменными окружения
cp .env.example .env

# Отредактировать .env файл
nano .env
```

**Основные переменные (.env)**

```env
# ============================================
# LLM PROVIDERS (минимум один обязателен)
# ============================================

# Groq - РЕКОМЕНДУЕТСЯ (самый быстрый, 30 req/min)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# Google Gemini (60 req/min)
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter (50 req/day free)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx

# HuggingFace (fallback)
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# ============================================
# LLM CONFIGURATION
# ============================================
# Профиль: auto_fallback, groq, google, free_openrouter, balanced
LLM_PROFILE=auto_fallback

# ============================================
# DATABASE
# ============================================
POSTGRES_USER=newsaggregator
POSTGRES_PASSWORD=changeme123
POSTGRES_DB=newsaggregator
POSTGRES_PORT=5433

# ============================================
# SERVICES
# ============================================
REDIS_URL=redis://redis:6379/0
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# ============================================
# API
# ============================================
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
```

---

## Быстрый Старт

### 1. Запуск сервисов

```bash
# Установить зависимости LLM провайдеров
pip install langchain-groq langchain-google-genai langchain-huggingface

# Запустить все контейнеры в фоновом режиме
docker-compose up -d

# Проверить статус всех сервисов
docker-compose ps

# Просмотр логов
docker-compose logs -f
```

### 2. Проверка работоспособности

```bash
# Проверить health endpoint API
curl http://localhost:8000/health

# Открыть API документацию в браузере
# http://localhost:8000/docs
```

### 3. Проверка LLM провайдеров

```bash
docker-compose exec api python -c "
from src.infrastructure.ai.llm_provider import get_llm_provider

llm = get_llm_provider('auto')
print('Доступные провайдеры:', llm.get_stats()['available'])
print('Тест:', llm.generate('Скажи привет одним словом'))
print('Использован провайдер:', llm.get_current_provider())
"
```

### 4. Первый запуск обработки статей

```bash
# Запустить парсинг и обработку 10 статей
docker-compose exec api python run_full_pipeline_fixed.py 10

# С фильтрацией по хабам
docker-compose exec api python run_full_pipeline_fixed.py 20 "python,devops"

# Проверить результаты в базе данных
docker-compose exec postgres psql -U newsaggregator -d newsaggregator -c "
SELECT 
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE relevance_score IS NOT NULL) as processed,
    AVG(relevance_score)::numeric(3,1) as avg_score
FROM articles;
"
```

---

## Конфигурация LLM Профилей

Файл `config/models.yaml` определяет профили для разных сценариев использования.

### Доступные Профили

```yaml
# Активный профиль по умолчанию
active_profile: auto_fallback

profiles:
  # Автоматический выбор с fallback (рекомендуется)
  auto_fallback:
    provider: auto
    agents:
      classifier:
        model: "auto"
        temperature: 0.3
      summarizer:
        model: "auto"
        temperature: 0.5

  # Только Groq (самый быстрый)
  groq:
    provider: groq
    agents:
      classifier:
        model: "llama-3.1-8b-instant"
        temperature: 0.3
      summarizer:
        model: "llama-3.1-70b-versatile"
        temperature: 0.5

  # Только Google Gemini
  google:
    provider: google
    agents:
      classifier:
        model: "gemini-1.5-flash"
        temperature: 0.3
      summarizer:
        model: "gemini-1.5-flash"
        temperature: 0.5

  # Бесплатные модели OpenRouter
  free_openrouter:
    provider: openrouter
    agents:
      classifier:
        model: "meta-llama/llama-3.3-70b-instruct:free"
        temperature: 0.3
      summarizer:
        model: "meta-llama/llama-3.3-70b-instruct:free"
        temperature: 0.5

  # Локальный Ollama
  balanced:
    provider: ollama
    agents:
      classifier:
        model: "qwen2.5:14b-instruct-q5_k_m"
        temperature: 0.3
      summarizer:
        model: "qwen2.5:14b-instruct-q5_k_m"
        temperature: 0.5
```

### Переключение профиля

```bash
# Через переменную окружения
export LLM_PROFILE=groq
docker-compose restart api

# Или в .env файле
LLM_PROFILE=auto_fallback
```

### Использование в коде

```python
from src.infrastructure.ai.llm_provider import get_llm_provider

# Автоматический выбор с fallback
llm = get_llm_provider("auto")
response = llm.generate("Привет!")
print(llm.get_current_provider())  # "groq" или другой доступный

# Конкретный провайдер
llm = get_llm_provider("groq", model="llama-3.1-70b-versatile")

# Конкретный провайдер без fallback
llm = get_llm_provider("groq", use_fallback=False)

# Статистика по провайдерам
print(llm.get_stats())
# {
#   'available': ['groq', 'google', 'openrouter'],
#   'current': 'groq',
#   'stats': {'groq': {'success': 10, 'failed': 0, 'in_cooldown': False}, ...}
# }
```

---

## Режимы Обработки Статей

Система поддерживает три режима работы для различных сценариев использования.

### Режим 1: Быстрый Парсинг (без AI обработки)

**Скрипт**: `run_scraper.py`  
**Назначение**: Массовый сбор статей без AI анализа  
**Скорость**: 1-2 секунды на статью  
**Результат**: Статьи сохраняются в PostgreSQL без оценки релевантности

```bash
# Базовое использование - собрать 50 статей
docker-compose exec api python run_scraper.py 50

# С фильтрацией по хабам
docker-compose exec api python run_scraper.py 100 "python,machine-learning,devops"
```

### Режим 2: Полный Конвейер (парсинг + AI + сохранение)

**Скрипт**: `run_full_pipeline_fixed.py`  
**Назначение**: Полный цикл обработки - от парсинга до векторной базы  
**Скорость**: 5-15 секунд на статью (зависит от провайдера)  
**Результат**: Полностью обработанные и проанализированные статьи

```bash
# Базовое использование - 10 статей
docker-compose exec api python run_full_pipeline_fixed.py 10

# С конкретными хабами
docker-compose exec api python run_full_pipeline_fixed.py 20 "python,devops"
```

### Режим 3: Обработка Существующих Статей

**Скрипт**: `process_existing_articles.py`  
**Назначение**: AI анализ статей, уже находящихся в базе данных  
**Скорость**: 5-15 секунд на статью  
**Результат**: Обновление записей в БД с AI-метаданными

```bash
# Обработать все статьи без AI анализа
docker-compose exec api python process_existing_articles.py

# Обработать только 10 статей
docker-compose exec api python process_existing_articles.py --limit 10

# Статьи за последние 7 дней
docker-compose exec api python process_existing_articles.py --days 7
```

### Сравнительная Таблица

| Характеристика | Парсинг | Полный Конвейер | Обработка Существующих |
|---------------|---------|-----------------|----------------------|
| **Источник данных** | Habr.com | Habr.com | PostgreSQL |
| **AI обработка** | Нет | Да | Да |
| **Сохранение в БД** | Да | Да | Да (обновление) |
| **Добавление в Qdrant** | Нет | Условно (score >= 5) | Условно (score >= 5) |

### Производительность по провайдерам (10 статей)

| Провайдер | Время | Качество |
|-----------|-------|----------|
| Groq | ~50 сек | Отличное |
| Google | ~60 сек | Отличное |
| OpenRouter | ~90 сек | Хорошее |
| Ollama (local) | ~5 мин | Хорошее |

---

## Ускорение с GPU

Для локального Ollama можно использовать GPU для ускорения в 3-4 раза.

### Требования

- NVIDIA GPU с 8+ GB VRAM
- NVIDIA драйвера (версия 525+)
- nvidia-container-toolkit

### Настройка

```bash
# Установить nvidia-container-toolkit (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Проверка

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## Использование API

### Swagger UI

Откройте в браузере: **http://localhost:8000/docs**

### Основные Endpoints

```bash
# Health Check
curl http://localhost:8000/health

# Список статей
curl "http://localhost:8000/api/v1/articles/?limit=20"

# С фильтрацией
curl "http://localhost:8000/api/v1/articles/?is_news=true&min_relevance=7"

# Семантический поиск
curl -X POST "http://localhost:8000/api/v1/articles/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "машинное обучение на Python", "limit": 10}'
```

### Python Client

```python
import httpx

BASE_URL = "http://localhost:8000"

# Список статей
response = httpx.get(f"{BASE_URL}/api/v1/articles/", params={"limit": 10})
articles = response.json()

# Поиск
response = httpx.post(
    f"{BASE_URL}/api/v1/articles/search",
    json={"query": "Python машинное обучение", "limit": 5}
)
results = response.json()
```

---

## Структура Проекта

```
news-aggregator-pro/
|
+-- src/                                   # Исходный код
|   +-- api/                              # Presentation Layer (FastAPI)
|   |   +-- routes/                       # HTTP endpoints
|   |   +-- middleware/                   # Middleware
|   |   +-- schemas/                      # Pydantic схемы
|   |
|   +-- application/                      # Application Layer
|   |   +-- commands/                     # CQRS Commands
|   |   +-- queries/                      # CQRS Queries
|   |   +-- services/                     # Application Services
|   |   +-- ai_services/                  # AI Agents Pipeline
|   |       +-- agents/                   # Специализированные агенты
|   |
|   +-- domain/                           # Domain Layer
|   |   +-- entities/                     # Domain Entities
|   |   +-- value_objects/                # Value Objects
|   |   +-- repositories/                 # Repository Interfaces
|   |
|   +-- infrastructure/                   # Infrastructure Layer
|   |   +-- database/                     # PostgreSQL
|   |   +-- ai/                          # LLM Providers
|   |   |   +-- llm_provider.py          # Multi-Provider с Fallback
|   |   +-- vector_store/                # Qdrant
|   |   +-- parsers/                     # Scrapers
|   |
|   +-- main.py                          # Entry Point
|
+-- config/                               # Конфигурация
|   +-- models.yaml                      # LLM профили
|
+-- tests/                                # Тесты
+-- scripts/                              # Утилиты
+-- .github/workflows/                    # CI/CD
|
+-- docker-compose.yml
+-- Dockerfile
+-- requirements.txt
+-- .env.example
+-- README.md
```

---

## Troubleshooting

### Проблема: Все провайдеры возвращают 429

```bash
# Проверить статус провайдеров
docker-compose exec api python -c "
from src.infrastructure.ai.llm_provider import get_llm_provider
llm = get_llm_provider('auto')
print(llm.get_stats())
"

# Подождать 1-2 минуты (cooldown автоматически сбросится)
```

### Проблема: "GROQ_API_KEY не установлен"

```bash
# Проверить .env
cat .env | grep GROQ

# Перезапустить с новыми переменными
docker-compose down && docker-compose up -d
```

### Проблема: "No module named 'langchain_groq'"

```bash
docker-compose exec api pip install langchain-groq langchain-google-genai langchain-huggingface
docker-compose restart api
```

### Просмотр логов

```bash
# Все логи
docker-compose logs -f

# Только API
docker-compose logs api -f

# С фильтрацией
docker-compose logs api -f | grep -E "(ERROR|WARNING|провайдер)"
```

---

## Roadmap

### Текущая Версия (3.0)
- [x] Multi-provider LLM (Groq, Google, OpenRouter, HuggingFace)
- [x] Auto-fallback при rate limit
- [x] 6-агентный AI пайплайн
- [x] GitHub Actions CI/CD
- [x] Hexagonal Architecture
- [x] CQRS pattern
- [x] PostgreSQL + Qdrant + Redis
- [x] Habr.com scraper

### Версия 3.1
- [ ] Telegram scraper
- [ ] RSS feeds scraper
- [ ] WebSocket real-time updates
- [ ] Prometheus metrics

### Версия 4.0
- [ ] React Admin Panel
- [ ] User authentication (JWT)
- [ ] Kubernetes deployment
- [ ] Elasticsearch integration

---

## Лицензия

MIT

---

## Быстрые Ссылки

| Ресурс | URL |
|--------|-----|
| API Docs | http://localhost:8000/docs |
| Groq Console | https://console.groq.com |
| Google AI Studio | https://aistudio.google.com/apikey |
| OpenRouter | https://openrouter.ai/keys |
| HuggingFace Tokens | https://huggingface.co/settings/tokens |