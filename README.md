# News Aggregator Pro - Профессиональная Система Агрегации Новостей

**Архитектура**: Hexagonal Architecture + CQRS  
**Язык**: Python 3.11+  
**Фреймворк**: FastAPI

---

## Содержание

1. [Обзор](#обзор)
2. [Ключевые Возможности](#ключевые-возможности)
3. [Архитектура](#архитектура)
4. [Технологический Стек](#технологический-стек)
5. [Установка](#установка)
6. [Быстрый Старт](#быстрый-старт)
7. [Режимы Обработки Статей](#режимы-обработки-статей)
8. [Ускорение с GPU](#ускорение-с-gpu)
9. [Использование API](#использование-api)
10. [Структура Проекта](#структура-проекта)
11. [Разработка](#разработка)
12. [Мониторинг и Диагностика](#мониторинг-и-диагностика)
13. [Production Deployment](#production-deployment)
14. [Troubleshooting](#troubleshooting)
15. [Roadmap](#roadmap)
16. [Контакты](#контакты)

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

### Сбор и Обработка Контента

**Автоматический парсинг**
- Сбор статей с Habr.com с поддержкой фильтрации по хабам
- Проверка дубликатов по URL в базе данных
- Расширяемая архитектура для добавления новых источников (Telegram, Reddit, Medium)

**AI-обработка с использованием LLM**
- Классификация контента (новости, статьи, обзоры, обсуждения)
- Оценка релевантности (0-10 баллов) для фильтрации качественного контента
- Автоматическая генерация редакторских тизеров (2-3 предложения)
- Улучшение заголовков для повышения кликабельности
- Нормализация стиля текста для единообразия

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

**Event-Driven**
- Асинхронная обработка через Redis
- Слабая связность компонентов
- Возможность построения event sourcing

### API и Интеграции

**RESTful API**
- FastAPI с автоматической документацией (Swagger/OpenAPI)
- Асинхронные endpoints для высокой производительности
- Валидация данных через Pydantic v2

**CMS и Автоматизация**
- Directus для управления контентом через web-интерфейс
- n8n для построения автоматизированных workflow
- Готовые интеграции с внешними сервисами

---

## Архитектура

### Hexagonal Architecture (Ports & Adapters)

```
┌─────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER (API)                      │
│                      FastAPI REST API                           │
│                   Middleware, Routes, Schemas                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                             │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Commands     │  │    Queries      │  │    Handlers     │ │
│  │   (CQRS)       │  │    (CQRS)       │  │                 │ │
│  └────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐│
│  │            Application Services                            ││
│  │     (Координация между Domain и Infrastructure)            ││
│  └────────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DOMAIN LAYER                               │
│                   (Чистая бизнес-логика)                        │
│  ┌─────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  Entities   │  │ Value Objects  │  │  Domain Services   │  │
│  │  (Article)  │  │ (ArticleStatus)│  │ (DuplicationCheck) │  │
│  └─────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           Repository Interfaces (Ports)                 │  │
│  │           Domain Events, Specifications                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                           │
│                     (Adapters)                                  │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  PostgreSQL  │  │   Redis    │  │   External APIs      │   │
│  │  Repository  │  │   Cache    │  │  (Ollama, Qdrant)    │   │
│  └──────────────┘  └────────────┘  └──────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      Message Queue, Config, Scrapers, AI Clients         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
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

**Преимущества**
- Оптимизация производительности (разные модели для чтения/записи)
- Независимое масштабирование операций чтения и записи
- Упрощение сложной бизнес-логики

### Применённые Паттерны

- **Repository Pattern** - абстракция доступа к данным
- **Factory Pattern** - создание сложных объектов
- **Strategy Pattern** - различные стратегии AI-обработки
- **Dependency Injection** - через FastAPI Depends
- **Singleton** - для глобальных сервисов (конфигурация)
- **Observer Pattern** - для доменных событий
- **Unit of Work** - для транзакционности

---

## Технологический Стек

### Backend
- **Python 3.11+** - основной язык программирования
- **FastAPI** - современный async веб-фреймворк
- **SQLAlchemy 2.0** - ORM с полной поддержкой async
- **Pydantic v2** - валидация и сериализация данных
- **Alembic** - миграции базы данных

### Базы Данных
- **PostgreSQL 15** - основное реляционное хранилище
- **Redis 7** - кэш и очереди сообщений
- **Qdrant** - векторная база данных для семантического поиска

### AI и Machine Learning
- **Ollama** - локальный сервер для запуска LLM
- **Qwen 2.5 14B** - основная языковая модель (quantized q5_k_m)
- **nomic-embed-text** - модель для создания эмбеддингов
- Поддержка GPU (NVIDIA CUDA) для ускорения обработки

### Инфраструктура
- **Docker & Docker Compose** - контейнеризация всех компонентов
- **Directus** - headless CMS для управления контентом
- **n8n** - платформа для автоматизации workflow
- **Nginx** - reverse proxy и load balancer (production)

### Инструменты Разработки
- **pytest** - фреймворк для тестирования
- **black** - автоматическое форматирование кода
- **mypy** - статическая проверка типов
- **ruff** - быстрый линтер для Python
- **pre-commit** - git hooks для контроля качества

---

## Установка

### Системные Требования

**Минимальные**
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10/11 + WSL2
- **CPU**: 4 cores
- **RAM**: 8 GB (16 GB рекомендуется для комфортной работы с LLM)
- **Диск**: 20 GB свободного пространства
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

**Рекомендуемые для Production**
- **CPU**: 8+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU с 8+ GB VRAM (для ускорения AI обработки)
- **Диск**: 100 GB SSD

### Установка Docker (если не установлен)

**Ubuntu/Debian**
```bash
# Обновить пакеты
sudo apt-get update

# Установить зависимости
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Добавить GPG ключ Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Добавить репозиторий
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Установить Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Добавить пользователя в группу docker
sudo usermod -aG docker $USER
newgrp docker

# Проверить установку
docker --version
docker compose version
```

**macOS**
```bash
# Установить через Homebrew
brew install --cask docker

# Или скачать Docker Desktop с официального сайта
# https://www.docker.com/products/docker-desktop
```

**Windows**
```powershell
# Установить WSL2
wsl --install

# Установить Docker Desktop для Windows
# https://www.docker.com/products/docker-desktop

# Включить WSL2 интеграцию в настройках Docker Desktop
```

---

## Быстрый Старт

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd news-aggregator-pro
```

### 2. Настройка окружения

```bash
# Создать файл с переменными окружения
cp .env.example .env

# Отредактировать .env файл (опционально)
nano .env
```

**Основные переменные (.env)**
```env
# Database
POSTGRES_USER=newsaggregator
POSTGRES_PASSWORD=changeme123
POSTGRES_DB=newsaggregator
POSTGRES_PORT=5433

# Redis
REDIS_URL=redis://redis:6379/0

# Qdrant
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Ollama
OLLAMA_PORT=11434
OLLAMA_BASE_URL=http://ollama:11434

# Directus CMS
DIRECTUS_PORT=8055
DIRECTUS_ADMIN_EMAIL=admin@example.com
DIRECTUS_ADMIN_PASSWORD=changeme123
DIRECTUS_KEY=your-random-key-here
DIRECTUS_SECRET=your-random-secret-here

# API
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
```

### 3. Запуск всех сервисов

```bash
# Запустить все контейнеры в фоновом режиме
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Проверить статус всех сервисов
docker-compose ps
```

### 4. Инициализация Ollama (первый запуск)

```bash
# Загрузить AI модели (выполняется автоматически при первом запуске)
# Процесс занимает 5-10 минут в зависимости от скорости интернета

# Проверить статус загрузки
docker-compose logs ollama-init -f

# После завершения проверить доступные модели
docker-compose exec ollama ollama list
```

**Ожидаемый вывод**
```
NAME                              ID              SIZE      MODIFIED
qwen2.5:14b-instruct-q5_k_m      abc123def456    8.9 GB    2 minutes ago
nomic-embed-text:latest          xyz789uvw012    274 MB    2 minutes ago
```

### 5. Проверка работоспособности

```bash
# Проверить health endpoint API
curl http://localhost:8000/health

# Ожидаемый ответ
# {"status": "healthy", "timestamp": "2026-01-12T14:30:00Z"}

# Открыть API документацию в браузере
# http://localhost:8000/docs

# Открыть Directus CMS
# http://localhost:8055
# Email: admin@example.com
# Password: changeme123
```

### 6. Первый запуск обработки статей

```bash
# Запустить парсинг и обработку 10 статей
docker-compose exec api python run_full_pipeline.py 10

# Проверить результаты в базе данных
docker-compose exec postgres psql -U newsaggregator -d newsaggregator -c "SELECT COUNT(*) FROM articles;"
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

# С детальным выводом
docker-compose exec api python run_scraper.py 50 --verbose

# Помощь по параметрам
docker-compose exec api python run_scraper.py --help
```

**Что происходит**
- Парсинг статей с Habr.com
- Проверка дубликатов по URL в базе данных
- Сохранение новых статей в PostgreSQL
- Поле `relevance_score` остаётся `NULL`

**Когда использовать**
- Массовый сбор данных для последующей обработки
- Тестирование парсера
- Ограниченные вычислительные ресурсы
- Необходимость быстро собрать большой объём контента

---

### Режим 2: Полный Конвейер (парсинг + AI + сохранение)

**Скрипт**: `run_full_pipeline.py`  
**Назначение**: Полный цикл обработки - от парсинга до векторной базы  
**Скорость**: 650-500 секунд на статью (CPU) или 40-60 секунд (GPU)  
**Результат**: Полностью обработанные и проанализированные статьи

```bash
# Базовое использование - 10 статей из всех хабов
docker-compose exec api python run_full_pipeline.py 10

# С конкретными хабами
docker-compose exec api python run_full_pipeline.py 20 "python,devops"

# С debug логированием (видны все шаги AI)
docker-compose exec api python run_full_pipeline.py 10 --debug

# Изменить порог релевантности для Qdrant
docker-compose exec api python run_full_pipeline.py 10 --min-relevance=7

# Комбинация параметров
docker-compose exec api python run_full_pipeline.py 30 "machine-learning,ai" --debug --min-relevance=5
```

**Фазы обработки**
```
ФАЗА 1: ПАРСИНГ СТАТЕЙ
├── Подключение к Habr.com
├── Парсинг с буфером 3x (для учёта дубликатов)
└── Время: 2-5 секунд

ФАЗА 2: ВАЛИДАЦИЯ В БАЗЕ ДАННЫХ
├── Извлечение URL для проверки
├── Массовый запрос существующих URL
├── Фильтрация новых статей
└── Время: 0.1-0.5 секунд

ФАЗА 3: AI ОБРАБОТКА И СОХРАНЕНИЕ
├── Классификация контента (новость/статья/обзор)
├── Оценка релевантности (0-10 баллов)
├── Генерация редакторского тизера
├── Улучшение заголовка
├── Нормализация стиля
├── Сохранение в PostgreSQL
└── Условное добавление в Qdrant (если score >= 5)
    Время: 15-20 мин/статья (CPU) или 2-5 мин/статья (GPU)
```

**Логика сохранения**
- **PostgreSQL**: ВСЕ статьи сохраняются, независимо от оценки
- **Qdrant**: Только статьи с `relevance_score >= min_relevance` (по умолчанию 5)

**Когда использовать**
- Регулярная обработка новых статей
- Production workflow
- Scheduled cron jobs (ежедневная/еженедельная обработка)
- Необходим полный анализ контента

**Пример вывода**
```
================================================================================
ПОЛНЫЙ КОНВЕЙЕР ОБРАБОТКИ СТАТЕЙ - PRODUCTION MODE
================================================================================
  Версия:...............................................................2.0.0
  Запущен:............................................2026-01-12 14:30:00
  Лимит статей:..........................................................10
  Целевые хабы:....................................................python
  Мин. релевантность:.................................................5/10

--------------------------------------------------------------------------------
СИСТЕМА: ВСЕ СЕРВИСЫ ОНЛАЙН
--------------------------------------------------------------------------------

================================================================================
ФАЗА 1: ПАРСИНГ СТАТЕЙ
================================================================================
2026-01-12 14:30:02 | INFO     | Спарсено статей: 30

================================================================================
ФАЗА 2: ВАЛИДАЦИЯ В БАЗЕ ДАННЫХ
================================================================================
  Спарсено статей:......................................................30
  Уже в базе данных:....................................................15
  Новых статей:.........................................................15
  К обработке:..........................................................10

================================================================================
ФАЗА 3: AI ОБРАБОТКА И СОХРАНЕНИЕ
================================================================================
2026-01-12 14:30:05 | INFO     | Обработка статьи 1/10
2026-01-12 14:30:20 | INFO     | AI конвейер завершён за 15.2с
2026-01-12 14:30:20 | INFO     | Оценка релевантности: 8/10
2026-01-12 14:30:21 | INFO     | Статья добавлена в Qdrant

Обработка статей: 100% |████████████| 10/10 [02:40<00:00, 16.0с/статья]

================================================================================
СВОДКА ОБРАБОТКИ
================================================================================
  Обработано успешно:...................................................10
  Добавлено в Qdrant:....................................................6
  Только в БД (низкая релевантность):....................................4
  Общее время:.......................................160.5с (2.7 мин)
```

---

### Режим 3: Обработка Существующих Статей

**Скрипт**: `process_existing_articles.py`  
**Назначение**: AI анализ статей, уже находящихся в базе данных  
**Скорость**: 15-20 секунд на статью (CPU) или 4-6 секунд (GPU)  
**Результат**: Обновление записей в БД с AI-метаданными

```bash
# Обработать все статьи без AI анализа
docker-compose exec api python process_existing_articles.py

# Обработать только 10 статей
docker-compose exec api python process_existing_articles.py --limit 10

# Статьи за последние 7 дней
docker-compose exec api python process_existing_articles.py --days 7

# Переобработать ВСЕ статьи (даже уже обработанные)
docker-compose exec api python process_existing_articles.py --reprocess-all

# С debug выводом
docker-compose exec api python process_existing_articles.py --limit 5 --debug

# Изменить порог для Qdrant
docker-compose exec api python process_existing_articles.py --min-relevance=7

# Комбинация параметров
docker-compose exec api python process_existing_articles.py --days 30 --limit 50 --min-relevance=6
```

**Что происходит**
- Загрузка статей из PostgreSQL где `relevance_score IS NULL`
- AI обработка каждой статьи
- Обновление записей в PostgreSQL
- Условное добавление в Qdrant (если score >= threshold)

**Когда использовать**
- После массового парсинга через `run_scraper.py`
- Переобработка статей после изменения AI моделей
- Исправление ошибок в предыдущей обработке
- Изменение порога релевантности для Qdrant
- Обновление метаданных старых статей

**Пример вывода**
```
================================================================================
ОБРАБОТКА СУЩЕСТВУЮЩИХ СТАТЕЙ - PRODUCTION MODE
================================================================================
  Лимит:............................................................10
  Режим:......................................Только необработанные

--------------------------------------------------------------------------------
ЗАГРУЗКА СТАТЕЙ ИЗ БД
--------------------------------------------------------------------------------
  Фильтр: relevance_score IS NULL
  Найдено статей: 10

================================================================================
AI ОБРАБОТКА
================================================================================
[1/10] Статья: Как оптимизировать производительность Python...
       AI завершён за 15.2с (оценка: 8/10)
       Сохранено в БД за 0.5с
       Добавлено в Qdrant за 0.3с
       ИТОГО: 16.0с

Обработка статей: 100% |████████████| 10/10 [02:40<00:00]

================================================================================
ИТОГОВАЯ СТАТИСТИКА
================================================================================
  Обработано успешно:...................................................10
  Добавлено в Qdrant:....................................................6
  Только БД:............................................................4
  Общее время:........................................160.5с (2.7 мин)
```

---

### Сравнительная Таблица Режимов

| Характеристика | Парсинг | Полный Конвейер | Обработка Существующих |
|---------------|---------|-----------------|----------------------|
| **Источник данных** | Habr.com | Habr.com | PostgreSQL |
| **AI обработка** | Нет | Да | Да |
| **Сохранение в БД** | Да | Да | Да (обновление) |
| **Добавление в Qdrant** | Нет | Условно (score >= 5) | Условно (score >= 5) |
| **Скорость (CPU)** | 1-2 сек/статья | 15-20 сек/статья | 15-20 сек/статья |
| **Скорость (GPU)** | 1-2 сек/статья | 4-6 сек/статья | 4-6 сек/статья |
| **10 статей (CPU)** | 20 секунд | 3 минуты | 3 минуты |
| **10 статей (GPU)** | 20 секунд | 60 секунд | 60 секунд |
| **Use Case** | Массовый сбор | Регулярная работа | Переобработка |

---

### Рекомендуемые Workflow

#### Workflow 1: Быстрый сбор + Отложенная обработка

**Сценарий**: Массовый сбор контента с последующей обработкой в удобное время

```bash
# Шаг 1: Быстро собрать 100 статей (2-3 минуты)
docker-compose exec api python run_scraper.py 100

# Шаг 2: Обработать их AI позже
# С CPU: 25-30 минут
# С GPU: 8-12 минут
docker-compose exec api python process_existing_articles.py
```

**Преимущества**
- Быстрый сбор контента
- Меньшая нагрузка на источник
- Обработка в удобное время (например, ночью)
- Можно прервать и возобновить

---

#### Workflow 2: Прямая обработка

**Сценарий**: Небольшие объёмы с немедленным результатом

```bash
# Парсинг + AI обработка сразу
docker-compose exec api python run_full_pipeline.py 10
```

**Преимущества**
- Простота (одна команда)
- Немедленный результат
- Подходит для production cron jobs

---

#### Workflow 3: Batch Processing (партиями)

**Сценарий**: Контролируемая обработка больших объёмов

```bash
# Шаг 1: Собрать 100 статей
docker-compose exec api python run_scraper.py 100

# Шаг 2: Обработать партиями по 10
docker-compose exec api python process_existing_articles.py --limit 10
# Повторить 10 раз или использовать скрипт

# Bash скрипт для автоматизации
for i in {1..10}; do
  echo "Обработка партии $i/10"
  docker-compose exec -T api python process_existing_articles.py --limit 10
  sleep 5  # Пауза между партиями
done
```

**Преимущества**
- Контроль над нагрузкой системы
- Легко останавливать и возобновлять
- Меньшее использование памяти
- Возможность мониторинга прогресса

---

### Scheduled Processing (автоматическая обработка)

#### Ежедневная обработка через Cron

```bash
# Редактировать crontab
crontab -e

# Добавить задачу (каждый день в 3:00 ночи)
0 3 * * * cd /path/to/news-aggregator-pro && docker-compose exec -T api python run_full_pipeline.py 50 >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

#### Еженедельная переобработка

```bash
# Каждое воскресенье в 2:00 - переобработать статьи за неделю
0 2 * * 0 cd /path/to/news-aggregator-pro && docker-compose exec -T api python process_existing_articles.py --days 7 --reprocess-all >> logs/weekly.log 2>&1
```

---

### Примеры для Различных Сценариев

#### Только качественный контент (высокий порог)

```bash
# Добавлять в Qdrant только статьи с оценкой >= 7
docker-compose exec api python run_full_pipeline.py 30 --min-relevance=7
```

#### Обработка конкретных хабов

```bash
# Только Python статьи
docker-compose exec api python run_full_pipeline.py 30 "python"

# Несколько тематических хабов
docker-compose exec api python run_full_pipeline.py 50 "python,machine-learning,ai,data-science"
```

#### Переобработка с новыми параметрами

```bash
# Переобработать все статьи за месяц с новым порогом
docker-compose exec api python process_existing_articles.py --days 30 --reprocess-all --min-relevance=6
```

---

## Ускорение с GPU

Для значительного ускорения AI обработки (в 3-4 раза) настройте поддержку NVIDIA GPU.

### Требования

**Оборудование**
- NVIDIA GPU с архитектурой Pascal или новее
- Минимум 8 GB VRAM (рекомендуется 11+ GB)
- Примеры подходящих карт: GTX 1080 Ti, RTX 2080, RTX 3070, RTX 4060 и выше

**Программное обеспечение**
- NVIDIA драйвера (версия 525+)
- nvidia-container-toolkit
- Docker 20.10+

### Проверка Системы

```bash
# Проверить наличие NVIDIA GPU
nvidia-smi

# Ожидаемый вывод
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0               |
|-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce GTX 1080 Ti  On    | 00000000:01:00.0  On   |                  N/A |
| 40%   45C    P2    60W / 250W          |   1024MiB / 11178MiB   |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### Установка nvidia-container-toolkit

#### Ubuntu/Debian

```bash
# 1. Добавить репозиторий NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Установить пакет
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Настроить Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Перезапустить Docker
sudo systemctl restart docker

# 5. Проверить установку
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

#### Arch Linux

```bash
sudo pacman -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Настройка docker-compose.yml

Добавьте GPU поддержку в секцию `ollama`:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: news-aggregator-ollama
    restart: unless-stopped
    
    # GPU Configuration
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    environment:
      OLLAMA_HOST: "0.0.0.0:11434"
      # GPU оптимизация
      OLLAMA_NUM_GPU: -1              # Авто-режим (рекомендуется)
      OLLAMA_GPU_MEMORY: 10GB         # Для GPU с 11GB VRAM
      OLLAMA_GPU_LAYERS: auto         # Гибридный режим GPU+CPU
    
    ports:
      - "11434:11434"
    volumes:
      - ./ollama_models:/root/.ollama
    networks:
      - news-aggregator-network
```

### Применение Изменений

```bash
# 1. Остановить контейнеры
docker-compose down

# 2. Запустить с новой конфигурацией
docker-compose up -d

# 3. Подождать инициализации
sleep 15

# 4. Проверить GPU внутри контейнера
docker-compose exec ollama nvidia-smi
```

**Ожидаемый результат**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0               |
|-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce GTX 1080 Ti  On    | 00000000:01:00.0  On   |                  N/A |
| 40%   45C    P2    60W / 250W          |   9500MiB / 11178MiB   |     95%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### Проверка Работы GPU

```bash
# Запустить тестовую обработку
docker-compose exec api python process_existing_articles.py --limit 1

# В другом терминале мониторить GPU
watch -n 1 nvidia-smi
```

**Что наблюдать**
- **GPU-Util**: должно быть 80-100% во время обработки
- **Memory-Usage**: ~9-10 GB из 11 GB для qwen2.5:14b
- **Power Usage**: близко к максимуму (например, 250W)

### Сравнение Производительности

| Операция | CPU (8 cores) | GPU (GTX 1080 Ti) | Ускорение |
|----------|--------------|-------------------|-----------|
| 1 статья | 15-20 сек | 4-6 сек | 3-4x |
| 10 статей | 2.5-3.5 мин | 50-70 сек | 3-4x |
| 50 статей | 12-18 мин | 4-6 мин | 3x |
| 100 статей | 25-35 мин | 8-12 мин | 3x |

### Troubleshooting GPU

#### Проблема: Docker не видит GPU

```bash
# Проверить что toolkit установлен
dpkg -l | grep nvidia-container-toolkit

# Проверить конфигурацию Docker
cat /etc/docker/daemon.json

# Должно содержать
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}

# Переустановить и перезапустить
sudo apt-get install --reinstall nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Проблема: Ollama не использует GPU

```bash
# Проверить переменные окружения
docker-compose exec ollama env | grep OLLAMA

# Проверить логи Ollama
docker-compose logs ollama | grep -i "cuda\|gpu"

# Должно быть
# CUDA available: true
# GPU 0: GeForce GTX 1080 Ti (11178 MiB)
```

#### Проблема: Out of Memory (OOM)

```bash
# Уменьшить использование VRAM в docker-compose.yml
environment:
  OLLAMA_NUM_GPU: 32           # Меньше слоёв на GPU
  OLLAMA_GPU_MEMORY: 9GB       # Больше запаса

# Перезапустить
docker-compose restart ollama
```

---

## Использование API

### Swagger UI (интерактивная документация)

Откройте в браузере: **http://localhost:8000/docs**

Swagger UI предоставляет:
- Полный список всех endpoints
- Интерактивное тестирование API
- Автоматическую валидацию запросов
- Примеры request/response

### Основные Endpoints

#### Health Check

```bash
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "timestamp": "2026-01-12T14:30:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "ollama": "ready"
  }
}
```

#### Создание Статьи

```bash
curl -X POST "http://localhost:8000/api/v1/articles/" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Новая статья о Python",
    "content": "Содержание статьи...",
    "url": "https://habr.com/ru/articles/123456/",
    "source": "habr",
    "author": "Иван Иванов",
    "tags": ["python", "programming"],
    "hubs": ["python", "development"]
  }'

# Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Новая статья о Python",
  "url": "https://habr.com/ru/articles/123456/",
  "status": "pending",
  "created_at": "2026-01-12T14:30:00Z"
}
```

#### Получение Статьи

```bash
curl "http://localhost:8000/api/v1/articles/550e8400-e29b-41d4-a716-446655440000"

# Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Новая статья о Python",
  "content": "Содержание статьи...",
  "url": "https://habr.com/ru/articles/123456/",
  "source": "habr",
  "author": "Иван Иванов",
  "is_news": true,
  "relevance_score": 8,
  "relevance_reason": "Высококачественная техническая статья...",
  "editorial_title": "Python: Современные Подходы к Разработке",
  "editorial_teaser": "Обзор современных практик...",
  "tags": ["python", "programming"],
  "hubs": ["python", "development"],
  "status": "published",
  "created_at": "2026-01-12T14:30:00Z",
  "updated_at": "2026-01-12T14:35:00Z"
}
```

#### Список Статей с Фильтрацией

```bash
# Все статьи (с пагинацией)
curl "http://localhost:8000/api/v1/articles/?limit=20&offset=0"

# Только новости
curl "http://localhost:8000/api/v1/articles/?is_news=true&limit=50"

# С высокой релевантностью
curl "http://localhost:8000/api/v1/articles/?min_relevance=7&limit=30"

# По конкретному хабу
curl "http://localhost:8000/api/v1/articles/?hub=python&limit=20"

# Комбинация фильтров
curl "http://localhost:8000/api/v1/articles/?is_news=true&min_relevance=8&hub=python&limit=10"
```

#### Поиск Статей (семантический)

```bash
curl -X POST "http://localhost:8000/api/v1/articles/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "машинное обучение на Python",
    "limit": 10,
    "min_score": 0.7
  }'

# Response
{
  "results": [
    {
      "article": {
        "id": "...",
        "title": "Практический ML на Python",
        "editorial_teaser": "..."
      },
      "score": 0.92,
      "relevance": "high"
    }
  ],
  "total": 10
}
```

### Python Client Examples

#### Асинхронный клиент с httpx

```python
import httpx
import asyncio

BASE_URL = "http://localhost:8000"

async def create_article():
    """Создать новую статью"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/articles/",
            json={
                "title": "Test Article",
                "content": "Content here...",
                "url": "https://example.com/test",
                "source": "habr",
                "tags": ["python", "test"]
            }
        )
        return response.json()

async def get_articles(is_news=None, min_relevance=None):
    """Получить список статей с фильтрацией"""
    async with httpx.AsyncClient() as client:
        params = {"limit": 20, "offset": 0}
        if is_news is not None:
            params["is_news"] = is_news
        if min_relevance is not None:
            params["min_relevance"] = min_relevance
        
        response = await client.get(
            f"{BASE_URL}/api/v1/articles/",
            params=params
        )
        return response.json()

async def search_articles(query: str):
    """Семантический поиск статей"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/articles/search",
            json={"query": query, "limit": 10}
        )
        return response.json()

# Использование
async def main():
    # Создать статью
    new_article = await create_article()
    print(f"Created: {new_article['id']}")
    
    # Получить качественные новости
    news = await get_articles(is_news=True, min_relevance=7)
    print(f"Found {len(news['items'])} news articles")
    
    # Поиск
    results = await search_articles("Python машинное обучение")
    print(f"Search found {results['total']} articles")

asyncio.run(main())
```

#### Синхронный клиент с requests

```python
import requests

BASE_URL = "http://localhost:8000"

def get_article(article_id: str):
    """Получить статью по ID"""
    response = requests.get(f"{BASE_URL}/api/v1/articles/{article_id}")
    response.raise_for_status()
    return response.json()

def list_articles(limit=20, offset=0, **filters):
    """Список статей с фильтрацией"""
    params = {"limit": limit, "offset": offset, **filters}
    response = requests.get(f"{BASE_URL}/api/v1/articles/", params=params)
    response.raise_for_status()
    return response.json()

# Использование
articles = list_articles(limit=10, is_news=True, min_relevance=8)
for article in articles['items']:
    print(f"{article['title']} - Score: {article['relevance_score']}")
```

---

## Структура Проекта

```
news-aggregator-pro/
│
├── src/                                   # Исходный код приложения
│   │
│   ├── api/                              # Presentation Layer (API)
│   │   ├── routes/                       # HTTP endpoints
│   │   │   ├── __init__.py
│   │   │   ├── articles.py              # CRUD для статей
│   │   │   ├── search.py                # Семантический поиск
│   │   │   └── health.py                # Health checks
│   │   ├── middleware/                   # Middleware компоненты
│   │   │   ├── __init__.py
│   │   │   ├── error_handler.py         # Обработка ошибок
│   │   │   ├── logging_middleware.py    # Логирование запросов
│   │   │   └── cors.py                  # CORS настройки
│   │   ├── schemas/                      # Pydantic схемы (DTO)
│   │   │   ├── __init__.py
│   │   │   ├── article_schemas.py       # Request/Response модели
│   │   │   └── search_schemas.py        # Схемы поиска
│   │   └── dependencies.py               # Dependency Injection
│   │
│   ├── application/                      # Application Layer
│   │   ├── commands/                     # CQRS Commands (Write)
│   │   │   ├── __init__.py
│   │   │   ├── create_article_command.py
│   │   │   ├── update_article_command.py
│   │   │   └── delete_article_command.py
│   │   ├── queries/                      # CQRS Queries (Read)
│   │   │   ├── __init__.py
│   │   │   ├── get_article_query.py
│   │   │   ├── list_articles_query.py
│   │   │   └── search_articles_query.py
│   │   ├── handlers/                     # Command/Query Handlers
│   │   │   ├── __init__.py
│   │   │   ├── article_command_handler.py
│   │   │   └── article_query_handler.py
│   │   ├── services/                     # Application Services
│   │   │   ├── __init__.py
│   │   │   └── article_service.py       # Координация workflow
│   │   └── ai_services/                  # AI Integration Services
│   │       ├── __init__.py
│   │       ├── orchestrator.py          # AI Orchestrator
│   │       └── agents/                   # Специализированные AI агенты
│   │           ├── classifier_agent.py   # Классификация контента
│   │           ├── relevance_agent.py    # Оценка релевантности
│   │           ├── summarizer_agent.py   # Генерация тизеров
│   │           └── style_normalizer_agent.py
│   │
│   ├── domain/                           # Domain Layer (Бизнес-логика)
│   │   ├── entities/                     # Domain Entities
│   │   │   ├── __init__.py
│   │   │   └── article.py               # Сущность статьи
│   │   ├── value_objects/                # Value Objects (неизменяемые)
│   │   │   ├── __init__.py
│   │   │   ├── article_status.py        # Статус статьи (enum)
│   │   │   └── source_type.py           # Тип источника (enum)
│   │   ├── repositories/                 # Repository Interfaces (Ports)
│   │   │   ├── __init__.py
│   │   │   └── article_repository.py    # Абстракция хранилища
│   │   ├── services/                     # Domain Services
│   │   │   ├── __init__.py
│   │   │   └── duplication_checker.py   # Проверка дубликатов
│   │   └── events/                       # Domain Events
│   │       ├── __init__.py
│   │       └── article_events.py        # События жизненного цикла
│   │
│   ├── infrastructure/                   # Infrastructure Layer (Adapters)
│   │   ├── persistence/                  # Database Implementations
│   │   │   ├── __init__.py
│   │   │   ├── models.py                # SQLAlchemy ORM модели
│   │   │   └── article_repository_impl.py # Реализация Repository
│   │   ├── ai/                          # AI Service Clients
│   │   │   ├── __init__.py
│   │   │   ├── ollama_client.py         # Клиент для Ollama
│   │   │   └── qdrant_client.py         # Клиент для Qdrant
│   │   ├── cache/                       # Caching Layer
│   │   │   ├── __init__.py
│   │   │   └── redis_cache.py           # Redis кэш
│   │   ├── messaging/                    # Message Queue
│   │   │   ├── __init__.py
│   │   │   └── redis_publisher.py       # Event publisher
│   │   └── config/                      # Configuration
│   │       ├── __init__.py
│   │       ├── database.py              # DB connection
│   │       └── settings.py              # App settings
│   │
│   ├── scrapers/                         # Content Scrapers Module
│   │   ├── __init__.py
│   │   ├── base/                        # Base scraper classes
│   │   │   ├── __init__.py
│   │   │   └── base_scraper.py
│   │   ├── habr/                        # Habr.com scraper
│   │   │   ├── __init__.py
│   │   │   ├── scraper_service.py       # Основной сервис
│   │   │   └── parser.py                # HTML парсер
│   │   ├── telegram/                    # Telegram scraper (future)
│   │   └── reddit/                      # Reddit scraper (future)
│   │
│   ├── shared/                           # Shared Kernel
│   │   ├── utils/                       # Утилиты
│   │   │   ├── __init__.py
│   │   │   ├── text_processing.py      # Обработка текста
│   │   │   └── date_utils.py           # Работа с датами
│   │   ├── exceptions/                   # Исключения
│   │   │   ├── __init__.py
│   │   │   ├── domain_exceptions.py     # Domain ошибки
│   │   │   └── infrastructure_exceptions.py
│   │   └── types/                       # Общие типы
│   │       ├── __init__.py
│   │       └── common_types.py
│   │
│   └── main.py                          # Application Entry Point
│
├── tests/                                # Тесты
│   ├── __init__.py
│   ├── unit/                            # Юнит-тесты
│   │   ├── domain/                      # Тесты domain layer
│   │   ├── application/                 # Тесты application layer
│   │   └── infrastructure/              # Тесты infrastructure
│   ├── integration/                     # Интеграционные тесты
│   │   ├── api/                        # Тесты API endpoints
│   │   └── repositories/                # Тесты repositories
│   ├── e2e/                            # End-to-end тесты
│   └── conftest.py                      # Pytest fixtures
│
├── scripts/                              # Utility Scripts
│   ├── run_scraper.py                   # Режим 1: Только парсинг
│   ├── run_full_pipeline.py             # Режим 2: Полный конвейер
│   ├── process_existing_articles.py     # Режим 3: Обработка существующих
│   ├── setup.sh                         # Setup script
│   └── migrate.sh                       # Migration helper
│
├── docker/                               # Docker configuration
│   ├── init-db.sql                      # PostgreSQL initialization
│   ├── Dockerfile.api                   # API container
│   └── Dockerfile.worker               # Worker container
│
├── config/                               # Configuration files
│   ├── models.yaml                      # AI models configuration
│   ├── prompts/                         # LLM prompts
│   │   ├── classifier.txt
│   │   ├── relevance.txt
│   │   └── summarizer.txt
│   └── logging.yaml                     # Logging configuration
│
├── docs/                                 # Документация
│   ├── architecture.md                  # Детальная архитектура
│   ├── api.md                           # API документация
│   ├── deployment.md                    # Deployment guide
│   └── contributing.md                  # Contribution guidelines
│
├── alembic/                              # Database migrations
│   ├── versions/                        # Migration files
│   └── env.py                           # Alembic configuration
│
├── logs/                                 # Log files (gitignored)
├── ollama_models/                        # Ollama model storage
│
├── docker-compose.yml                    # Docker Compose config
├── Dockerfile                            # Main Dockerfile
├── requirements.txt                      # Python dependencies
├── requirements-dev.txt                  # Development dependencies
├── .env.example                          # Example environment file
├── .gitignore                            # Git ignore rules
├── pytest.ini                            # Pytest configuration
├── pyproject.toml                        # Project metadata
├── alembic.ini                           # Alembic configuration
└── README.md                             # This file
```

---

## Разработка

### Настройка Окружения Разработки

```bash
# Создать виртуальное окружение
python -m venv venv

# Активировать
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Установить зависимости разработки
pip install -r requirements-dev.txt

# Установить pre-commit hooks
pre-commit install
```

### Инструменты Качества Кода

```bash
# Форматирование кода (black)
black src/ tests/

# Проверка типов (mypy)
mypy src/

# Линтер (ruff)
ruff check src/ tests/ --fix

# Сортировка импортов (isort)
isort src/ tests/

# Проверка всего сразу
pre-commit run --all-files
```

### Работа с Миграциями БД

```bash
# Создать новую миграцию (автоматическая генерация)
alembic revision --autogenerate -m "Add new field to articles"

# Просмотреть SQL перед применением
alembic upgrade head --sql

# Применить все миграции
alembic upgrade head

# Откатить последнюю миграцию
alembic downgrade -1

# Показать историю миграций
alembic history

# Показать текущую версию
alembic current
```

### Добавление Нового Источника Данных

**Шаг 1**: Создать scraper class

```python
# src/scrapers/newsource/scraper_service.py
from src.scrapers.base.base_scraper import BaseScraper

class NewSourceScraper(BaseScraper):
    def __init__(self):
        super().__init__(source_name="newsource")
    
    async def scrape(self, limit: int = 10) -> List[Dict]:
        # Реализация парсинга
        pass
```

**Шаг 2**: Добавить source type

```python
# src/domain/value_objects/source_type.py
class SourceType(str, Enum):
    HABR = "habr"
    NEWSOURCE = "newsource"  # Новый источник
```

**Шаг 3**: Создать routes (опционально)

```python
# src/api/routes/newsource.py
from fastapi import APIRouter

router = APIRouter(prefix="/newsource", tags=["newsource"])

@router.post("/scrape")
async def scrape_newsource(limit: int = 10):
    # Реализация endpoint
    pass
```

**Шаг 4**: Написать тесты

```python
# tests/integration/scrapers/test_newsource_scraper.py
import pytest
from src.scrapers.newsource.scraper_service import NewSourceScraper

@pytest.mark.asyncio
async def test_scrape_newsource():
    scraper = NewSourceScraper()
    results = await scraper.scrape(limit=5)
    assert len(results) <= 5
    assert all('title' in r for r in results)
```

---

## Мониторинг и Диагностика

### Проверка Статуса Системы

```bash
# Статус всех контейнеров
docker-compose ps

# Использование ресурсов
docker stats

# Логи конкретного сервиса
docker-compose logs api -f
docker-compose logs ollama -f
docker-compose logs postgres -f
```

### Мониторинг База Данных

```bash
# Подключиться к PostgreSQL
docker-compose exec postgres psql -U newsaggregator -d newsaggregator

# Количество статей
SELECT COUNT(*) FROM articles;

# Статистика по обработке
SELECT 
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE relevance_score IS NULL) as unprocessed,
    COUNT(*) FILTER (WHERE relevance_score >= 5) as in_qdrant,
    COUNT(*) FILTER (WHERE is_news = true) as news_articles,
    AVG(relevance_score) as avg_score
FROM articles;

# Последние обработанные статьи
SELECT 
    id, 
    title, 
    relevance_score, 
    is_news,
    updated_at 
FROM articles 
WHERE relevance_score IS NOT NULL 
ORDER BY updated_at DESC 
LIMIT 10;

# Распределение по источникам
SELECT source, COUNT(*) as count 
FROM articles 
GROUP BY source;

# Топ хабов
SELECT 
    unnest(hubs) as hub, 
    COUNT(*) as articles_count 
FROM articles 
GROUP BY hub 
ORDER BY articles_count DESC 
LIMIT 10;
```

### Мониторинг Qdrant

```bash
# Статистика коллекции
curl http://localhost:6333/collections/articles

# Количество векторов
curl http://localhost:6333/collections/articles | jq '.result.points_count'

# Тестовый поиск
curl -X POST http://localhost:6333/collections/articles/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],  # Пример вектора
    "limit": 5
  }'
```

### Просмотр в Directus CMS

1. Откройте http://localhost:8055
2. Войдите с учётными данными (см. секцию Быстрый Старт)
3. Перейдите в коллекцию "Articles"
4. Используйте фильтры:
   - `relevance_score >= 7` - только качественные
   - `is_news = true` - только новости
   - `source = habr` - только с Habr
5. Сортировка по `updated_at DESC` - последние обработанные

### Логирование

**Уровни логирования**
- `DEBUG` - детальная диагностическая информация
- `INFO` - подтверждение что всё работает как ожидается
- `WARNING` - индикация потенциальной проблемы
- `ERROR` - серьёзная проблема, функциональность нарушена

**Просмотр логов**
```bash
# Все логи API
docker-compose logs api

# Последние 100 строк
docker-compose logs api --tail 100

# Follow mode (real-time)
docker-compose logs api -f

# С временными метками
docker-compose logs api -f --timestamps

# Фильтрация по уровню (требует настроенного grep)
docker-compose logs api | grep ERROR
docker-compose logs api | grep WARNING
```

**Логи сохраняются в**
- Внутри контейнера: `/app/logs/`
- На хосте: `./logs/` (если примонтирован volume)

---

## Production Deployment

### Checklist перед Deployment

- [ ] Изменить все пароли по умолчанию в `.env`
- [ ] Настроить HTTPS через Nginx/Traefik
- [ ] Настроить регулярные бэкапы PostgreSQL
- [ ] Настроить мониторинг (Prometheus + Grafana)
- [ ] Настроить логирование в centralised систему (ELK, Loki)
- [ ] Настроить rate limiting в API
- [ ] Проверить security headers
- [ ] Настроить GPU если доступно
- [ ] Настроить автоматические обновления security patches
- [ ] Документировать процедуры восстановления

### Production docker-compose.yml

```yaml
version: '3.8'

services:
  # Override для production
  api:
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - WORKERS=4
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}  # Из secure secrets
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups  # Для бэкапов
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: always

volumes:
  postgres_data:
    driver: local
```

### Environment Variables для Production

```env
# Security
DEBUG=false
SECRET_KEY=<generate-strong-random-key>
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# Database (используйте strong passwords)
POSTGRES_PASSWORD=<strong-password>
POSTGRES_USER=newsaggregator
POSTGRES_DB=newsaggregator

# Redis
REDIS_PASSWORD=<strong-password>

# API
API_WORKERS=4
API_PORT=8000

# Directus
DIRECTUS_ADMIN_PASSWORD=<strong-password>
DIRECTUS_KEY=<random-uuid>
DIRECTUS_SECRET=<random-secret>

# Monitoring
SENTRY_DSN=<your-sentry-dsn>  # Опционально
```

### Nginx Configuration Example

```nginx
# /nginx/nginx.conf
upstream api {
    server api:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs {
        # Restrict access to docs in production
        allow 10.0.0.0/8;  # Internal network
        deny all;
        
        proxy_pass http://api;
    }
}
```

### Автоматические Бэкапы PostgreSQL

```bash
# Создать скрипт бэкапа
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="newsaggregator"

docker-compose exec -T postgres pg_dump -U newsaggregator $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Удалить бэкапы старше 7 дней
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: backup_$DATE.sql.gz"
EOF

chmod +x backup.sh

# Добавить в crontab (каждый день в 2:00)
crontab -e
0 2 * * * /path/to/backup.sh >> /var/log/backup.log 2>&1
```

### Мониторинг с Prometheus

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

---

## Troubleshooting

### Общие Проблемы

#### Проблема: Контейнеры не запускаются

```bash
# Проверить логи
docker-compose logs

# Проверить конфигурацию
docker-compose config

# Пересоздать контейнеры
docker-compose down -v
docker-compose up -d --build
```

#### Проблема: Ollama не отвечает

```bash
# Проверить статус
docker-compose ps ollama

# Логи Ollama
docker-compose logs ollama --tail 50

# Перезапустить
docker-compose restart ollama

# Проверить доступность
curl http://localhost:11434/api/tags
```

#### Проблема: PostgreSQL connection refused

```bash
# Проверить статус
docker-compose ps postgres

# Логи
docker-compose logs postgres --tail 50

# Проверить порт
netstat -an | grep 5433

# Проверить credentials
docker-compose exec postgres psql -U newsaggregator -d newsaggregator -c "SELECT 1;"
```

#### Проблема: Медленная AI обработка

**Причины**
- Использование CPU вместо GPU
- Недостаточно RAM
- Большой размер статей

**Решения**
1. Настроить GPU (см. секцию "Ускорение с GPU")
2. Увеличить ресурсы Docker
3. Обрабатывать меньшими партиями
4. Использовать меньшую модель (7B вместо 14B)

```bash
# Проверить использование ресурсов
docker stats

# Увеличить ресурсы в Docker Desktop (Settings -> Resources)
# CPU: 4+ cores
# Memory: 16+ GB
```

#### Проблема: Out of Memory при обработке

```bash
# Уменьшить batch size
docker-compose exec api python process_existing_articles.py --limit 5

# Или обрабатывать по одной
docker-compose exec api python process_existing_articles.py --limit 1
```

#### Проблема: Дубликаты статей в БД

```bash
# Найти дубликаты
docker-compose exec postgres psql -U newsaggregator -d newsaggregator -c "
SELECT url, COUNT(*) as count
FROM articles
GROUP BY url
HAVING COUNT(*) > 1;
"

# Удалить дубликаты (оставить последний)
docker-compose exec postgres psql -U newsaggregator -d newsaggregator -c "
DELETE FROM articles a
USING articles b
WHERE a.id < b.id 
AND a.url = b.url;
"
```

### Debug Режим

```bash
# Запустить с максимальным выводом
docker-compose exec api python run_full_pipeline.py 1 --debug

# Проверить все переменные окружения
docker-compose exec api env

# Проверить connectivity
docker-compose exec api python -c "
import httpx
print(httpx.get('http://ollama:11434/api/tags').json())
"
```

### Проблемы с GPU

См. секцию "Troubleshooting GPU" в "Ускорение с GPU"

---

## Roadmap

### Текущая Версия
- [x] Hexagonal Architecture implementation
- [x] CQRS pattern
- [x] PostgreSQL integration
- [x] FastAPI REST API
- [x] Docker Compose setup
- [x] Habr.com scraper
- [x] Ollama AI integration
- [x] Qdrant vector search
- [x] Directus CMS
- [x] Три режима обработки
- [x] GPU support

### Версия 2.1
- [ ] Telegram scraper
- [ ] Reddit scraper
- [ ] Medium scraper
- [ ] WebSocket для real-time updates
- [ ] GraphQL API
- [ ] Улучшенная дедупликация (semantic similarity)

### Версия 2.2
- [ ] React Admin Panel
- [ ] User authentication (JWT)
- [ ] Role-based access control (RBAC)
- [ ] API rate limiting
- [ ] Webhook notifications

### Версия 3.0
- [ ] Kubernetes manifests
- [ ] Helm charts
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Automated testing in CI
- [ ] Multi-language support
- [ ] Elasticsearch integration

### Будущее
- [ ] Mobile app (React Native)
- [ ] Browser extension
- [ ] Email digests
- [ ] RSS feed generation
- [ ] Social media integration
- [ ] Machine learning for personalization

---

## Вклад в Проект

Приветствуются любые contributions! Пожалуйста, следуйте guidelines:

### Процесс

1. Fork репозитория
2. Создать feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit изменений
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. Push в branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. Открыть Pull Request

### Code Style

- Следовать PEP 8
- Использовать type hints
- Писать docstrings для всех public функций
- Покрывать тестами новый функционал
- Использовать meaningful commit messages

### Тестирование

```bash
# Запустить все тесты
pytest

# С coverage
pytest --cov=src --cov-report=html

# Только unit тесты
pytest tests/unit/
```

```bash
# Быстрый старт
docker-compose up -d
docker-compose exec api python run_full_pipeline.py 10

# Проверить результаты
curl http://localhost:8000/api/v1/articles/?limit=10
```
