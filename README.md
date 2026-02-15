# News Aggregator Pro

**Профессиональная система агрегации, AI-обработки и публикации технических новостей**

Hexagonal Architecture · FastAPI · 7 AI-агентов · Multi-Provider LLM · Ollama / Groq / Google / OpenRouter

---

## Содержание

1. [Обзор](#обзор)
2. [Архитектура](#архитектура)
3. [Быстрый старт (Docker)](#быстрый-старт-docker)
4. [Конфигурация LLM](#конфигурация-llm)
5. [Per-Agent модели (Ollama)](#per-agent-модели-ollama)
6. [AI-агенты и пайплайн](#ai-агенты-и-пайплайн)
7. [StyleNormalizer: Chunking](#stylenormalizer-chunking)
8. [Запуск Pipeline](#запуск-pipeline)
9. [API](#api)
10. [Telegram и Telegraph](#telegram-и-telegraph)
11. [Оптимизация для Ollama](#оптимизация-для-ollama)
12. [Структура проекта](#структура-проекта)
13. [Troubleshooting](#troubleshooting)

---

## Обзор

**News Aggregator Pro** собирает статьи с Habr, пропускает через 7-этапный AI-конвейер (классификация → релевантность → тизер → заголовок → рерайт → валидация → Telegram-форматирование), сохраняет в PostgreSQL + Qdrant и публикует в Telegram-канал через Telegraph.

### Ключевые возможности

- **7 AI-агентов** — полный цикл обработки статьи от парсинга до публикации
- **Multi-Provider LLM** — Ollama (локально), Groq, Google Gemini, OpenRouter с автоматическим fallback
- **Per-Agent модели** — разные модели для разных задач (быстрая для классификации, мощная для рерайта)
- **Chunking** — обработка статей до 100K символов через разбиение на части
- **Docker-first** — полный стек в `docker compose up`

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Compose                        │
├──────────┬──────────┬─────────┬────────┬────────┬───────────┤
│  FastAPI │  Ollama  │ Postgres│ Qdrant │ Redis  │ Directus  │
│  (API)   │  (LLM)   │  (DB)   │(Vector)│(Cache) │  (CMS)    │
├──────────┴──────────┴─────────┴────────┴────────┴───────────┤
│                      n8n (Automation)                        │
└─────────────────────────────────────────────────────────────┘
```

### Пайплайн обработки

```
Habr → Scraper → [Classifier → Relevance → Summarizer → Rewriter
                   → StyleNormalizer → Validator → TelegramFormatter]
               → PostgreSQL + Qdrant → Telegraph → Telegram
```

---

## Быстрый старт (Docker)

### Предварительные требования

- Docker и Docker Compose
- NVIDIA GPU (опционально, для Ollama) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Минимум 16 GB RAM (с Ollama), 8 GB без

### 1. Клонирование и настройка

```bash
git clone https://github.com/your-repo/news-aggregator-pro.git
cd news-aggregator-pro
cp .env.example .env
```

### 2. Настройка `.env`

```env
# LLM Provider
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:14b-instruct-q5_k_m
ENABLE_FALLBACK=false

# Для облачных провайдеров (опционально)
GROQ_API_KEY=gsk_xxx
GOOGLE_API_KEY=AIzaSy_xxx
OPENROUTER_API_KEY=sk-or-v1-xxx

# Telegram (опционально)
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHANNEL_ID=@your_channel
TELEGRAPH_ACCESS_TOKEN=xxx

# Database
POSTGRES_PASSWORD=your_secure_password
```

### 3. Запуск

```bash
docker compose up -d
docker compose ps          # Проверить статус
docker compose logs api -f # Логи API
```

### 4. Первый запуск pipeline

```bash
# Через API
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "provider": "ollama", "no_fallback": true}'

# Через CLI
docker compose exec api python run_full_pipeline.py 1 --provider ollama --no-fallback
```

### 5. `.dockerignore` (важно!)

Если в проекте есть папка `ollama_models/`, добавьте `.dockerignore` чтобы Docker не копировал модели (десятки GB) внутрь контейнера:

```
ollama_models/
.git/
__pycache__/
.venv/
logs/
cache/
*.log
```

---

## Конфигурация LLM

### Приоритет конфигурации

```
CLI аргументы  →  ENV переменные  →  config/models.yaml  →  hardcoded defaults
  (высший)           (средний)           (основной)             (низший)
```

### `config/models.yaml` — главный конфиг

```yaml
defaults:
  provider: ollama
  strategy: balanced
  enable_fallback: true

ollama:
  model: qwen2.5:14b-instruct-q5_k_m   # Дефолтная модель
  base_url: http://ollama:11434
  context_length: 32768

# Per-agent модели (опционально, см. раздел ниже)
agent_models:
  rewriter: glm-4.7-flash:q4_K_M
  style_normalizer: glm-4.7-flash:q4_K_M
  telegram_formatter: glm-4.7-flash:q4_K_M

temperatures:
  classifier: 0.1
  relevance: 0.3
  quality_validator: 0.1
  summarizer: 0.5
  rewriter: 0.7
  style_normalizer: 0.3
  telegram_formatter: 0.5
```

### Провайдеры

| Провайдер | Лимит (бесплатно) | Скорость | Качество |
|-----------|-------------------|----------|----------|
| **Ollama** | Без лимитов (локально) | Зависит от GPU | ⭐⭐⭐⭐ |
| **Groq** | 30 req/min, 14K/day | Самый быстрый | ⭐⭐⭐⭐ |
| **Google Gemini** | 60 req/min, 1.5K/day | Быстро | ⭐⭐⭐⭐⭐ |
| **OpenRouter** | 50 req/day (free tier) | Средне | ⭐⭐⭐⭐ |

### Smart Fallback

```
Primary Provider → Fallback #1 → Fallback #2 → Fallback #3
     Groq       →   OpenRouter  →    Google   →    Ollama
```

При ошибке или rate limit (429) система автоматически переключается на следующего провайдера. Отключается через `--no-fallback` или `ENABLE_FALLBACK=false`.

---

## Per-Agent модели (Ollama)

Разные задачи требуют разных моделей. Классификация — быстрая и лёгкая задача, а рерайт текста — тяжёлая. Система поддерживает назначение отдельной модели каждому агенту.

### Как работает

Приоритет выбора модели для каждого агента:

```
agent_models.<agent_name>  →  env OLLAMA_MODEL  →  ollama.model  →  default
     (из YAML)                  (из .env)           (из YAML)       (hardcoded)
```

### Конфигурация в `models.yaml`

```yaml
ollama:
  model: qwen2.5:14b-instruct-q5_k_m   # Дефолт для лёгких задач

agent_models:
  # Лёгкие задачи → используют ollama.model (qwen)
  # classifier:        (не указан — берёт дефолт)
  # relevance:         (не указан — берёт дефолт)
  # quality_validator: (не указан — берёт дефолт)
  # summarizer:        (не указан — берёт дефолт)

  # Генерация текста → тяжёлая модель
  rewriter: glm-4.7-flash:q4_K_M
  style_normalizer: glm-4.7-flash:q4_K_M
  telegram_formatter: glm-4.7-flash:q4_K_M
```

### Результат

| Агент | Модель | Задача |
|-------|--------|--------|
| ClassifierAgent | qwen2.5:14b | JSON-классификация |
| RelevanceAgent | qwen2.5:14b | Числовой score |
| SummarizerAgent | qwen2.5:14b | Короткий тизер |
| QualityValidatorAgent | qwen2.5:14b | Да/нет валидация |
| **RewriterAgent** | **glm-4.7-flash** | **Переписание заголовка** |
| **StyleNormalizerAgent** | **glm-4.7-flash** | **Полный рерайт статьи** |
| **TelegramFormatterAgent** | **glm-4.7-flash** | **Форматирование текста** |

### Переключение моделей и GPU

Ollama автоматически выгружает/загружает модели при переключении. На одной GPU одновременно живёт одна модель, переключение занимает ~5–10 секунд. Модели кэшируются в VRAM на 5 минут после последнего запроса (`OLLAMA_KEEP_ALIVE`).

Если обе модели не помещаются в VRAM одновременно — закомментируйте `agent_models` и используйте одну модель для всего:

```yaml
# agent_models:
#   rewriter: glm-4.7-flash:q4_K_M
#   style_normalizer: glm-4.7-flash:q4_K_M
#   telegram_formatter: glm-4.7-flash:q4_K_M
```

---

## AI-агенты и пайплайн

### Схема обработки

```
ВХОДНАЯ СТАТЬЯ (заголовок + контент + метаданные)
         │
         ▼
┌─ ШАГ 1: ClassifierAgent ──────────────────────────────────────┐
│  Тип: НОВОСТЬ или СТАТЬЯ  │  Модель: light  │  ~2-10 сек     │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ШАГ 2: RelevanceAgent ───────────────────────────────────────┐
│  Оценка: 0-10 для IT-аудитории  │  Модель: light             │
└────────────────────────────────────────────────────────────────┘
         │
         ▼  (score < min_relevance → SKIP)
         │
┌─ ШАГ 3: SummarizerAgent ──────────────────────────────────────┐
│  Тизер: 2-4 предложения  │  Модель: medium                    │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ШАГ 4: RewriterAgent ────────────────────────────────────────┐
│  Заголовок: информативный, 40-80 символов  │  Модель: medium  │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ШАГ 5: StyleNormalizerAgent ─────────────────────────────────┐
│  Рерайт: нормализация стиля + chunking  │  Модель: heavy      │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ШАГ 6: QualityValidatorAgent ────────────────────────────────┐
│  Валидация: длина, соотношение, артефакты  │  Модель: light    │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ШАГ 7: TelegramFormatterAgent ───────────────────────────────┐
│  Форматирование: HTML для Telegram  │  Модель: medium         │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
ОБРАБОТАННАЯ СТАТЬЯ → PostgreSQL + Qdrant → Telegraph → Telegram
```

### Типы задач (TaskType)

| Тип | Описание | Примеры |
|-----|----------|---------|
| `LIGHT` | Простые, быстрые | Classifier, Relevance, Validator |
| `MEDIUM` | Средние | Summarizer, Rewriter, Formatter |
| `HEAVY` | Тяжёлые, длинная генерация | StyleNormalizer |

---

## StyleNormalizer: Chunking

### Проблема

Локальные модели ограничены контекстным окном (~32K токенов). Длинные статьи (10K+ символов) могут вызвать timeout или обрезку.

### Решение

StyleNormalizerAgent разбивает текст на чанки по абзацам, обрабатывает каждый отдельно и склеивает результат:

```
Статья 24873 символов
         │
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
 Chunk1 Chunk2 Chunk3 Chunk4
 10282  5388   5299   3862
    │    │      │      │
    ▼    ▼      ▼      ▼
  5712  4318   4361   3337    (каждый обработан отдельно)
    │    │      │      │
    └────┼──────┼──────┘
         ▼
   Результат: 17734 символов (71% от оригинала)
```

### Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `MAX_CONTENT_LENGTH` | 8000 | Порог для включения chunking |
| `MAX_CHUNK_SIZE` | 6000 | Максимум символов в одном чанке |
| `MIN_RESPONSE_RATIO` | 0.3 | Минимум 30% от входа (иначе fallback) |
| `max_tokens` | `len(chunk) // 2 + 1000` | Адаптивно по размеру чанка |

### Fallback

Если LLM возвращает слишком короткий ответ, применяется базовая очистка без LLM: удаление приветствий ("Привет, Хабр!"), прощаний ("Спасибо за внимание"), конвертация первого лица в безличную форму.

---

## Запуск Pipeline

### Через HTTP API (рекомендуется для автоматизации)

```bash
# Запуск
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "provider": "ollama",
    "no_fallback": true,
    "publish_telegraph": true,
    "publish_telegram": true,
    "min_publish_score": 7
  }'

# Проверка статуса
curl http://localhost:8000/api/v1/pipeline/status

# Остановка (сброс блокировки)
curl -X POST http://localhost:8000/api/v1/pipeline/stop
```

### Через CLI

```bash
# Ollama, 10 статей, с публикацией
docker compose exec api python run_full_pipeline.py 10 \
  --provider ollama --no-fallback --publish

# Groq с fallback
docker compose exec api python run_full_pipeline.py 10 --provider groq

# Google, только Telegraph
docker compose exec api python run_full_pipeline.py 10 \
  --provider google --telegraph --min-publish-score 8

# Через ENV
docker compose exec -e LLM_PROVIDER=ollama -e ENABLE_FALLBACK=false \
  api python run_full_pipeline.py 5
```

### Через n8n (автоматизация по расписанию)

HTTP Request Node → `POST http://api:8000/api/v1/pipeline/run`

```json
{
  "limit": 10,
  "publish_telegraph": true,
  "publish_telegram": true
}
```

### Параметры CLI

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| `--provider` | LLM провайдер (ollama/groq/google/openrouter) | из .env |
| `--no-fallback` | Отключить fallback | false |
| `--strategy` | Стратегия (balanced/cost_optimized/quality_focused) | balanced |
| `--publish` | Telegraph + Telegram | false |
| `--telegraph` | Только Telegraph | false |
| `--telegram` | Только Telegram | false |
| `--min-publish-score` | Мин. score для публикации | 7 |
| `--min-relevance` | Мин. score для Qdrant | 5 |
| `--max-retries` | Макс. повторов | 3 |
| `--verbose` | Подробный вывод | false |

---

## API

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Основные эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/pipeline/run` | Запустить pipeline |
| `GET` | `/api/v1/pipeline/status` | Статус pipeline |
| `POST` | `/api/v1/pipeline/stop` | Остановить pipeline |

---

## Telegram и Telegraph

### Процесс публикации

1. **Pipeline** обрабатывает статью (score ≥ `min_publish_score`)
2. **TelegraphPublisher** создаёт страницу на Telegraph с полным текстом + изображениями
3. **TelegramPublisher** отправляет пост в канал: заголовок + тизер + ссылка на Telegraph

### Настройка

```env
# .env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHANNEL_ID=@your_channel
TELEGRAPH_ACCESS_TOKEN=your_token
```

Получение Telegraph токена:

```bash
curl https://api.telegra.ph/createAccount?short_name=NewsBot&author_name=News
```

---

## Оптимизация для Ollama

### Рекомендуемые модели

| Модель | VRAM | Качество | Скорость |
|--------|------|----------|----------|
| `qwen2.5:7b-instruct` | 6 GB | ⭐⭐⭐ | Быстро |
| `qwen2.5:14b-instruct-q5_k_m` | 12 GB | ⭐⭐⭐⭐ | Средне |
| `qwen2.5:32b-instruct-q4_k_m` | 24 GB | ⭐⭐⭐⭐⭐ | Медленно |
| `glm-4.7-flash:q4_K_M` | 18 GB | ⭐⭐⭐⭐⭐ (генерация) | Медленно |

### GPU-специфичные рекомендации

| GPU | VRAM | Рекомендуемая модель |
|-----|------|---------------------|
| GTX 1080 Ti | 11 GB | `qwen2.5:14b-instruct-q5_k_m` (частично на CPU) |
| RTX 3060 | 12 GB | `qwen2.5:14b-instruct-q5_k_m` |
| RTX 3090/4090 | 24 GB | `qwen2.5:32b` или per-agent (qwen + glm) |

### Docker Compose для GPU

```yaml
# docker-compose.yml (фрагмент)
ollama:
  image: ollama/ollama:0.9.6
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  volumes:
    - ollama_models:/root/.ollama
```

Модели монтируются через Docker volume — не нужно копировать их внутрь контейнера.

---

## Структура проекта

```
news-aggregator-pro/
├── config/
│   └── models.yaml                    # Конфигурация LLM и агентов
├── src/
│   ├── api/routes/
│   │   └── pipeline.py                # HTTP API для pipeline
│   ├── application/ai_services/
│   │   ├── agents/
│   │   │   ├── base_agent.py          # Базовый класс агентов
│   │   │   ├── classifier_agent.py    # Классификация
│   │   │   ├── relevance_agent.py     # Релевантность
│   │   │   ├── summarizer_agent.py    # Тизеры
│   │   │   ├── rewriter_agent.py      # Заголовки
│   │   │   ├── style_normalizer_agent.py  # Рерайт + chunking
│   │   │   ├── quality_validator_agent.py # Валидация
│   │   │   └── telegram_formatter_agent.py
│   │   └── orchestrator.py            # Оркестратор пайплайна
│   ├── config/
│   │   └── models_config.py           # ModelsConfig + per-agent routing
│   ├── infrastructure/
│   │   ├── ai/
│   │   │   ├── llm_provider.py        # Multi-provider + factory
│   │   │   ├── ollama_client.py       # Ollama HTTP client
│   │   │   └── qdrant_client.py       # Vector search
│   │   ├── telegram/
│   │   │   ├── telegraph_publisher.py
│   │   │   └── telegram_publisher.py
│   │   └── skiplist/                   # Пропуск проблемных URL
│   ├── domain/entities/
│   │   └── article.py                 # Доменная модель статьи
│   └── scrapers/habr/
│       └── scraper_service.py         # Парсер Habr
├── run_full_pipeline.py               # CLI для полного конвейера
├── docker-compose.yml
├── .env.example
├── .dockerignore
└── README.md
```

---

## Troubleshooting

### Pipeline timeout (10 мин)

Модель слишком тяжёлая для GPU. StyleNormalizer на длинных статьях может не укладываться в timeout.

```bash
# Решение 1: Использовать более лёгкую модель
# .env: OLLAMA_MODEL=qwen2.5:7b-instruct

# Решение 2: Закомментировать agent_models (одна модель для всего)
# config/models.yaml: # agent_models: ...
```

### Агент возвращает пустой ответ (0 chars)

Некоторые модели (например, GLM) чувствительны к формату промпта и могут не генерировать ответ. Используйте qwen2.5 — она стабильна на всех агентах.

### Все провайдеры возвращают 429

Rate limit. Подождите 1–2 минуты или переключитесь на Ollama:

```bash
docker compose exec -e LLM_PROVIDER=ollama api python run_full_pipeline.py 5
```

### Docker build копирует 30 GB

Нет `.dockerignore`. Добавьте файл с исключением `ollama_models/` — модели монтируются через volume, копировать не нужно.

### Модель не найдена в Ollama

```bash
# Скачать модель
docker compose exec ollama ollama pull qwen2.5:14b-instruct-q5_k_m

# Проверить доступные
docker compose exec ollama ollama list
```

### Логи

```bash
# Все логи API
docker compose logs api -f

# Фильтр по LLM
docker compose logs api -f | grep -E "(model=|Ollama|Chunk|StyleNormalizer)"

# Только ошибки
docker compose logs api -f | grep -E "(ERROR|WARNING)"
```

---

## API ключи (бесплатные)

Для облачных провайдеров нужен минимум один ключ:

| Провайдер | Регистрация | Формат ключа |
|-----------|-------------|--------------|
| **Groq** | [console.groq.com](https://console.groq.com) | `gsk_...` |
| **Google AI** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `AIzaSy...` |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai/keys) | `sk-or-v1-...` |

Для Ollama ключи не нужны — модели работают локально.

---

## Roadmap

- [x] Multi-provider LLM с Smart Fallback
- [x] 7-агентный AI пайплайн с chunking
- [x] Per-agent модели (models.yaml → agent_models)
- [x] Telegram + Telegraph публикация
- [x] Skiplist для проблемных URL
- [x] Docker-first деплой
- [ ] Web UI для управления (Streamlit)
- [ ] Prometheus метрики
- [ ] Параллельная обработка чанков
- [ ] RAG (семантический поиск по статьям)
- [ ] Мультиязычная поддержка