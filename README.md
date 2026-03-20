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
9. [Обработка существующих статей](#обработка-существующих-статей)
10. [Supabase (облачная БД)](#supabase-облачная-бд)
11. [GitHub Actions (автоматизация)](#github-actions-автоматизация)
12. [Публикация статей](#публикация-статей)
13. [RSS Feed](#rss-feed)
14. [Telegram Tool (утилита)](#telegram-tool-утилита)
15. [API](#api)
16. [Telegram и Telegraph](#telegram-и-telegraph)
17. [Оптимизация для Ollama](#оптимизация-для-ollama)
18. [Streamlit Dashboard (Frontend)](#streamlit-dashboard-frontend)
19. [Порты сервисов](#порты-сервисов)
20. [Структура проекта](#структура-проекта)
21. [Troubleshooting](#troubleshooting)

---

## Обзор

**News Aggregator Pro** собирает статьи с Habr, пропускает через 7-этапный AI-конвейер (классификация → релевантность → тизер → заголовок → рерайт → валидация → Telegram-форматирование), сохраняет в PostgreSQL + Qdrant и публикует в Telegram-канал через Telegraph.

### Ключевые возможности

- **7 AI-агентов** — полный цикл обработки статьи от парсинга до публикации
- **Multi-Provider LLM** — Ollama (локально), Groq, Google Gemini, OpenRouter с автоматическим fallback
- **Per-Agent модели** — разные модели для разных задач (быстрая для классификации, мощная для рерайта)
- **Chunking** — обработка статей до 100K символов через разбиение на части
- **Docker-first** — полный стек в `docker compose up`
- **Supabase** — облачная PostgreSQL для хранения статей (бесплатный tier)
- **GitHub Actions** — автоматический парсинг + публикация + RSS
- **Telegraph + Telegram** — форматированные статьи с изображениями

### Полный цикл (автоматический)

```
GitHub Actions (каждые 2 дня)    →  Habr → Supabase (status=pending)
Ваш ПК (вручную, когда хотите)  →  AI обработка → Supabase (status=processed)
GitHub Actions (ежедневно)       →  Telegraph + Telegram + RSS (status=published)
```

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
         │                                        │
         ▼                                        ▼
┌─────────────────┐                    ┌─────────────────────┐
│ Supabase (Cloud)│                    │  GitHub Actions     │
│ PostgreSQL      │◄───────────────────│  Scrape + Publish   │
│ (статьи)        │                    │  + RSS              │
└─────────────────┘                    └─────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Telegraph (статьи) + Telegram (посты) + RSS (feed.xml)  │
└─────────────────────────────────────┘
```

### Пайплайн обработки

```
Habr → Scraper → [Classifier → Relevance → Summarizer → Rewriter
                   → StyleNormalizer → Validator → TelegramFormatter]
               → PostgreSQL/Supabase + Qdrant → Telegraph → Telegram → RSS
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

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=-100xxxxxxxxxx

# Database (локальная)
DATABASE_URL=postgresql://newsaggregator:changeme123@postgres:5432/news_aggregator
POSTGRES_PASSWORD=your_secure_password

# Supabase (облачная, опционально)
# SUPABASE_DATABASE_URL=postgresql://postgres.xxx:PASS@aws-1-eu-west-1.pooler.supabase.com:6543/postgres
```

### 3. Запуск

```bash
docker compose up -d
docker compose ps          # Проверить статус
docker compose logs api -f # Логи API
```

### 4. Первый запуск pipeline

```bash
# Через CLI — одна статья
docker compose exec api python run_full_pipeline.py --url https://habr.com/ru/news/1004288/ -p ollama --publish --verbose

# Через API
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "provider": "ollama", "no_fallback": true}'
```

### 5. `.dockerignore` (важно!)

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
  model: qwen2.5:14b-instruct-q5_k_m
  base_url: http://ollama:11434
  context_length: 32768

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

---

## Per-Agent модели (Ollama)

| Агент | Модель | Задача |
|-------|--------|--------|
| ClassifierAgent | qwen2.5:14b | JSON-классификация |
| RelevanceAgent | qwen2.5:14b | Числовой score |
| SummarizerAgent | qwen2.5:14b | Короткий тизер |
| QualityValidatorAgent | qwen2.5:14b | Да/нет валидация |
| **RewriterAgent** | **glm-4.7-flash** | **Переписание заголовка** |
| **StyleNormalizerAgent** | **glm-4.7-flash** | **Полный рерайт статьи** |
| **TelegramFormatterAgent** | **glm-4.7-flash** | **Форматирование текста** |

---

## AI-агенты и пайплайн

```
ВХОДНАЯ СТАТЬЯ (заголовок + контент + метаданные)
         │
    ШАГ 1: ClassifierAgent      → НОВОСТЬ или СТАТЬЯ
    ШАГ 2: RelevanceAgent        → Оценка 0-10 (< min → SKIP)
    ШАГ 3: SummarizerAgent       → Тизер 2-4 предложения
    ШАГ 4: RewriterAgent         → Улучшенный заголовок
    ШАГ 5: StyleNormalizerAgent  → Рерайт в безличный стиль + chunking
    ШАГ 6: QualityValidatorAgent → Проверка качества
    ШАГ 7: TelegramFormatterAgent → HTML пост для Telegram
         │
         ▼
ОБРАБОТАННАЯ СТАТЬЯ → DB (status=processed)
```

---

## StyleNormalizer: Chunking

Длинные статьи разбиваются на чанки по абзацам и обрабатываются отдельно:

| Параметр | Значение | Описание |
|----------|----------|----------|
| `MAX_CONTENT_LENGTH` | 8000 | Порог для включения chunking |
| `MAX_CHUNK_SIZE` | 6000 | Максимум символов в одном чанке |
| `MIN_RESPONSE_RATIO` | 0.3 | Минимум 30% от входа (иначе fallback) |

Fallback: удаление приветствий, прощаний, конвертация в безличную форму — без LLM.

---

## Запуск Pipeline

### Полный конвейер (парсинг + AI + публикация)

```bash
# Одна статья по URL
docker compose exec api python run_full_pipeline.py \
  --url https://habr.com/ru/news/1004288/ -p ollama --publish --verbose

# С публикацией при любом score
docker compose exec api python run_full_pipeline.py \
  --url https://habr.com/ru/articles/123456/ -p ollama --publish --min-relevance 1 --verbose

# Через Supabase (из Docker)
docker compose exec -e DATABASE_URL="postgresql://..." api python run_full_pipeline.py \
  --url https://habr.com/ru/news/1004288/ -p ollama --publish --verbose
```

### Флаги run_full_pipeline.py

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| `--url URL` | Конкретная статья по ссылке | — |
| `--feed N` | Парсинг N статей из ленты Habr | — |
| `--provider`, `-p` | LLM провайдер (ollama/groq/google/openrouter) | из .env |
| `--publish` | Публикация в Telegraph + Telegram | false |
| `--telegraph` | Только Telegraph | false |
| `--telegram` | Только Telegram | false |
| `--min-relevance N` | Мин. score для AI обработки | 5 |
| `--min-publish-score N` | Мин. score для публикации | 7 |
| `--no-fallback` | Отключить fallback | false |
| `--verbose` | Подробный вывод | false |
| `--hubs` | Фильтр по хабам (для --feed) | все |

---

## Обработка существующих статей

`process_existing_articles.py` — AI обработка статей, уже находящихся в базе данных.
Используется для обработки статей из Supabase (спарсенных GitHub Actions).

### Использование

```bash
# Через Docker (рекомендуется — Ollama доступен)
docker compose exec -e DATABASE_URL="postgresql://postgres.xxx:PASS@aws-1-eu-west-1.pooler.supabase.com:6543/postgres" \
  api python process_existing_articles.py -p ollama --limit 5

# С явным --db
python process_existing_articles.py \
  --db "postgresql://postgres.xxx:PASS@aws-1-eu-west-1.pooler.supabase.com:6543/postgres" \
  -p ollama --limit 3

# Конкретная статья по URL
python process_existing_articles.py --db "..." -p ollama \
  --url https://habr.com/ru/articles/1006098/

# Конкретная статья по ID
python process_existing_articles.py --db "..." -p ollama \
  --id 550e8400-e29b-41d4-a716-446655440000

# Несколько статей
python process_existing_articles.py --db "..." -p ollama \
  --url https://habr.com/ru/articles/111/,https://habr.com/ru/articles/222/

# Переобработать все за неделю
python process_existing_articles.py --db "..." -p ollama \
  --days 7 --reprocess-all --limit 10

# Локальная БД (без --db, берёт из .env)
docker compose exec api python process_existing_articles.py -p ollama --limit 5
```

### Флаги process_existing_articles.py

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| `--db URL` | DATABASE_URL (Supabase pooler) | из .env |
| `--url`, `-u` | URL статьи (можно через запятую) | — |
| `--id` | UUID статьи (можно через запятую) | — |
| `--limit N` | Макс. количество статей | без лимита |
| `--days N` | Только за последние N дней | всё время |
| `--reprocess-all` | Переобработать все (включая обработанные) | false |
| `--min-relevance N` | Мин. score для Qdrant | 5 |
| `--provider`, `-p` | LLM провайдер | из config |
| `--strategy`, `-s` | Стратегия (balanced/cost_optimized/quality_focused) | balanced |
| `--no-fallback` | Отключить fallback | false |
| `--verbose`, `-v` | Подробный вывод | false |
| `--debug` | Debug режим | false |

### Что делает

1. Загружает из БД статьи с `relevance_score IS NULL` (или по фильтру)
2. Прогоняет через AIOrchestrator (7 агентов)
3. Записывает результат обратно: `editorial_title`, `editorial_rewritten`, `telegram_post_text`, `relevance_score`, `status=processed`

---

## Supabase (облачная БД)

Supabase используется как облачная PostgreSQL для хранения статей. GitHub Actions пишет туда, ваш ПК читает и обрабатывает.

### Настройка

1. Создать проект на [supabase.com](https://supabase.com)
2. Project Settings → Database → Connection string (Session mode, IPv4/pooler)
3. Применить миграцию: `supabase_migration.sql`
4. Добавить URL в GitHub Secrets: `DATABASE_URL`

### Pooler URL (обязательно IPv4)

```
postgresql://postgres.XXXXX:PASSWORD@aws-1-eu-west-1.pooler.supabase.com:6543/postgres
```

### Free tier ограничения

- 500 MB хранилище (~17K-30K статей)
- Пауза после 7 дней неактивности (cron каждые 2 дня будит)
- 5 GB bandwidth/month

### Таблица articles (основные поля)

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | uuid | Первичный ключ |
| `title` | text | Оригинальный заголовок |
| `content` | text | Полный текст |
| `url` | text | URL (уникальный) |
| `status` | text | pending → processed → published |
| `relevance_score` | float | AI скор 0-10 |
| `editorial_title` | text | AI заголовок |
| `editorial_teaser` | text | AI тизер |
| `editorial_rewritten` | text | AI полный рерайт |
| `telegram_post_text` | text | Готовый пост для Telegram (HTML) |
| `telegraph_url` | text | URL страницы на Telegraph |
| `tags` | text[] | Теги |
| `images` | text[] | URL изображений |

---

## GitHub Actions (автоматизация)

Два workflow для полной автоматизации.

### Scrape & Cleanup (каждые 2 дня)

Файл: `.github/workflows/scrape.yml`

```
Habr feed → parse 50 articles → save to Supabase (status=pending)
         → cleanup old articles → health check (keep Supabase alive)
```

Ручной запуск: Actions → "Scrape & Cleanup" → Run workflow (limit, hubs)

### Publish & RSS (ежедневно, 12:00 MSK)

Файл: `.github/workflows/publish.yml`

```
Supabase (status=processed, score≥7) → Telegraph → Telegram → status=published
                                     → generate RSS feed → commit to GitHub Pages
```

Ручной запуск: Actions → "Publish & RSS" → Run workflow (min_score, limit, dry_run)

### GitHub Secrets (обязательно)

| Secret | Описание |
|--------|----------|
| `DATABASE_URL` | Supabase pooler URL |
| `TELEGRAM_BOT_TOKEN` | Токен бота от BotFather |
| `TELEGRAM_CHAT_ID` | ID канала (-100xxxxxxxxxx) |

### GitHub Pages (для RSS)

Settings → Pages → Source: "Deploy from a branch" → Branch: `main`, Folder: `/docs`

RSS доступен по: `https://username.github.io/news-aggregator-pro/feed.xml`

---

## Публикация статей

`publish_pending.py` — берёт обработанные статьи из Supabase и публикует.

### Использование

```bash
# Опубликовать все processed (score >= 7)
python publish_pending.py

# Явно указать БД
python publish_pending.py --db "postgresql://..."

# Другой порог
python publish_pending.py --min-score 5

# Только Telegraph (без Telegram)
python publish_pending.py --no-telegram

# Лимит публикаций
python publish_pending.py --limit 5

# Dry run (показать без публикации)
python publish_pending.py --dry-run
```

### Флаги publish_pending.py

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| `--min-score N` | Мин. score для публикации | 7 |
| `--limit N` | Макс. количество | 20 |
| `--no-telegram` | Пропустить Telegram | false |
| `--dry-run` | Показать что будет опубликовано | false |
| `--delay N` | Задержка между публикациями (сек) | 3.0 |
| `--verbose` | Подробный вывод | false |

### Что делает для каждой статьи

1. Создаёт Telegraph страницу (JSON Node API, с изображениями между абзацами)
2. Отправляет пост в Telegram канал (полный `telegram_post_text` с "Читать полностью →")
3. Обновляет статус в Supabase → `status=published`, `telegraph_url=...`

---

## RSS Feed

`generate_rss.py` — генерирует RSS ленту из опубликованных статей.

### Использование

```bash
python generate_rss.py                        # docs/feed.xml, 50 статей
python generate_rss.py --output feed.xml      # другой путь
python generate_rss.py --limit 100            # больше статей
```

RSS автоматически генерируется в publish workflow и коммитится в `docs/feed.xml`.
GitHub Pages хостит файл по адресу: `https://username.github.io/news-aggregator-pro/feed.xml`

Подписка через любой RSS-ридер (Feedly, Inoreader, etc.).

---

## Telegram Tool (утилита)

`telegram_tool.py` — утилита для работы с Telegram ботом и БД.

### Команды

```bash
# Проверить подключение бота, канала, БД
python telegram_tool.py status

# Отправить тестовое сообщение в канал
python telegram_tool.py test

# Отправить форматированный пост
python telegram_tool.py post --title "AI новости" --body "OpenAI выпустил GPT-5" \
  --link "https://example.com" --tags "ai,openai"

# Показать статьи из БД
python telegram_tool.py list                          # processed
python telegram_tool.py list --status pending --limit 20

# Отправить конкретную статью из БД
python telegram_tool.py send --id 0929dd03

# Отправить все processed статьи
python telegram_tool.py send-all --min-score 5 --limit 3

# С явным указанием БД
python telegram_tool.py --db "postgresql://..." status
python telegram_tool.py --db "postgresql://..." list --status pending
```

Скрипт автоматически читает `.env` для токенов и DATABASE_URL.

---

## API

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/pipeline/run` | Запустить pipeline |
| `GET` | `/api/v1/pipeline/status` | Статус pipeline |
| `POST` | `/api/v1/pipeline/stop` | Остановить pipeline |

---

## Telegram и Telegraph

### Telegraph v3.0

- JSON Node API (вместо html_content)
- Зависимость: `requests` (не `telegraph`)
- Изображения распределяются между абзацами
- Inline форматирование: `**bold**` → `<strong>`, `*italic*` → `<em>`, `` `code` `` → `<code>`
- Подзаголовки: короткие строки → `<h4>`
- Без ссылки на оригинал (трафик в Telegram канал)

### Telegram Publisher v2.1

- Подробное логирование (token/chat_id при инициализации, HTTP статус при отправке)
- Отправка полного `telegram_post_text` (от TelegramFormatterAgent) с подстановкой Telegraph URL
- Fallback: title + teaser + link если `telegram_post_text` пуст
- Без звука (`disable_notification=True`)

### Настройка бота

1. BotFather → `/newbot` → получить токен
2. Создать канал → добавить бота как админа (Post Messages)
3. Узнать chat_id: отправить сообщение в канал → `https://api.telegram.org/botTOKEN/getUpdates`
4. Добавить в `.env`:

```env
TELEGRAM_BOT_TOKEN=8688656163:AAE95sOLqX33ZUwdQ-...
TELEGRAM_CHAT_ID=-1003875931100
```

5. Добавить в `docker-compose.yml` (секция `api` → `environment`):

```yaml
TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}
TELEGRAM_CHAT_ID: ${TELEGRAM_CHAT_ID}
TELEGRAM_CHANNEL_URL: ${TELEGRAM_CHANNEL_URL}
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
| GTX 1080 Ti | 11 GB | `qwen2.5:14b-instruct-q5_k_m` |
| RTX 3060 | 12 GB | `qwen2.5:14b-instruct-q5_k_m` |
| RTX 3090/4090 | 24 GB | `qwen2.5:32b` или per-agent |

---

## Streamlit Dashboard (Frontend)

**URL:** [http://localhost:8501](http://localhost:8501)

| Страница | Описание |
|----------|----------|
| **📊 Дашборд** | Статистика, графики |
| **📰 Статьи** | Таблица с фильтрами |
| **✏️ Редактор** | 5 вкладок для статьи |
| **⚙️ Настройки** | Провайдер, модели, температуры |

---

## Порты сервисов

| Сервис | Порт | URL |
|--------|------|-----|
| **Streamlit** | 8501 | http://localhost:8501 |
| **FastAPI** | 8000 | http://localhost:8000/docs |
| **Directus** | 8055 | http://localhost:8055 |
| **Ollama** | 11434 | http://localhost:11434 |
| **Qdrant** | 6333 | http://localhost:6333/dashboard |
| **PostgreSQL** | 5432 | postgresql://localhost:5432 |
| **Redis** | 6379 | redis://localhost:6379 |
| **n8n** | 5678 | http://localhost:5678 |

---

## Структура проекта

```
news-aggregator-pro/
├── .github/
│   └── workflows/
│       ├── scrape.yml                 # Автопарсинг каждые 2 дня
│       └── publish.yml                # Публикация + RSS ежедневно
├── config/
│   ├── models.yaml                    # Конфигурация LLM и агентов
│   └── settings.json                  # Настройки из UI (авто)
├── docs/
│   └── feed.xml                       # RSS лента (авто-генерируется)
├── src/
│   ├── api/routes/
│   │   └── pipeline.py                # HTTP API
│   ├── application/ai_services/
│   │   ├── agents/
│   │   │   ├── base_agent.py
│   │   │   ├── classifier_agent.py
│   │   │   ├── relevance_agent.py
│   │   │   ├── summarizer_agent.py
│   │   │   ├── rewriter_agent.py
│   │   │   ├── style_normalizer_agent.py
│   │   │   ├── quality_validator_agent.py
│   │   │   └── telegram_formatter_agent.py
│   │   └── orchestrator.py
│   ├── config/
│   │   └── models_config.py
│   ├── infrastructure/
│   │   ├── ai/
│   │   │   ├── llm_provider.py
│   │   │   ├── ollama_client.py
│   │   │   └── qdrant_client.py
│   │   ├── telegram/
│   │   │   ├── telegraph_publisher.py # v3.0 (JSON Node API)
│   │   │   ├── telegram_publisher.py  # v2.1 (detailed logging)
│   │   │   └── telegram_config.py
│   │   └── skiplist/
│   ├── domain/entities/
│   │   └── article.py
│   └── scrapers/habr/
│       └── scraper_service.py
├── streamlit_app/                     # Frontend Dashboard
│   ├── app.py
│   ├── settings_page.py
│   └── models.py
├── run_full_pipeline.py               # Полный конвейер (parse + AI + publish)
├── run_scraper.py                     # Парсер (--url, --feed)
├── process_existing_articles.py       # AI обработка из БД
├── publish_pending.py                 # Публикация processed → Telegraph + Telegram
├── generate_rss.py                    # Генерация RSS feed
├── cleanup_articles.py                # Очистка старых статей
├── telegram_tool.py                   # Утилита Telegram бота
├── requirements-actions.txt           # Зависимости для GitHub Actions
├── supabase_migration.sql             # SQL миграция для Supabase
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Troubleshooting

### Pipeline timeout (10 мин)

StyleNormalizer на длинных статьях может не укладываться. Решение: более лёгкая модель или закомментировать `agent_models`.

### Ollama: "Failed to resolve 'ollama'"

Скрипт запущен из WSL, а Ollama в Docker. Используйте `docker compose exec api ...` или `OLLAMA_BASE_URL=http://localhost:11434`.

### Supabase: "Tenant or user not found"

Проект на паузе (7 дней без активности). Разпаузить в Supabase dashboard, подождать 1-2 минуты.

### WSL DNS: "Temporary failure in name resolution"

```bash
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

Перманентно:

```bash
sudo tee /etc/wsl.conf << 'EOF'
[network]
generateResolvConf = false
EOF
# В PowerShell: wsl --shutdown, потом открыть WSL заново
```

### Telegram: "bot_token пустой"

Добавить в `docker-compose.yml` секцию `api` → `environment`:

```yaml
TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}
TELEGRAM_CHAT_ID: ${TELEGRAM_CHAT_ID}
```

### Все провайдеры возвращают 429

Rate limit. Подождите 1-2 минуты или переключитесь на Ollama.

### Логи

```bash
docker compose logs api -f                                        # Все
docker compose logs api -f | grep -E "(Telegraph|Telegram)"       # Публикация
docker compose logs api -f | grep -E "(ERROR|WARNING)"            # Ошибки
```

---

## API ключи (бесплатные)

| Провайдер | Регистрация | Формат ключа |
|-----------|-------------|--------------|
| **Groq** | [console.groq.com](https://console.groq.com) | `gsk_...` |
| **Google AI** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `AIzaSy...` |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai/keys) | `sk-or-v1-...` |

Для Ollama ключи не нужны.

---

## Roadmap

- [x] Multi-provider LLM с Smart Fallback
- [x] 7-агентный AI пайплайн с chunking
- [x] Per-agent модели (models.yaml → agent_models)
- [x] Telegram + Telegraph публикация
- [x] Telegraph v3.0 (JSON Node API, изображения между абзацами)
- [x] Telegram Formatter v10.1 (bold ключевые моменты, полный пост)
- [x] Skiplist для проблемных URL
- [x] Docker-first деплой
- [x] Supabase интеграция (облачная PostgreSQL)
- [x] GitHub Actions: автопарсинг каждые 2 дня
- [x] GitHub Actions: автопубликация ежедневно
- [x] RSS feed (GitHub Pages)
- [x] telegram_tool.py (статус, тест, отправка из БД)
- [x] process_existing_articles.py (AI из Supabase)
- [ ] Веб-дашборд (GUI) с tray icon
- [ ] Миграция Qdrant → pgvector (Supabase)
- [ ] Prometheus метрики
- [ ] Параллельная обработка чанков
- [ ] RAG (семантический поиск по статьям)
- [ ] Мультиязычная поддержка
