# 🕷️ Руководство по Запуску Парсеров

## 🎯 Быстрый Старт

### Вариант 1: Простой Запуск (Рекомендуется)

```bash
# Запустить парсер Habr (10 статей)
python run_scraper.py

# С параметрами
python run_scraper.py 50                    # 50 статей
python run_scraper.py 30 "python,devops"    # 30 статей из хабов
```

### Вариант 2: CLI с Rich Progress

```bash
# Установить зависимости
pip install click rich

# Запустить через CLI
python cli.py scrape habr --limit 20
python cli.py scrape habr --limit 50 --hubs "python,machine_learning"
```

### Вариант 3: Через Docker

```bash
# Запустить в контейнере
docker-compose exec api python run_scraper.py 20

# Или через CLI
docker-compose exec api python cli.py scrape habr --limit 30
```

---

## 📖 Детальное Руководство

### 1. Подготовка

```bash
# Убедиться что БД запущена
docker-compose up -d postgres

# Или проверить подключение
psql postgresql://newsaggregator:changeme123@localhost:5433/news_aggregator
```

### 2. Установка Зависимостей

```bash
pip install -r requirements.txt
```

Необходимые пакеты:
- `beautifulsoup4` - парсинг HTML
- `lxml` - быстрый парсер
- `aiohttp` - асинхронные HTTP запросы
- `click` - CLI framework
- `rich` - красивый вывод в консоль

### 3. Запуск Парсера

#### Способ A: Простой скрипт

```bash
# Базовый запуск (10 статей)
python run_scraper.py

# 50 статей
python run_scraper.py 50

# 30 статей из конкретных хабов
python run_scraper.py 30 "python,devops,docker"
```

**Пример вывода:**
```
🚀 Запуск Habr парсера
   Лимит: 10
   Хабы: все

✅ Готово!
   Собрано: 10
   Сохранено: 8
   Дубликатов: 2
   Ошибок: 0
```

#### Способ B: CLI с прогресс-баром

```bash
# Базовый запуск
python cli.py scrape habr --limit 10

# С фильтром по хабам
python cli.py scrape habr --limit 50 --hubs "python,javascript"

# Только из хаба Python
python cli.py scrape habr --limit 20 --hubs "python"
```

**Пример вывода:**
```
🚀 Запуск Habr парсера
Лимит: 20
Хабы: python

Парсинг статей... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 20/20

✅ Готово!
Собрано: 20
Сохранено: 18
Дубликатов: 2
Ошибок: 0
```

#### Способ C: Из Docker контейнера

```bash
# Запустить в существующем контейнере
docker-compose exec api python run_scraper.py 15

# С хабами
docker-compose exec api python run_scraper.py 30 "python,devops"

# Через CLI
docker-compose exec api python cli.py scrape habr --limit 25
```

### 4. Программный Запуск (Python API)

```python
import asyncio
from src.scrapers.habr.scraper_service import HabrScraperService

async def main():
    """Программный запуск парсера."""
    service = HabrScraperService()
    
    # Запустить парсинг
    results = await service.scrape_and_save(
        limit=20,
        hubs=['python', 'devops']
    )
    
    print(f"Сохранено: {results['saved']}")
    print(f"Дубликатов: {results['duplicates']}")

# Запуск
asyncio.run(main())
```

### 5. Проверка Результатов

```bash
# Через psql
psql postgresql://newsaggregator:changeme123@localhost:5433/news_aggregator

# SQL запросы
SELECT COUNT(*) FROM articles;
SELECT title, author, created_at FROM articles ORDER BY created_at DESC LIMIT 10;
SELECT * FROM articles WHERE source = 'habr';
```

**Или через API:**
```bash
# Получить все статьи
curl http://localhost:8000/api/v1/articles/

# Последние 5
curl "http://localhost:8000/api/v1/articles/?limit=5"
```

---

## ⚙️ Параметры Парсера

### Основные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `limit` | int | 10 | Количество статей |
| `hubs` | str | "" | Хабы через запятую |
| `progress_callback` | callable | None | Функция прогресса |

### Примеры использования

```bash
# 100 статей
python run_scraper.py 100

# Из топ хабов
python run_scraper.py 50 "python,javascript,devops,docker,kubernetes"

# Только Python
python run_scraper.py 30 "python"
```

---

## 🔧 Расширенное Использование

### Кастомная обработка

```python
import asyncio
from src.scrapers.habr.scraper_service import HabrScraperService

async def scrape_with_progress():
    """Парсинг с кастомным прогрессом."""
    service = HabrScraperService()
    
    count = 0
    def progress():
        nonlocal count
        count += 1
        print(f"Обработано: {count}")
    
    results = await service.scrape_and_save(
        limit=50,
        hubs=['python'],
        progress_callback=progress
    )
    
    print(f"\nИтого: {results['saved']} статей сохранено")

asyncio.run(scrape_with_progress())
```

### Фоновый парсинг (Celery-style)

```python
# tasks.py
import asyncio
from src.scrapers.habr.scraper_service import HabrScraperService

async def background_scrape():
    """Фоновая задача парсинга."""
    service = HabrScraperService()
    
    while True:
        # Парсить каждый час
        results = await service.scrape_and_save(limit=20)
        print(f"Сохранено: {results['saved']}")
        
        # Подождать 1 час
        await asyncio.sleep(3600)

# Запуск
asyncio.run(background_scrape())
```

---

## 🐛 Troubleshooting

### Проблема: "Connection refused" при обращении к БД

**Решение:**
```bash
# Проверить что PostgreSQL запущен
docker-compose ps postgres

# Если нет - запустить
docker-compose up -d postgres

# Проверить подключение
psql postgresql://newsaggregator:changeme123@localhost:5433/news_aggregator -c "SELECT version();"
```

### Проблема: "Table does not exist"

**Решение:**
```bash
# Инициализировать БД
docker-compose exec postgres psql -U newsaggregator -d news_aggregator -f /docker-entrypoint-initdb.d/init.sql

# Или применить миграции (если используете Alembic)
alembic upgrade head
```

### Проблема: Парсер не находит статьи

**Решение:**
- Проверить интернет соединение
- Возможно изменилась структура HTML на Habr
- Обновить селекторы в `scraper_service.py`

### Проблема: "Already exists" ошибки

**Решение:**
Это нормально - парсер проверяет дубликаты по URL.
Дубликаты учитываются в статистике.

---

## 📊 Мониторинг

### Логирование

```python
import logging

# Включить debug логи
logging.basicConfig(level=logging.DEBUG)

# Запустить парсер
python run_scraper.py 10
```

### Метрики

```python
# После парсинга проверить метрики
results = await service.scrape_and_save(limit=100)

print(f"Успешность: {results['saved'] / results['scraped'] * 100:.1f}%")
print(f"Дубликатов: {results['duplicates']}")
print(f"Ошибок: {results['errors']}")
```

---

## 🔄 Автоматизация

### Cron Job (Linux)

```bash
# Добавить в crontab
crontab -e

# Парсить каждый час
0 * * * * cd /path/to/project && python run_scraper.py 20 >> /var/log/scraper.log 2>&1

# Парсить каждые 6 часов
0 */6 * * * cd /path/to/project && python run_scraper.py 50 "python,devops"
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/news-scraper.service
[Unit]
Description=News Aggregator Scraper
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python3 run_scraper.py 30
Restart=always
RestartSec=3600

[Install]
WantedBy=multi-user.target
```

```bash
# Запустить
sudo systemctl start news-scraper
sudo systemctl enable news-scraper
```

### n8n Workflow

Используйте встроенный n8n для автоматизации:

1. Откройте http://localhost:5678
2. Создайте workflow:
   - Schedule Trigger (каждый час)
   - Execute Command: `python run_scraper.py 20`
3. Активируйте workflow

---

## 📚 Дополнительно

### Создание своего парсера

```python
# src/scrapers/medium/scraper_service.py
from src.scrapers.base_scraper import BaseScraper

class MediumScraperService(BaseScraper):
    """Парсер для Medium."""
    
    async def scrape_articles(self, limit: int):
        # Ваша логика парсинга
        pass
```

### Интеграция с другими источниками

Смотрите примеры:
- `src/scrapers/habr/` - Habr
- `src/scrapers/telegram/` - Telegram (в разработке)
- `src/scrapers/reddit/` - Reddit (в разработке)

---

## ✅ Checklist

- [ ] PostgreSQL запущен
- [ ] Зависимости установлены (`pip install -r requirements.txt`)
- [ ] Проверено подключение к БД
- [ ] Запущен первый парсинг (`python run_scraper.py`)
- [ ] Проверены результаты в БД
- [ ] Настроена автоматизация (опционально)

**Готово! 🎉**

Начните с: `python run_scraper.py 10`
