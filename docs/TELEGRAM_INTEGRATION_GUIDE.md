# 📱 Telegram Integration Guide

## 🎯 Возможности

✅ **Автоматическая публикация** статей в канал/группу  
✅ **Полный текст** + изображения  
✅ **Фильтрация** по релевантности  
✅ **Форматирование** HTML для Telegram  
✅ **Batch публикация** с задержкой

---

## 🔧 Настройка

### Шаг 1: Получить Telegram API Credentials

1. Перейти на https://my.telegram.org
2. Войти с номером телефона
3. **API Development Tools**
4. **Create Application**
5. Скопировать:
   - `api_id`
   - `api_hash`

### Шаг 2: Создать Telegram Бота (Вариант 1)

1. Написать @BotFather в Telegram
2. `/newbot`
3. Придумать имя и username
4. Скопировать **Bot Token**
5. Добавить бота в канал/группу
6. Сделать бота **администратором** (права на отправку сообщений)

### Шаг 3: Или Использовать Свой Аккаунт (Вариант 2)

Если хотите публиковать от своего имени:
- Не нужен Bot Token
- Используйте свой номер телефона
- Вы должны быть администратором канала

### Шаг 4: Настроить .env

```bash
# Скопировать пример
cp .env.example .env

# Отредактировать
nano .env
```

**Для публикации через бота:**
```bash
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHANNEL=@your_channel  # или -1001234567890
```

**Для публикации от своего аккаунта:**
```bash
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
# Не указывайте TELEGRAM_BOT_TOKEN
TELEGRAM_PHONE=+1234567890
TELEGRAM_CHANNEL=@your_channel
```

### Шаг 5: Установить Telethon

```bash
# В контейнере
docker-compose exec api pip install telethon==1.35.0

# Или локально
pip install telethon==1.35.0
```

---

## 🚀 Использование

### Публикация Статей

```bash
# Опубликовать 5 топ статей
docker-compose exec api python publish_to_telegram.py 5

# С параметрами:
# publish_to_telegram.py [количество] [мин_релевантность] [изображения] [задержка]

# Примеры:
docker-compose exec api python publish_to_telegram.py 10            # 10 статей
docker-compose exec api python publish_to_telegram.py 5 8.0         # мин. оценка 8
docker-compose exec api python publish_to_telegram.py 5 7.0 false   # без картинок
docker-compose exec api python publish_to_telegram.py 5 7.0 true 120  # задержка 120сек
```

### Вывод:

```
📱 ПУБЛИКАЦИЯ В TELEGRAM
   Канал: @my_channel
   Лимит: 5
   Мин. релевантность: 7.0
   Изображения: Да
   Задержка: 60сек

📊 Найдено статей: 8

🔌 Подключение к Telegram...
✅ Подключено!

[1/5] ChatGPT Atlas не нужен: Пишем свой AI-«браузер»
✅ Опубликовано: AI-браузер для настольных игр: практический гайд
⏳ Ждём 60 секунд...

[2/5] Подземелье c LLM-экспертами...
✅ Опубликовано: RAG система для настольных игр с LangChain
⏳ Ждём 60 секунд...

...

═══════════════════════════════════════════════════════════
📊 ИТОГИ:
   Опубликовано: 5
   Пропущено: 0
   Ошибок: 0
═══════════════════════════════════════════════════════════
```

---

## 📝 Формат Сообщения в Telegram

### Структура:

```
<b>Заголовок (AI улучшенный)</b>

<i>Краткое описание (тизер)</i>

Основной текст (нормализованный, без "я")...

🔗 Читать полностью

#python #ai #rag

📊 8/10 | 📰 Новость
```

### HTML Форматирование:

- `<b>bold</b>` - жирный
- `<i>italic</i>` - курсив
- `<a href="...">link</a>` - ссылка
- `#hashtag` - теги

---

## ⚙️ Программное Использование

```python
from src.infrastructure.telegram.telegram_publisher import TelegramPublisher
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl

async def publish():
    # Подключиться
    publisher = TelegramPublisher(
        api_id="12345678",
        api_hash="abc...",
        bot_token="123:ABC..."
    )
    
    await publisher.connect()
    
    # Получить статьи
    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)
        articles = await repo.find_all(limit=5)
    
    # Опубликовать
    for article in articles:
        await publisher.publish_article(
            article=article,
            channel="@my_channel",
            include_images=True,
            min_relevance=7.0
        )
    
    await publisher.disconnect()
```

---

## 🎯 Автоматизация

### Cron Job (Linux)

```bash
# Публиковать каждый час
0 * * * * cd /path/to/project && docker-compose exec api python publish_to_telegram.py 5 >> logs/telegram.log 2>&1
```

### systemd Timer (Linux)

```bash
# /etc/systemd/system/telegram-publisher.timer
[Unit]
Description=Publish articles to Telegram every hour

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

### Windows Task Scheduler

1. Task Scheduler → Create Task
2. Trigger: Hourly
3. Action: Run Program
   - Program: `docker-compose`
   - Arguments: `exec api python publish_to_telegram.py 5`
   - Start in: `C:\path\to\project`

---

## 🔍 Фильтрация

### По Релевантности

```bash
# Только топовые (8+)
python publish_to_telegram.py 10 8.0

# Средние и выше (6+)
python publish_to_telegram.py 20 6.0
```

### По Типу

Отредактируйте `publish_to_telegram.py`:

```python
# Только новости
filtered = [
    a for a in articles
    if a.is_news == True
    and a.relevance_score >= min_relevance
]

# Только статьи
filtered = [
    a for a in articles
    if a.is_news == False
    and a.relevance_score >= min_relevance
]
```

---

## 🖼️ Изображения

### Включить/Выключить

```bash
# С изображениями
python publish_to_telegram.py 5 7.0 true

# Без изображений
python publish_to_telegram.py 5 7.0 false
```

### Ограничения Telegram

- **Максимум 10 изображений** в одном посте
- **Форматы**: JPG, PNG, GIF
- **Размер**: до 10MB каждое

### Что Делает Publisher

1. Берёт первые 10 изображений из статьи
2. Если 1 изображение → отправляет с текстом
3. Если >1 → создаёт медиа-группу (album)

---

## 🐛 Устранение Проблем

### "Could not find the input entity"

**Проблема**: Неправильный ID канала

**Решение**:
1. Для публичного канала: `@channel_username`
2. Для приватного:
   - Добавить бота в канал
   - Сделать администратором
   - Использовать ID: `-1001234567890`

**Как узнать ID:**
```python
# Запустить в Python
from telethon import TelegramClient
client = TelegramClient('session', api_id, api_hash)
await client.start(bot_token=bot_token)
dialogs = await client.get_dialogs()
for dialog in dialogs:
    print(f"{dialog.name}: {dialog.id}")
```

### "You don't have enough rights"

**Проблема**: Бот не администратор

**Решение**:
1. Канал → Администраторы
2. Добавить бота
3. Включить "Post messages"

### "Phone number invalid"

**Проблема**: Неправильный формат номера

**Решение**:
```bash
# Правильно:
TELEGRAM_PHONE=+1234567890

# Неправильно:
TELEGRAM_PHONE=1234567890
TELEGRAM_PHONE=+1 (234) 567-89-00
```

### Первый запуск с номером телефона

При первом запуске попросит **код подтверждения**:

```bash
docker-compose exec api python publish_to_telegram.py 1

# Введите код из Telegram
# Если включена 2FA - введите пароль
```

После этого сессия сохранится в `user_session.session`

---

## 📊 Статистика

Publisher возвращает статистику:

```python
stats = await publisher.publish_batch(articles, channel)
# {
#   'published': 5,
#   'skipped': 2,   # Низкая релевантность
#   'errors': 0
# }
```

---

## ✅ Рекомендации

### Для Канала

- ✅ Используйте **Bot Token**
- ✅ Задержка 60+ секунд между постами
- ✅ Фильтр relevance >= 7.0
- ✅ Включайте изображения

### Для Группы

- ✅ Используйте **Phone** (от своего имени)
- ✅ Или бот с правами администратора
- ✅ Можно без задержки (в группах меньше ограничений)

### Для Тестов

```bash
# Создайте тестовый канал
# Публикуйте туда с delay=5
python publish_to_telegram.py 3 5.0 true 5
```

---

## 🎓 Полный Workflow

```bash
# 1. Спарсить статьи
docker-compose exec api python run_scraper.py 20

# 2. Обработать AI
docker-compose exec api python run_full_pipeline.py 20

# 3. Проверить в Directus
open http://localhost:8055

# 4. Опубликовать топовые в Telegram
docker-compose exec api python publish_to_telegram.py 5 8.0

# Готово! 🚀
```

---

## 📚 API Reference

### TelegramPublisher

```python
publisher = TelegramPublisher(
    api_id: str,           # API ID from my.telegram.org
    api_hash: str,         # API Hash
    bot_token: str = None, # Bot token (optional)
    phone: str = None      # Phone number (optional)
)

await publisher.connect()

await publisher.publish_article(
    article: Article,           # Article entity
    channel: str,               # @channel or -1001234567890
    include_images: bool = True, # Include images
    min_relevance: float = 7.0  # Min relevance score
) -> bool  # True if published

await publisher.publish_batch(
    articles: List[Article],
    channel: str,
    delay: int = 60,        # Delay between posts (seconds)
    **kwargs                # Additional params for publish_article
) -> dict  # Statistics

await publisher.disconnect()
```

---

## 🎉 Готово!

Теперь у вас:
- ✅ Парсинг полного текста
- ✅ Парсинг изображений
- ✅ AI обработка
- ✅ Автопубликация в Telegram
- ✅ Полный контроль над контентом

**Запустите прямо сейчас:**

```bash
docker-compose exec api python publish_to_telegram.py 5
```
