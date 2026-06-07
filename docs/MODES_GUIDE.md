# 📚 Руководство по 3 Режимам Работы

## 🎯 Обзор Режимов

```
MODE 1: PARSING ONLY ✅
├─ Только парсинг Habr
├─ Сохранение в PostgreSQL
└─ status: "pending"

MODE 2: AI PROCESSING 🤖
├─ Берёт статьи со status="pending"
├─ AI обработка (Ollama)
│   ├─ Классификация (новость/статья)
│   ├─ Оценка релевантности (0-10)
│   ├─ Создание тизера
│   └─ Улучшение заголовка
└─ status: "processed"

MODE 3: FULL PIPELINE 🚀
├─ Парсинг
├─ Qdrant проверка дубликатов
├─ AI обработка
├─ Сохранение в PostgreSQL
├─ Векторное сохранение в Qdrant
└─ status: "completed"
```

---

## 📖 MODE 1: Только Парсинг

### Запуск:

```bash
# В Docker
docker-compose exec api python scripts/pipeline/run_scraper.py 20 "python,devops"

# Локально (если установлены зависимости)
python scripts/pipeline/run_scraper.py 20 "python,devops"
```

### Результат:
- ✅ 20 статей спарсено
- ✅ Сохранено в PostgreSQL
- ⏳ status = "pending"
- ⏳ Нет AI обработки
- ⏳ Нет векторов в Qdrant

### Когда использовать:
- Быстрый сбор данных
- Тестирование парсера
- Сбор архива для последующей обработки

---

## 🤖 MODE 2: AI Обработка

### Запуск:

```bash
# Обработать 10 необработанных статей
docker-compose exec api python scripts/pipeline/process_articles.py 10

# Обработать все pending
docker-compose exec api python scripts/pipeline/process_articles.py 1000
```

### Что делает:

1. **Classifier Agent** → определяет is_news
   ```
   НОВОСТЬ: "Релиз Python 3.12", "Apple представила..."
   СТАТЬЯ: "Как настроить Docker", "Туториал по React"
   ```

2. **Relevance Agent** → оценивает relevance_score (0-10)
   ```
   10: Прорывные технологии, major релизы
   7-9: Новые подходы, полезные инструменты
   4-6: Обычные туториалы
   1-3: Базовые темы
   0: Не релевантно
   ```

3. **Summarizer Agent** → создаёт editorial_teaser
   ```
   Краткое описание в 2-3 предложениях
   ```

4. **Rewriter Agent** → улучшает editorial_title
   ```
   Было: "Статья про Docker"
   Стало: "Docker для начинающих: полное руководство"
   ```

### Результат:
- ✅ is_news заполнен
- ✅ relevance_score (0-10)
- ✅ editorial_teaser
- ✅ editorial_title
- ✅ status = "processed"

### Когда использовать:
- Обработка уже собранных статей
- Экспериментирование с промптами
- Раздельная обработка (парсинг + AI)

---

## 🚀 MODE 3: Полный Пайплайн

### Запуск:

```bash
# 20 статей с полной обработкой
docker-compose exec api python scripts/pipeline/run_full_pipeline.py 20 "python,devops"

# 50 статей, все хабы
docker-compose exec api python scripts/pipeline/run_full_pipeline.py 50
```

### Что делает:

```
Статья → Qdrant Check → AI Processing → PostgreSQL + Qdrant
         (дубликат?)     (4 агента)     (сохранение)
```

**Детальный процесс:**

1. **Парсинг** - собирает статьи с Habr
2. **Qdrant проверка** - схожесть > 90% = дубликат
3. **PostgreSQL проверка** - exists_by_url
4. **AI обработка** - все 4 агента
5. **Сохранение** - PostgreSQL + Qdrant вектор

### Результат:
- ✅ Полная обработка
- ✅ Нет дубликатов
- ✅ AI метаданные
- ✅ Векторный поиск готов
- ✅ status = "completed"

### Когда использовать:
- Production сбор данных
- Максимальное качество
- Когда нужен векторный поиск

---

## 📊 Просмотр Данных

### 1. Swagger UI
```
http://localhost:8000/docs
```
✅ Интерактивный API  
✅ Фильтры  
✅ Пагинация  

### 2. API Endpoints

```bash
# Все статьи
curl http://localhost:8000/api/v1/articles/

# Только обработанные AI
curl http://localhost:8000/api/v1/articles/?status=processed

# Только новости
curl http://localhost:8000/api/v1/articles/?is_news=true

# Топ по релевантности
curl http://localhost:8000/api/v1/articles/?limit=10
# (добавить сортировку в API если нужно)
```

### 3. PostgreSQL

```bash
# Подключиться
docker-compose exec postgres psql -U newsaggregator -d news_aggregator

# Статистика
SELECT 
  status,
  COUNT(*) as count,
  AVG(relevance_score) as avg_score
FROM articles 
GROUP BY status;

# Топ статей
SELECT title, relevance_score, is_news 
FROM articles 
WHERE relevance_score >= 8
ORDER BY relevance_score DESC
LIMIT 10;
```

---

## 🔧 Требуемые Модели Ollama

### Установка моделей:

```bash
# Mistral (быстрая, универсальная)
docker-compose exec ollama ollama pull mistral

# Llama 3 (качественная)
docker-compose exec ollama ollama pull llama3

# DeepSeek R1 20B (опционально, для лучшего качества)
docker-compose exec ollama ollama pull deepseek-r1:20b
```

### Проверка:

```bash
docker-compose exec ollama ollama list
```

---

## 🎓 Примеры Использования

### Сценарий 1: Быстрый Сбор + Отложенная Обработка

```bash
# Утром: собрать 100 статей (быстро)
docker-compose exec api python scripts/pipeline/run_scraper.py 100

# Вечером: обработать через AI (медленно)
docker-compose exec api python scripts/pipeline/process_articles.py 100
```

### Сценарий 2: Полный Production Pipeline

```bash
# Раз в час через cron
0 * * * * cd /path && docker-compose exec -T api python scripts/pipeline/run_full_pipeline.py 20
```

### Сценарий 3: Эксперименты с AI

```bash
# 1. Собрать тестовый набор
docker-compose exec api python scripts/pipeline/run_scraper.py 10

# 2. Протестировать разные промпты
docker-compose exec api python scripts/pipeline/process_articles.py 5

# 3. Проверить результаты
curl http://localhost:8000/api/v1/articles/?limit=5
```

---

## 📈 Производительность

| Режим | Скорость | Ресурсы | Качество |
|-------|----------|---------|----------|
| MODE 1 | ⚡⚡⚡ Быстро | CPU | - |
| MODE 2 | ⚡ Медленно | GPU/CPU + RAM | ⭐⭐⭐ |
| MODE 3 | ⚡ Медленно | GPU/CPU + RAM | ⭐⭐⭐⭐⭐ |

**Примерное время:**
- MODE 1: ~1 сек/статья
- MODE 2: ~10-30 сек/статья (зависит от модели)
- MODE 3: ~15-35 сек/статья

---

## ❓ FAQ

**Q: Какой режим использовать?**  
A: MODE 3 для production, MODE 1+2 для экспериментов

**Q: Нужен ли GPU?**  
A: Нет, но с GPU будет быстрее (Ollama поддерживает CUDA)

**Q: Можно ли изменить промпты?**  
A: Да, редактируйте файлы в `src/application/ai_services/agents/`

**Q: Qdrant обязателен?**  
A: Нет, MODE 2 работает без него

**Q: Как мониторить прогресс?**  
A: Смотрите логи: `docker-compose logs -f api`
