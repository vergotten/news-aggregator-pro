# 🤖 Руководство по Моделям Ollama

## 📦 Какие Модели Нужны

### ✅ Обязательные (Минимум)

#### 1. **Mistral** (~4.1 GB)
```bash
docker-compose exec ollama ollama pull mistral:latest
```

**Использование:**
- ✅ Классификация (новость/статья)
- ✅ Оценка релевантности
- ✅ Улучшение заголовков
- ✅ Суммаризация (по умолчанию)

**Характеристики:**
- Скорость: ⚡⚡⚡ Быстрая
- Качество: ⭐⭐⭐ Хорошее
- RAM: ~4-6 GB

---

### ⭐ Рекомендуемые

#### 2. **Llama 3** (~4.7 GB)
```bash
docker-compose exec ollama ollama pull llama3:latest
```

**Использование:**
- ✅ Суммаризация (лучше чем Mistral)
- ✅ Генерация текста

**Характеристики:**
- Скорость: ⚡⚡ Средняя
- Качество: ⭐⭐⭐⭐ Отличное
- RAM: ~6-8 GB

---

### 🔥 Для Максимального Качества

#### 3. **DeepSeek R1 20B** (~13 GB)
```bash
docker-compose exec ollama ollama pull deepseek-r1:20b
```

**Использование:**
- ✅ ВСЁ (лучшее качество)
- ✅ Сложные задачи
- ✅ Reasoning

**Характеристики:**
- Скорость: ⚡ Медленная (~30-60 сек на задачу)
- Качество: ⭐⭐⭐⭐⭐ Превосходное
- RAM: ~20-24 GB (ВАЖНО!)

**⚠️ Требования:**
- Минимум 20 GB RAM
- Рекомендуется GPU

---

## 🚀 Установка Всех Моделей

### Способ 1: Автоматический Скрипт (Рекомендуется)

```bash
# Запустить скрипт
scripts/ops/install_models.sh

# Он установит:
# 1. mistral:latest
# 2. llama3:latest
# 3. Спросит про deepseek-r1:20b
```

### Способ 2: Вручную

```bash
# Обязательные
docker-compose exec ollama ollama pull mistral:latest
docker-compose exec ollama ollama pull llama3:latest

# Опционально (если есть RAM)
docker-compose exec ollama ollama pull deepseek-r1:20b
```

### Способ 3: При Сборке Docker (Медленно)

```bash
# Раскомментировать в docker/ollama/Dockerfile
# Пересобрать:
docker-compose build ollama
docker-compose up -d ollama
```

---

## 📊 Сравнение Моделей

| Модель | Размер | Скорость | Качество | RAM | Задачи |
|--------|--------|----------|----------|-----|--------|
| **Mistral** | 4.1 GB | ⚡⚡⚡ | ⭐⭐⭐ | 4-6 GB | Все базовые |
| **Llama 3** | 4.7 GB | ⚡⚡ | ⭐⭐⭐⭐ | 6-8 GB | Суммаризация |
| **DeepSeek R1** | 13 GB | ⚡ | ⭐⭐⭐⭐⭐ | 20+ GB | Сложные |

---

## ⚙️ Настройка Моделей в Агентах

### По Умолчанию (Текущая Конфигурация)

```python
# src/application/ai_services/agents/

classifier_agent.py:    model="mistral:latest"
relevance_agent.py:     model="mistral:latest"
summarizer_agent.py:    model="mistral:latest"  # ← ИСПРАВЛЕНО
rewriter_agent.py:      model="mistral:latest"
```

### Использовать Llama 3 для Суммаризации

```python
# src/application/ai_services/agents/summarizer_agent.py

response = self.ollama.generate(
    prompt=prompt,
    model="llama3:latest",  # ← Изменить на llama3
    temperature=0.7,
    max_tokens=150
)
```

### Использовать DeepSeek R1 для Всего

```python
# Во всех агентах изменить на:
model="deepseek-r1:20b"
```

**⚠️ Внимание:** DeepSeek ОЧЕНЬ медленный (~30-60 сек на задачу)

---

## 💾 Где Хранятся Модели

```bash
# Docker volume
ollama_data:/root/.ollama

# Модели сохраняются между перезапусками!
# Скачивать нужно только ОДИН РАЗ
```

### Проверка

```bash
# Список моделей
docker-compose exec ollama ollama list

# Размер volume
docker volume inspect ollama_data

# Удалить все модели (очистить volume)
docker-compose down
docker volume rm news-aggregator-pro_ollama_data
```

---

## 🎯 Рекомендации

### Для Разработки
```
mistral:latest
```
- Быстро
- Достаточное качество
- Мало RAM

### Для Production
```
mistral:latest + llama3:latest
```
- Баланс скорости и качества
- Mistral для классификации
- Llama3 для суммаризации

### Для Максимального Качества
```
deepseek-r1:20b
```
- Лучшее качество
- Требует мощный сервер
- Медленно (~5-10x медленнее Mistral)

---

## 📈 Производительность

### С Mistral
```
1 статья: ~20-30 сек
10 статей: ~3-5 минут
100 статей: ~30-50 минут
```

### С Llama 3
```
1 статья: ~30-40 сек
10 статей: ~5-7 минут
100 статей: ~50-70 минут
```

### С DeepSeek R1 20B
```
1 статья: ~60-120 сек
10 статей: ~10-20 минут
100 статей: ~100-200 минут
```

---

## 🔧 Устранение Проблем

### Модель не найдена
```bash
# Установить модель
docker-compose exec ollama ollama pull mistral:latest

# Проверить
docker-compose exec ollama ollama list
```

### Нехватка памяти
```bash
# Использовать меньшую модель
# mistral вместо llama3
# llama3 вместо deepseek-r1

# Или добавить RAM/swap
```

### Медленная работа
```bash
# 1. Использовать меньшую модель
# 2. Уменьшить max_tokens в агентах
# 3. Использовать GPU (если есть)
```

---

## ✅ Быстрый Старт

```bash
# 1. Установить модели
scripts/ops/install_models.sh

# 2. Проверить
docker-compose exec ollama ollama list

# 3. Запустить тест
docker-compose exec api python scripts/pipeline/run_full_pipeline.py 1

# 4. Готово! 🚀
```

---

## 🎓 Дополнительно

### Другие Полезные Модели

```bash
# Phi-3 (малая, быстрая)
ollama pull phi3:mini

# Gemma 2 (от Google)
ollama pull gemma2

# Qwen (китайская, хороша для мультиязычности)
ollama pull qwen2
```

### GPU Ускорение

Если есть NVIDIA GPU:
```bash
# Docker будет автоматически использовать GPU
# Проверить:
docker-compose exec ollama nvidia-smi
```

---

## 📚 Документация

- Ollama: https://ollama.ai/
- Модели: https://ollama.ai/library
- DeepSeek R1: https://github.com/deepseek-ai/DeepSeek-R1
