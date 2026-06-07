# 🔄 Миграция Моделей Ollama в Локальную Директорию

## 🎯 Что Изменилось

### БЫЛО (Docker Volume):
```
Модели → Docker Volume (ollama_data)
Путь: /var/lib/docker/volumes/...
❌ Удаляются при: docker-compose down -v
❌ Не видны в файловой системе
```

### СТАЛО (Bind Mount):
```
Модели → Локальная папка (./ollama_models)
Путь: news-aggregator-pro/ollama_models/
✅ НЕ удаляются при docker-compose down -v
✅ Видны в файловой системе
✅ Легко делать бэкап
✅ Переносимы между машинами
```

---

## 📂 Структура

```
news-aggregator-pro/
├── ollama_models/           ← НОВАЯ ПАПКА!
│   ├── models/
│   │   ├── manifests/
│   │   │   └── registry.ollama.ai/
│   │   │       └── library/
│   │   │           ├── mistral/
│   │   │           └── llama3/
│   │   └── blobs/
│   │       ├── sha256-abc... (4.4 GB - mistral)
│   │       ├── sha256-def... (4.7 GB - llama3)
│   │       └── ...
│   └── ...
├── docker-compose.yml
├── src/
└── ...
```

---

## 🔄 МИГРАЦИЯ СУЩЕСТВУЮЩИХ МОДЕЛЕЙ

### Способ 1: Автоматический (Рекомендуется)

```bash
cd /mnt/f/Code/news-aggregator-pro

# 1. Создать директорию
mkdir -p ollama_models

# 2. Копировать модели из старого volume
docker run --rm \
  -v news-aggregator-pro_ollama_data:/source \
  -v $(pwd)/ollama_models:/destination \
  alpine sh -c "cp -r /source/* /destination/"

# 3. Проверить что скопировалось
ls -lh ollama_models/

# 4. Перезапустить с новыми настройками
docker-compose down
docker-compose up -d ollama

# 5. Проверить что модели доступны
docker-compose exec ollama ollama list

# 6. (Опционально) Удалить старый volume
docker volume rm news-aggregator-pro_ollama_data
```

### Способ 2: Скачать Заново

Если миграция не получается или хотите чистый старт:

```bash
cd /mnt/f/Code/news-aggregator-pro

# 1. Создать директорию
mkdir -p ollama_models

# 2. Перезапустить
docker-compose down
docker-compose up -d ollama

# 3. Скачать модели заново
docker-compose exec ollama ollama pull mistral:latest
docker-compose exec ollama ollama pull llama3:latest

# Готово!
```

---

## ✅ ПРОВЕРКА

### Модели в директории:

```bash
# Linux/WSL
ls -lh ollama_models/models/blobs/

# Windows (PowerShell)
dir ollama_models\models\blobs\
```

**Должно быть:**
```
-rw-r--r-- 1 root root 4.4G  sha256-abc123...
-rw-r--r-- 1 root root 4.7G  sha256-def456...
...
```

### Размер директории:

```bash
# Linux/WSL
du -sh ollama_models/

# Windows (PowerShell)
Get-ChildItem ollama_models -Recurse | Measure-Object -Property Length -Sum
```

**Ожидается:** ~9-10 GB (mistral + llama3)

### Модели работают:

```bash
docker-compose exec ollama ollama list
```

---

## 🎯 ПРЕИМУЩЕСТВА

### 1. Защита от Удаления

```bash
# БЕЗОПАСНО - модели НЕ удалятся!
docker-compose down -v
docker volume prune
docker system prune -a --volumes
```

Модели остаются в `./ollama_models` ✅

### 2. Бэкап / Перенос

```bash
# Запаковать модели
tar -czf ollama_models_backup.tar.gz ollama_models/

# Скопировать на другой компьютер
scp ollama_models_backup.tar.gz user@server:/path/

# Распаковать
tar -xzf ollama_models_backup.tar.gz
```

### 3. Git Ignore

```
# .gitignore уже настроен:
ollama_models/
```

Модели НЕ попадут в git (слишком большие)

### 4. Видимость

Можете смотреть файлы прямо в проводнике:
```
Проводник → news-aggregator-pro → ollama_models
```

---

## 📊 УПРАВЛЕНИЕ

### Добавить Модель

```bash
docker-compose exec ollama ollama pull qwen2:latest
```

Скачается в: `./ollama_models/models/blobs/`

### Удалить Модель

```bash
docker-compose exec ollama ollama rm mistral:latest
```

Удалится из: `./ollama_models/`

### Очистить Всё

```bash
# Остановить Ollama
docker-compose stop ollama

# Удалить директорию
rm -rf ollama_models/

# Создать заново
mkdir ollama_models

# Запустить и скачать модели
docker-compose up -d ollama
docker-compose exec ollama ollama pull mistral:latest
```

---

## 🔐 ПРАВА ДОСТУПА (Linux/macOS)

Если возникают проблемы с правами:

```bash
# Дать права на чтение/запись
sudo chown -R $USER:$USER ollama_models/

# Или для Docker
sudo chown -R 1000:1000 ollama_models/
```

---

## 🐛 Устранение Проблем

### Модели не загружаются

```bash
# 1. Проверить что папка существует
ls -la ollama_models/

# 2. Проверить монтирование
docker inspect news-aggregator-ollama | grep ollama_models

# 3. Пересоздать контейнер
docker-compose down
docker-compose up -d ollama
```

### Нет места на диске

```bash
# Проверить место
df -h

# Удалить ненужные модели
docker-compose exec ollama ollama list
docker-compose exec ollama ollama rm <model-name>
```

### Старый volume конфликтует

```bash
# Удалить старый volume
docker volume rm news-aggregator-pro_ollama_data

# Если не удаляется
docker-compose down
docker volume rm news-aggregator-pro_ollama_data
```

---

## ✅ Быстрая Миграция (копируй и выполняй)

```bash
cd /mnt/f/Code/news-aggregator-pro

# Создать директорию
mkdir -p ollama_models

# Копировать из старого volume
docker run --rm \
  -v news-aggregator-pro_ollama_data:/source \
  -v $(pwd)/ollama_models:/destination \
  alpine sh -c "cp -r /source/* /destination/"

# Перезапустить
docker-compose down
docker-compose up -d

# Проверить
docker-compose exec ollama ollama list

# Удалить старый volume
docker volume rm news-aggregator-pro_ollama_data

# Готово! ✅
```

---

## 📚 Итого

**Модели теперь в:**
```
./ollama_models/
```

**Не удаляются при:**
```
✅ docker-compose down -v
✅ docker volume prune
✅ docker system prune --volumes
✅ Удалении контейнеров
```

**Легко:**
```
✅ Делать бэкап (tar/zip)
✅ Переносить между компьютерами
✅ Смотреть в проводнике
✅ Контролировать размер
```

**Готово!** 🚀
