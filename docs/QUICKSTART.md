# Быстрый Старт

## 3 Шага до Запуска

### 1. Подготовка

```bash
# Клонировать
git clone <repo-url>
cd news-aggregator-pro

# Создать .env
cp .env.example .env
```

### 2. Запуск

```bash
# Docker (рекомендуется)
docker-compose up -d

# Или локально
pip install -r requirements.txt
uvicorn src.main:app --reload
```

### 3. Проверка

```bash
# Health check
curl http://localhost:8000/health

# API Docs
open http://localhost:8000/docs
```

## Что Дальше?

- 📖 Читайте [README.md](README.md) для полной документации
- 🧪 Запустите тесты: `pytest`
- 🚀 Деплойте в production

**Готово!** 🎉
