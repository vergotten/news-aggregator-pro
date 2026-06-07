# 🎨 Directus CMS - Руководство

## 📊 Что такое Directus?

Directus - это современная админ-панель для PostgreSQL с:
- ✅ Визуализация всех таблиц
- ✅ Редактирование данных
- ✅ Создание дашбордов
- ✅ REST API
- ✅ GraphQL API
- ✅ Файловый менеджер

---

## 🚀 Запуск

### 1. Добавить в .env (если ещё не создан)

```bash
cp .env.example .env
```

### 2. Запустить Directus

```bash
docker-compose up -d directus
```

**Первый запуск займёт 1-2 минуты** - Directus создаст свои таблицы

### 3. Открыть в браузере

```
http://localhost:8055
```

### 4. Войти

```
Email: admin@example.com
Password: admin
```

**⚠️ ВАЖНО:** Смените пароль после первого входа!

---

## 🎯 Что Можно Делать

### 1. Просмотр Статей

```
Content → articles
```

Вы увидите все статьи с:
- ✅ Оригинальными заголовками
- ✅ AI-улучшенными заголовками
- ✅ Тизерами
- ✅ Оценками релевантности
- ✅ Тегами и хабами

### 2. Фильтрация

```
Filters → Add Filter
```

Примеры:
- `is_news = true` - только новости
- `relevance_score >= 8` - только топовые статьи
- `source = habr` - только с Habr
- `status = processed` - только обработанные AI

### 3. Сортировка

Кликните на заголовок столбца:
- `created_at` - по дате
- `relevance_score` - по оценке
- `title` - по алфавиту

### 4. Экспорт

```
⋮ (три точки) → Export Items → CSV/JSON
```

Скачает все отфильтрованные статьи!

---

## 📊 Создание Дашборда

### Пример: "Топ Статей"

1. **Insights → Create Panel**

2. **Metric (Number)**
   - Collection: `articles`
   - Function: `count`
   - Label: "Всего статей"

3. **List**
   - Collection: `articles`
   - Filter: `relevance_score >= 8`
   - Sort: `relevance_score DESC`
   - Limit: 10
   - Display: `title, relevance_score`

4. **Chart (Bar)**
   - Collection: `articles`
   - Group By: `source`
   - Function: `count`

---

## ⚙️ Настройки

### Сменить Пароль Администратора

```
Settings → Users → Admin → Edit
→ Password → Сохранить
```

### Добавить Пользователя

```
Settings → Users → Create User
```

Роли:
- **Administrator** - полный доступ
- **Public** - только чтение

### Настроить Коллекции

```
Settings → Data Model → articles
```

Можно:
- Скрыть поля
- Изменить порядок
- Добавить валидацию
- Настроить отображение

---

## 🔧 Расширенные Функции

### 1. REST API

```bash
# Получить все статьи
curl http://localhost:8055/items/articles

# С фильтром
curl "http://localhost:8055/items/articles?filter[is_news][_eq]=true"

# С аутентификацией
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8055/items/articles
```

### 2. GraphQL API

```
http://localhost:8055/graphql
```

Пример запроса:
```graphql
query {
  articles(filter: {is_news: {_eq: true}}) {
    id
    title
    editorial_title
    relevance_score
  }
}
```

### 3. Webhooks

```
Settings → Webhooks → Create Webhook
```

Можно отправлять уведомления при:
- Создании статьи
- Обновлении статьи
- Достижении определённого relevance_score

---

## 📱 Мобильное Приложение

Directus работает на мобильных устройствах!

Просто откройте `http://localhost:8055` на телефоне (если в одной сети)

---

## 🎨 Кастомизация

### Тёмная Тема

```
User Menu (справа вверху) → Dark Mode
```

### Язык

```
Settings → Project Settings → Default Language → Russian
```

### Логотип

```
Settings → Project Settings → Project Logo
```

---

## 🔐 Безопасность

### Production Настройки

В `.env` измените:

```bash
# Генерируйте случайные значения!
DIRECTUS_KEY=$(openssl rand -base64 32)
DIRECTUS_SECRET=$(openssl rand -base64 32)

# Сложный пароль
DIRECTUS_ADMIN_PASSWORD=very-strong-password-here

# Реальный email
DIRECTUS_ADMIN_EMAIL=your-email@example.com
```

### Ограничить Доступ

В `docker-compose.yml`:

```yaml
directus:
  ports:
    - "127.0.0.1:8055:8055"  # Только localhost
```

Или используйте reverse proxy (Nginx, Traefik)

---

## 📊 Примеры Использования

### Найти Статьи без AI Обработки

```
Filters:
  relevance_score is null
  OR
  editorial_teaser is null
```

### Топ 10 Статей Недели

```
Filters:
  created_at >= (сегодня - 7 дней)

Sort:
  relevance_score DESC

Limit: 10
```

### Экспорт для Рассылки

```
Filters:
  is_news = true
  created_at >= (сегодня)
  relevance_score >= 7

Export → JSON
```

---

## 🛠️ Устранение Проблем

### Directus не запускается

```bash
# Проверить логи
docker-compose logs directus

# Пересоздать контейнер
docker-compose down
docker-compose up -d directus
```

### Забыли пароль

```bash
# Сбросить через CLI
docker-compose exec directus npx directus users create \
  --email admin@example.com \
  --password newpassword \
  --role administrator
```

### Медленно работает

```bash
# Проверить нагрузку на PostgreSQL
docker stats news-aggregator-db

# Добавить индексы если нужно
```

---

## 📚 Ресурсы

- Официальная документация: https://docs.directus.io/
- GitHub: https://github.com/directus/directus
- Community: https://directus.chat/

---

## ✅ Быстрый Старт

```bash
# 1. Запустить Directus
docker-compose up -d directus

# 2. Подождать 1-2 минуты

# 3. Открыть в браузере
open http://localhost:8055

# 4. Войти
# Email: admin@example.com
# Password: admin

# 5. Перейти в Content → articles

# 6. Готово! 🎉
```
