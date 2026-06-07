### 1. **Модульная конфигурация моделей**
Вместо одного большого `models.yaml` теперь отдельные файлы для каждого провайдера:

```
config/models/
├── models.yaml          # Главный файл с импортами и алиасами
├── base.yaml            # Общие настройки
├── openrouter.yaml      # OpenRouter профили
├── google.yaml          # Google Gemini профили  
├── groq.yaml            # Groq профили
├── ollama.yaml          # Ollama (локальные) профили
└── auto_fallback.yaml   # Auto-fallback профили
```

### 2. **Новые AI агенты**

#### TelegramFormatterAgent 📱
Форматирование статей для Telegram каналов с учётом лимита 4096 символов.

**Возможности:**
- Автоматическая адаптация под лимиты Telegram
- Создание preview для длинных статей
- Генерация Telegraph HTML контента
- Telegram HTML/Markdown форматирование
- Автоматические хештеги

**Пример использования:**
```python
from src.application.ai_services.agents import TelegramFormatterAgent

agent = TelegramFormatterAgent(default_author="YourChannel")

post = agent.format_for_telegram(
    title="Python 3.13 Released: New JIT Compiler",
    content="Full article content...",  # Любая длина
    source_url="https://habr.com/article/123",
    tags=["python", "jit", "performance"]
)

if post.telegraph_needed:
    # Статья слишком длинная - нужен Telegraph
    telegraph_url = publish_to_telegraph(
        title="Python 3.13 Released",
        author="YourChannel",
        content_html=post.telegraph_content
    )
    final_text = post.text.replace("{TELEGRAPH_URL}", telegraph_url)
else:
    # Статья помещается в Telegram
    final_text = post.text

# Отправляем в Telegram
bot.send_message(
    chat_id=CHANNEL_ID,
    text=final_text,
    parse_mode="HTML",
    disable_web_page_preview=False
)
```

**Логика работы:**
- **Короткие статьи** (< 3500 символов): Публикуются полностью
- **Средние статьи** (3500-5000): Сокращаются с сохранением сути
- **Длинные статьи** (> 5000): Preview в Telegram + полная версия в Telegraph

#### SEOOptimizerAgent 🔍
SEO оптимизация контента для поисковых систем.

**Возможности:**
- Генерация title tags (50-60 символов)
- Meta descriptions (150-160 символов)
- Подбор ключевых слов
- URL-friendly slugs
- Open Graph данные
- Анализ структуры контента

**Пример использования:**
```python
from src.application.ai_services.agents import SEOOptimizerAgent

agent = SEOOptimizerAgent()

seo = agent.optimize(
    title="Python 3.13 Released",
    content="Full article content...",
    target_keywords=["python", "jit compiler", "performance"]
)

# Используем в meta tags
print(f"<title>{seo.title_tag}</title>")
print(f'<meta name="description" content="{seo.meta_description}">')
print(f'<meta name="keywords" content="{", ".join(seo.keywords)}">')

# Open Graph
print(f'<meta property="og:title" content="{seo.og_title}">')
print(f'<meta property="og:description" content="{seo.og_description}">')

# URL
article_url = f"https://example.com/articles/{seo.slug}"
```

### 3. **Улучшенные конфигурации**

#### Быстрый старт - выбор профиля

**Для разработки:**
```bash
export LLM_PROFILE=groq_free
# Groq - самый быстрый бесплатный провайдер (30 req/min)
```

**Для production с высокой нагрузкой:**
```bash
export LLM_PROFILE=auto_aggressive
# Автоматический fallback: Groq → Google → OpenRouter → Ollama
```

**Для максимального качества:**
```bash
export LLM_PROFILE=auto_quality
# Google Pro → OpenRouter Premium → Groq
```

**Для минимальных затрат:**
```bash
export LLM_PROFILE=auto_free
# Только бесплатные ресурсы
```

**Для приватности (локально):**
```bash
export LLM_PROFILE=ollama_balanced
# Все модели работают локально через Ollama
```

#### Все доступные профили

**OpenRouter:**
- `openrouter_free` - Бесплатные модели (50 req/day)
- `openrouter_balanced` - GPT-4o-mini (оптимальное качество/цена)
- `openrouter_premium` - GPT-4o + Claude 3.5 Sonnet (максимум качества)
- `openrouter_hybrid` - Умное распределение (дешёвые для простых задач, дорогие для сложных)
- `openrouter_specialized` - Специализированные модели под каждую задачу

**Google Gemini:**
- `google_free` - Gemini 1.5 Flash (60 req/min, бесплатно)
- `google_balanced` - Gemini 1.5 Pro (лучшее качество)
- `google_hybrid` - Flash для простых задач, Pro для сложных
- `google_experimental` - Gemini 2.0 Flash Experimental (новейшие фичи)

**Groq (самый быстрый!):**
- `groq_free` - Llama 3.3 70B (30 req/min, очень быстро!)
- `groq_speed` - Только Llama 3.1 8B (максимальная скорость)
- `groq_quality` - Только Llama 3.3 70B (максимальное качество)
- `groq_hybrid` - 8B для простых задач, 70B для сложных
- `groq_mixtral` - Mixtral 8x7B (большой контекст 32k tokens)

**Ollama (локальные модели):**
- `ollama_lightweight` - 7-14B модели (работает на CPU)
- `ollama_balanced` - 14-32B модели (требует 16GB+ RAM)
- `ollama_quality` - 70B модели (требует GPU, 48GB+ RAM)
- `ollama_multilingual` - Qwen модели (отличная поддержка русского)
- `ollama_specialized` - Разные модели под разные задачи

**Auto Fallback (рекомендуется!):**
- `auto_aggressive` - Groq → Google → OpenRouter → Ollama (приоритет скорости)
- `auto_quality` - Google Pro → OpenRouter Premium → Groq (приоритет качества)
- `auto_free` - Только бесплатные ресурсы
- `auto_offline` - Приоритет локальным моделям (приватность)

---

## 📦 Установка и настройка

### 1. Скопируйте новые файлы в проект

```bash
# Конфигурации моделей
cp -r config/models/* /path/to/your/project/config/

# Новые агенты
cp agents/telegram_formatter_agent.py /path/to/your/project/src/application/ai_services/agents/
cp agents/seo_optimizer_agent.py /path/to/your/project/src/application/ai_services/agents/

# Обновлённый __init__.py
cp agents/__init__.py /path/to/your/project/src/application/ai_services/agents/
```

### 2. Настройте environment variables

```bash
# API ключи провайдеров
export GROQ_API_KEY=gsk_...
export GOOGLE_API_KEY=AI...
export OPENROUTER_API_KEY=sk-or-...
export HUGGINGFACEHUB_API_TOKEN=hf_...

# Выберите профиль
export LLM_PROFILE=auto_aggressive

# Для локальных моделей
export OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Установите зависимости (если нужно)

Новые агенты используют те же зависимости, что и существующие:
```bash
pip install pydantic>=2.0.0
```

---

## 🚀 Примеры использования

### Полный pipeline с новыми агентами

```python
from src.application.ai_services.agents import (
    ClassifierAgent,
    RelevanceAgent,
    SummarizerAgent,
    RewriterAgent,
    StyleNormalizerAgent,
    QualityValidatorAgent,
    TelegramFormatterAgent,
    SEOOptimizerAgent
)

# Инициализация агентов
classifier = ClassifierAgent()
relevance = RelevanceAgent()
summarizer = SummarizerAgent()
rewriter = RewriterAgent()
normalizer = StyleNormalizerAgent()
validator = QualityValidatorAgent()
telegram = TelegramFormatterAgent(default_author="TechNews")
seo = SEOOptimizerAgent()

# Обработка статьи
def process_article(title, content, tags):
    # 1. Классификация
    is_news = classifier.classify(title, content)
    print(f"Type: {'NEWS' if is_news else 'ARTICLE'}")
    
    # 2. Оценка релевантности
    score, reason = relevance.score(title, content, tags)
    print(f"Relevance: {score}/10 - {reason}")
    
    if score < 5:
        print("❌ Low relevance, skipping")
        return None
    
    # 3. Создание summary
    summary = summarizer.summarize(title, content)
    print(f"Summary: {summary[:100]}...")
    
    # 4. Улучшение заголовка
    improved_title = rewriter.rewrite_title(title, content)
    print(f"Title: {title} → {improved_title}")
    
    # 5. Нормализация стиля
    normalized = normalizer.normalize_full_text(content)
    
    # 6. Валидация
    validation = validator.validate(content, normalized)
    print(f"Validation: score={validation.score:.2f}")
    
    # 7. SEO оптимизация
    seo_data = seo.optimize(
        title=improved_title,
        content=normalized,
        target_keywords=tags
    )
    print(f"SEO: {seo_data.title_tag}")
    print(f"Keywords: {', '.join(seo_data.keywords)}")
    
    # 8. Telegram форматирование
    telegram_post = telegram.format_for_telegram(
        title=improved_title,
        content=normalized,
        source_url=f"https://example.com/articles/{seo_data.slug}",
        tags=tags
    )
    
    return {
        'is_news': is_news,
        'relevance_score': score,
        'summary': summary,
        'improved_title': improved_title,
        'normalized_content': normalized,
        'seo': seo_data,
        'telegram_post': telegram_post
    }

# Использование
article = process_article(
    title="Как я написал свой фреймворк за неделю",
    content="Длинный текст статьи...",
    tags=["python", "framework", "web"]
)

if article:
    # Публикация в Telegram
    if article['telegram_post'].telegraph_needed:
        telegraph_url = publish_to_telegraph(
            title=article['improved_title'],
            content_html=article['telegram_post'].telegraph_content
        )
        telegram_text = article['telegram_post'].text.replace(
            "{TELEGRAPH_URL}", telegraph_url
        )
    else:
        telegram_text = article['telegram_post'].text
    
    send_to_telegram(telegram_text, parse_mode="HTML")
    
    # Сохранение в БД с SEO данными
    save_article(
        title=article['improved_title'],
        content=article['normalized_content'],
        summary=article['summary'],
        slug=article['seo'].slug,
        meta_title=article['seo'].title_tag,
        meta_description=article['seo'].meta_description,
        keywords=article['seo'].keywords
    )
```

### Переключение профилей в runtime

```python
from src.config.models_config import ModelsConfig, set_models_config

# Способ 1: Через environment variable
import os
os.environ['LLM_PROFILE'] = 'groq_free'

# Способ 2: Программно
config = ModelsConfig(profile='openrouter_premium')
set_models_config(config)

# Теперь все новые агенты используют этот профиль
agent = ClassifierAgent()
print(agent.model)  # Покажет модель из профиля
```

### Использование разных профилей для разных агентов

```python
from src.infrastructure.ai.llm_provider import LLMProviderFactory
from src.config.models_config import ModelsConfig

# Быстрые агенты - Groq
groq_config = ModelsConfig(profile='groq_speed')
groq_llm = LLMProviderFactory.create(
    groq_config.get_llm_config('classifier')
)
classifier = ClassifierAgent(llm_provider=groq_llm)

# Качественные агенты - OpenRouter Premium
premium_config = ModelsConfig(profile='openrouter_premium')
premium_llm = LLMProviderFactory.create(
    premium_config.get_llm_config('summarizer')
)
summarizer = SummarizerAgent(llm_provider=premium_llm)
```

---

## 🔧 Кастомизация

### Создание своего профиля

Создайте файл `config/models/custom.yaml`:

```yaml
provider: openrouter

profiles:
  my_custom_profile:
    provider: openrouter
    description: "Мой кастомный профиль"
    cost_tier: medium
    
    agents:
      classifier:
        model: "openai/gpt-4o-mini"
        temperature: 0.3
        max_tokens: 100
      
      telegram_formatter:
        model: "anthropic/claude-3.5-haiku"
        temperature: 0.4
        max_tokens: 600
      
      # ... другие агенты
```

Добавьте импорт в главный `models.yaml`:

```yaml
imports:
  - base.yaml
  - openrouter.yaml
  - google.yaml
  - groq.yaml
  - ollama.yaml
  - auto_fallback.yaml
  - custom.yaml  # ← добавьте эту строку
```

Использование:

```bash
export LLM_PROFILE=my_custom_profile
```

### Настройка Telegram форматирования

```python
# Кастомный автор и CTA
telegram_agent = TelegramFormatterAgent(
    default_author="MyTechBlog",
    add_source_link=True  # Добавлять ли ссылку на источник
)

# Оценка какой пост будет создан (без генерации)
post_type = telegram_agent.estimate_post_type(len(content))
# Возвращает: "full_post", "condensed_post", или "preview_with_telegraph"
```

### Настройка SEO оптимизации

```python
# SEO с анализом структуры
seo = seo_agent.optimize(
    title="Article Title",
    content="Content...",
    target_keywords=["python", "ai"],
    analyze_structure=True  # Анализирует H1-H6, абзацы, списки
)

# Валидация SEO
validation = seo_agent.validate_seo(seo)
print(f"Score: {validation['score']}/100")
print(f"Issues: {validation['issues']}")
print(f"Warnings: {validation['warnings']}")

# Генерация alt-текстов для изображений
alt_texts = seo_agent.generate_alt_texts(
    image_context=content,
    num_images=3
)
```

---

## 📊 Мониторинг и метрики

### Получение метрик агентов

```python
# Метрики отдельного агента
metrics = classifier.get_metrics()
print(metrics)
# {
#     'agent': 'classifier',
#     'model': 'llama-3.3-70b-versatile',
#     'provider': 'groq',
#     'total_calls': 150,
#     'successful_calls': 148,
#     'failed_calls': 2,
#     'success_rate': '98.67%',
#     'avg_latency_ms': '342'
# }

# Сброс метрик
classifier.reset_metrics()
```

### Rate Limit Handling

Система автоматически обрабатывает rate limits:

```python
# Auto-fallback профиль автоматически переключится
# на следующего провайдера при 429 ошибке
config = ModelsConfig(profile='auto_aggressive')

# Настройка retry стратегии
agent = ClassifierAgent(
    config=config,
    max_retries=5,           # Максимум попыток
    retry_delay=2.0          # Задержка между попытками
)
```

---

## 🔄 Миграция с v1.0

### Автоматическая совместимость

Старые профили автоматически маппятся:

| Старый профиль | Новый профиль |
|----------------|---------------|
| `auto_fallback` | `auto_aggressive` |
| `balanced` | `ollama_balanced` |
| `fast` | `groq_speed` |
| `free_openrouter` | `openrouter_free` |
| `cloud_balanced` | `openrouter_balanced` |
| `cloud_quality` | `openrouter_premium` |

### Обновление кода

**Было:**
```python
from src.config.models_config import get_models_config

config = get_models_config()
# Использовал профиль из models.yaml: active_profile
```

**Стало (рекомендуется):**
```python
from src.config.models_config import ModelsConfig

# Явное указание профиля
config = ModelsConfig(profile='auto_aggressive')

# Или через environment
# export LLM_PROFILE=auto_aggressive
config = ModelsConfig()  # Автоматически использует LLM_PROFILE
```

**Новые агенты - просто импортируйте:**
```python
from src.application.ai_services.agents import (
    TelegramFormatterAgent,  # Новый!
    SEOOptimizerAgent        # Новый!
)
```

---

## 🎓 Best Practices

### 1. Выбор профиля

**Разработка:**
- `groq_free` - быстро и бесплатно
- Альтернатива: `google_free` (больше лимит)

**Production:**
- `auto_aggressive` - высокая доступность
- Альтернатива: `auto_quality` - качество важнее

**Экономия:**
- `auto_free` - только бесплатное
- Альтернатива: `groq_free` - один провайдер

**Приватность:**
- `ollama_balanced` - полностью локально
- Альтернатива: `auto_offline` - локально с fallback

### 2. Telegram публикация

```python
# ✅ Правильно - проверяем telegraph_needed
if post.telegraph_needed:
    telegraph_url = publish_to_telegraph(...)
    final_text = post.text.replace("{TELEGRAPH_URL}", telegraph_url)
else:
    final_text = post.text

# ❌ Неправильно - не проверяем, может быть placeholder
send_to_telegram(post.text)  # Может отправить "{TELEGRAPH_URL}"
```

### 3. SEO оптимизация

```python
# ✅ Правильно - валидируем перед использованием
seo = seo_agent.optimize(title, content, target_keywords)
validation = seo_agent.validate_seo(seo)

if validation['valid']:
    save_to_db(seo)
else:
    logger.warning(f"SEO issues: {validation['issues']}")
    # Используем fallback или исправляем

# ❌ Неправильно - слепо доверяем результату
seo = seo_agent.optimize(title, content)
save_to_db(seo)  # Может быть невалидным
```

### 4. Error Handling

```python
# ✅ Правильно - обрабатываем исключения
try:
    post = telegram_agent.format_for_telegram(title, content)
except Exception as e:
    logger.error(f"Telegram formatting failed: {e}")
    # Fallback: простое форматирование или пропуск
    post = create_simple_telegram_post(title, content)

# ❌ Неправильно - не обрабатываем
post = telegram_agent.format_for_telegram(title, content)
# Может упасть весь pipeline
```

---

## 🐛 Troubleshooting

### Проблема: Rate limit errors при использовании Groq

**Решение:**
```bash
# Используйте auto-fallback профиль
export LLM_PROFILE=auto_aggressive

# Или уменьшите нагрузку
export LLM_PROFILE=groq_hybrid  # 8B для простых задач
```

### Проблема: Telegram пост всё равно слишком длинный

**Решение:**
```python
# Проверьте фактическую длину
if len(post.text) > 4096:
    logger.error("Post exceeds Telegram limit!")
    # Принудительное сокращение
    post.text = post.text[:4000] + "..."

# Или используйте более агрессивную стратегию
agent = TelegramFormatterAgent()
# Принудительно создать preview даже для средних статей
if len(content) > 3000:
    post = agent._create_preview_post(...)
```

### Проблема: SEO метаданные невалидны

**Решение:**
```python
# Используйте более мощную модель
config = ModelsConfig(profile='openrouter_premium')
llm = LLMProviderFactory.create(config.get_llm_config('seo_optimizer'))
seo_agent = SEOOptimizerAgent(llm_provider=llm)

# Или исправьте вручную
seo = seo_agent.optimize(title, content)
validation = seo_agent.validate_seo(seo)

if not validation['valid']:
    # Ручное исправление
    if len(seo.title_tag) < 30:
        seo.title_tag += " - Complete Guide"
```

### Проблема: Ollama модели медленные

**Решение:**
```bash
# Используйте более лёгкие модели
export LLM_PROFILE=ollama_lightweight

# Или используйте облачные провайдеры
export LLM_PROFILE=auto_aggressive
```

---

## 📞 Support

Если возникли вопросы или проблемы:

1. Проверьте примеры в этом README
2. Посмотрите логи агентов (они логируют все действия)
3. Попробуйте другой профиль
4. Проверьте API ключи и лимиты

---

## 📄 Changelog

### v2.0.0 (2026-01-27)
- ✨ Добавлен TelegramFormatterAgent для публикации в Telegram
- ✨ Добавлен SEOOptimizerAgent для SEO оптимизации
- 🔧 Разделение конфигураций на отдельные файлы по провайдерам
- 🚀 20+ новых профилей для разных сценариев
- 📚 Comprehensive документация и примеры
- 🐛 Улучшенная обработка ошибок и fallback
- ⚡ Оптимизация для разных сценариев использования

### v1.0.0 (Предыдущая версия)
- Базовые агенты: Classifier, Relevance, Summarizer, Rewriter, StyleNormalizer, QualityValidator
- Монолитная конфигурация в одном файле
- Поддержка Groq, Google, OpenRouter, Ollama

---

## 🚀 Быстрый старт - Шпаргалка

```bash
# 1. Установка
cp -r config agents /path/to/project/

# 2. Настройка
export LLM_PROFILE=auto_aggressive
export GROQ_API_KEY=your_key
export GOOGLE_API_KEY=your_key

# 3. Код
from src.application.ai_services.agents import (
    ClassifierAgent, TelegramFormatterAgent, SEOOptimizerAgent
)

classifier = ClassifierAgent()
telegram = TelegramFormatterAgent()
seo = SEOOptimizerAgent()

# Готово! 🎉
```

---

**Version:** 2.0.0  
**Date:** 2026-01-27  
**Author:** AI System Architect  
**License:** Use freely in your project
