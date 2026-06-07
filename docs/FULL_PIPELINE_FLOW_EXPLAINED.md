# 🔄 Полная документация пайплайна обработки статей

## Общая архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run_full_pipeline_fixed.py                          │
│                           (Точка входа)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ФАЗА 1: ПАРСИНГ                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  HabrScraperService._scrape_articles()                                  ││
│  │  → Скачивает статьи с Habr                                              ││
│  │  → Возвращает: [{title, content, url, author, tags, hubs}, ...]         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ФАЗА 2: ВАЛИДАЦИЯ БД                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ArticleRepositoryImpl.get_existing_urls()                              ││
│  │  → Проверяет какие URL уже есть в PostgreSQL                            ││
│  │  → Фильтрует дубликаты                                                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ФАЗА 3: AI ОБРАБОТКА (для каждой новой статьи)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        AIOrchestrator                                   ││
│  │  ┌───────────────────────────────────────────────────────────────────┐  ││
│  │  │ 1️⃣ ClassifierAgent     → Новость или Статья?                      │  ││
│  │  │ 2️⃣ RelevanceAgent      → Оценка 0-10                              │  ││
│  │  │ 3️⃣ SummarizerAgent     → Тизер (краткое описание)                 │  ││
│  │  │ 4️⃣ RewriterAgent       → Улучшенный заголовок                     │  ││
│  │  │ 5️⃣ StyleNormalizerAgent → Переписанный текст                      │  ││
│  │  │ 6️⃣ QualityValidatorAgent → Проверка качества                      │  ││
│  │  └───────────────────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ФАЗА 4: СОХРАНЕНИЕ                                                         │
│  ┌────────────────────────┐    ┌────────────────────────────────────────┐  │
│  │  PostgreSQL            │    │  Qdrant (если score >= min_relevance)  │  │
│  │  → Все статьи          │    │  → Только релевантные для поиска       │  │
│  └────────────────────────┘    └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Структура файлов

```
news-aggregator-pro/
├── run_full_pipeline_fixed.py          # Точка входа
├── config/
│   └── models.yaml                     # Конфигурация моделей
├── src/
│   ├── config/
│   │   └── models_config.py            # Загрузка конфигурации
│   ├── application/
│   │   └── ai_services/
│   │       ├── orchestrator.py         # Координатор агентов
│   │       └── agents/
│   │           ├── base_agent.py       # Базовый класс агента
│   │           ├── classifier_agent.py # Классификация
│   │           ├── relevance_agent.py  # Релевантность
│   │           ├── summarizer_agent.py # Тизеры
│   │           ├── rewriter_agent.py   # Заголовки
│   │           ├── style_normalizer_agent.py  # Нормализация
│   │           └── quality_validator_agent.py # Валидация
│   ├── infrastructure/
│   │   └── ai/
│   │       ├── llm_provider.py         # Абстракция LLM
│   │       └── qdrant_client.py        # Векторная БД
│   └── scrapers/
│       └── habr/
│           └── scraper_service.py      # Парсер Habr
```

---

## 🚀 Детальный разбор: run_full_pipeline_fixed.py

### Строки 27-35: Импорты

```python
from src.scrapers.habr.scraper_service import HabrScraperService     # Парсер
from src.application.ai_services.orchestrator import AIOrchestrator  # Оркестратор AI
from src.infrastructure.ai.qdrant_client import QdrantService        # Векторный поиск
from src.infrastructure.config.database import AsyncSessionLocal     # БД сессия
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl  # Репозиторий
from src.application.commands.create_article_command import CreateArticleCommand  # Команда создания
from src.domain.value_objects.source_type import SourceType          # Тип источника
from src.domain.entities.article import Article                      # Доменная сущность
from src.config.models_config import get_models_config               # Конфиг моделей
```

### Строки 64-77: Создание объекта Article

```python
def create_article_from_data(data: Dict[str, Any]) -> Article:
    """Создать объект Article из словаря данных парсера."""
    article = Article(
        id=uuid.uuid4(),                           # Генерируем UUID
        title=data.get('title', ''),               # Заголовок
        content=data.get('content', ''),           # Полный текст
        url=data.get('url', ''),                   # URL статьи
        source=SourceType.HABR,                    # Источник (Habr)
        author=data.get('author'),                 # Автор
        published_at=data.get('published_at'),     # Дата публикации
        tags=data.get('tags', []),                 # Теги
        hubs=data.get('hubs', [])                  # Хабы
    )
    return article
```

### Строки 151-165: Инициализация сервисов

```python
# 1. Создаём парсер Habr
scraper = HabrScraperService()

# 2. Создаём AI оркестратор
#    Внутри инициализируются все 6 агентов
orchestrator = AIOrchestrator()

# 3. Создаём клиент Qdrant для векторного поиска
qdrant = QdrantService()
```

### Строки 204-218: ФАЗА 1 - Парсинг

```python
# Парсим статьи с Habr
# hubs_list = ["machine-learning", "ai"] - фильтр по хабам
# parse_limit = limit * 3 - парсим с запасом (часть может быть в БД)

articles_data = await scraper._scrape_articles(parse_limit, hubs_list)

# articles_data = [
#     {
#         'title': 'Как я обучил нейросеть...',
#         'content': 'Полный текст статьи...',
#         'url': 'https://habr.com/ru/articles/123456/',
#         'author': 'username',
#         'published_at': datetime(2026, 1, 24),
#         'tags': ['python', 'ml'],
#         'hubs': ['machine-learning', 'python']
#     },
#     ...
# ]
```

### Строки 220-236: ФАЗА 2 - Проверка дубликатов

```python
async with AsyncSessionLocal() as session:
    repo = ArticleRepositoryImpl(session)
    
    # Собираем все URL из спарсенных статей
    urls = [d['url'] for d in articles_data]
    
    # Проверяем какие уже есть в БД
    existing = await repo.get_existing_urls(urls)
    # existing = {'https://habr.com/ru/articles/111/', 'https://habr.com/ru/articles/222/'}
    
    # Оставляем только новые статьи (не в БД), берём первые {limit}
    new_articles_data = [d for d in articles_data if d['url'] not in existing][:limit]
```

### Строки 253-311: ФАЗА 3 - AI Обработка (главный цикл)

```python
for i, data in enumerate(new_articles_data, 1):
    # 1. Создаём объект Article из сырых данных
    article = create_article_from_data(data)
    
    # 2. Передаём в оркестратор для AI обработки
    #    Внутри происходит вся магия (6 шагов)
    processed_article = orchestrator.process_article(
        article=article,
        verbose=verbose,
        min_relevance=min_relevance
    )
    
    # 3. Сохраняем в PostgreSQL
    command = CreateArticleCommand(
        title=processed_article.title,
        content=processed_article.content,
        url=processed_article.url,
        ai_summary=processed_article.editorial_teaser,    # Тизер от AI
        relevance_score=score,                            # Оценка 0-10
        relevance_reason=processed_article.relevance_reason  # Почему такая оценка
    )
    db_article = await repo.create(command)
    
    # 4. Если релевантность >= порога, добавляем в Qdrant для поиска
    if score >= min_relevance:
        qdrant.add_article(str(db_article.id), db_article.title, db_article.content)
```

---

## 🤖 Детальный разбор: AIOrchestrator

**Файл:** `src/application/ai_services/orchestrator.py`

### Строки 74-109: Инициализация оркестратора

```python
class AIOrchestrator:
    def __init__(
        self,
        provider: Optional[str] = None,      # "ollama" или "openrouter"
        api_key: Optional[str] = None,       # Для OpenRouter
        config_path: str = "config/models.yaml",
        enable_validation: bool = True,       # Включить QualityValidator
        max_retries: int = 2                  # Повторы при ошибке валидации
    ):
        # Загружаем конфиг из models.yaml
        self.config = ModelsConfig(config_path)
        
        # Логируем что используем
        # Например: "AIOrchestrator: profile=free_openrouter, provider=openrouter"
        profile = self.config.get_profile()
        logger.info(f"AIOrchestrator: profile={profile.name}, provider={profile.provider.value}")
        
        # Инициализируем все 6 агентов
        self._init_agents()
```

### Строки 111-122: Создание агентов

```python
def _init_agents(self):
    """Инициализация всех AI агентов."""
    # Каждый агент получает конфиг и создаёт свой LLM провайдер
    self.classifier = ClassifierAgent(config=self.config)      # 1️⃣
    self.relevance = RelevanceAgent(config=self.config)        # 2️⃣
    self.summarizer = SummarizerAgent(config=self.config)      # 3️⃣
    self.rewriter = RewriterAgent(config=self.config)          # 4️⃣
    self.style_normalizer = StyleNormalizerAgent(config=self.config)  # 5️⃣
    
    if self.enable_validation:
        self.validator = QualityValidatorAgent(config=self.config)    # 6️⃣
```

### Строки 124-190: Главный метод process_article()

```python
def process_article(self, article, normalize_style=True, validate_quality=True, min_relevance=5):
    """Полная AI обработка статьи."""
    
    # ═══════════════════════════════════════════════════════════════════
    # ШАГ 1️⃣: КЛАССИФИКАЦИЯ
    # Определяем: это новость или аналитическая статья?
    # ═══════════════════════════════════════════════════════════════════
    result = self.classifier.classify_with_details(article.title, article.content)
    article.is_news = result.is_news  # True = новость, False = статья
    # Логирует: "Классификация: НОВОСТЬ" или "Классификация: СТАТЬЯ"
    
    # ═══════════════════════════════════════════════════════════════════
    # ШАГ 2️⃣: ОЦЕНКА РЕЛЕВАНТНОСТИ
    # Насколько статья интересна для целевой аудитории (0-10)
    # ═══════════════════════════════════════════════════════════════════
    result = self.relevance.score_with_details(article.title, article.content, article.tags)
    article.set_relevance(result.score, result.reason)
    # result.score = 8
    # result.reason = "Статья о практическом применении ML, актуальная тема"
    # Логирует: "Релевантность: 8/10"
    
    # ═══════════════════════════════════════════════════════════════════
    # ШАГ 3️⃣: СОЗДАНИЕ ТИЗЕРА
    # Краткое описание для превью (2-3 предложения)
    # ═══════════════════════════════════════════════════════════════════
    result = self.summarizer.summarize_with_details(article.title, article.content)
    article.editorial_teaser = result.teaser
    # result.teaser = "Разработчик показал как обучить GPT на своих данных 
    #                  за $50. Подробный гайд с кодом и примерами."
    
    # ═══════════════════════════════════════════════════════════════════
    # ШАГ 4️⃣: УЛУЧШЕНИЕ ЗАГОЛОВКА
    # Более привлекательный и информативный заголовок
    # ═══════════════════════════════════════════════════════════════════
    result = self.rewriter.rewrite_with_details(article.title, article.content)
    article.editorial_title = result.improved_title
    # Было:    "Мой опыт с ML"
    # Стало:   "Как я обучил GPT на домашнем компьютере за выходные"
    
    # ═══════════════════════════════════════════════════════════════════
    # ШАГ 5️⃣: НОРМАЛИЗАЦИЯ СТИЛЯ
    # Переписываем текст в едином редакционном стиле
    # ═══════════════════════════════════════════════════════════════════
    if normalize_style and article.content:
        normalized = self._normalize_with_retry(article.content, validate_quality, metrics)
        article.editorial_rewritten = normalized
    
    # ═══════════════════════════════════════════════════════════════════
    # ШАГ 6️⃣: ВАЛИДАЦИЯ КАЧЕСТВА (внутри _normalize_with_retry)
    # Проверяем что переписанный текст не потерял смысл
    # ═══════════════════════════════════════════════════════════════════
    # validation = self.validator.validate(original, normalized)
    # validation.is_valid = True/False
    # validation.score = 0.85
    # validation.issues = ["Потеряны ключевые факты"]
    
    return article  # Возвращаем обогащённую статью
```

---

## 🧠 Детальный разбор каждого агента

### 1️⃣ ClassifierAgent (classifier_agent.py)

**Задача:** Определить тип контента — новость или аналитическая статья.

**Промпт (упрощённо):**
```
Classify this content:
Title: {title}
Content: {content[:2000]}

Is this a NEWS article (short, factual, recent event) 
or an ARTICLE (analysis, tutorial, opinion)?

Respond with JSON:
{"is_news": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
```

**Результат:**
```python
ClassificationResult(
    is_news=False,
    confidence=0.85,
    reasoning="Это tutorial с пошаговыми инструкциями, не новость"
)
```

---

### 2️⃣ RelevanceAgent (relevance_agent.py)

**Задача:** Оценить насколько статья интересна целевой аудитории.

**Критерии оценки:**
- Актуальность темы
- Практическая ценность
- Глубина материала
- Целевая аудитория (AI/ML специалисты)

**Промпт (упрощённо):**
```
Rate this article's relevance for AI/ML professionals (0-10):
Title: {title}
Tags: {tags}
Content preview: {content[:3000]}

Consider:
- Is it about AI, ML, Data Science?
- Is it practical or theoretical?
- Is it for beginners or experts?

Respond with JSON:
{"score": 8, "reason": "Practical ML tutorial with code examples"}
```

**Результат:**
```python
RelevanceResult(
    score=8,
    reason="Практический туториал по fine-tuning LLM с примерами кода"
)
```

---

### 3️⃣ SummarizerAgent (summarizer_agent.py)

**Задача:** Создать краткий тизер для превью статьи.

**Требования к тизеру:**
- 2-3 предложения
- Захватывает внимание
- Передаёт суть статьи
- Без спойлеров

**Промпт (упрощённо):**
```
Create an engaging teaser (2-3 sentences) for this article:
Title: {title}
Content: {content[:4000]}

The teaser should:
- Hook the reader
- Summarize the main value
- Not give away everything

Respond with JSON:
{"teaser": "...", "key_points": ["point1", "point2"]}
```

**Результат:**
```python
SummaryResult(
    teaser="Разработчик из Google показал как дообучить Llama 3 на своих 
            данных за $50 используя LoRA. Пошаговый гайд с кодом и 
            сравнением результатов.",
    key_points=["LoRA fine-tuning", "Llama 3", "Дёшево и эффективно"]
)
```

---

### 4️⃣ RewriterAgent (rewriter_agent.py)

**Задача:** Улучшить заголовок — сделать его более привлекательным.

**Требования:**
- Кликбейт, но не жёлтый
- Информативный
- SEO-оптимизированный
- Сохраняет суть

**Промпт (упрощённо):**
```
Improve this article title:
Original: {title}
Content preview: {content[:2000]}

Requirements:
- More engaging and specific
- Include key benefit or insight
- Keep it under 80 characters
- No clickbait lies

Respond with JSON:
{"improved_title": "...", "changes_made": ["..."]}
```

**Результат:**
```python
RewriteResult(
    original_title="Мой опыт с ML",
    improved_title="Как я обучил GPT на домашнем компьютере за $50 и выходные",
    changes_made=["Добавлена конкретика", "Указана стоимость", "Указано время"]
)
```

---

### 5️⃣ StyleNormalizerAgent (style_normalizer_agent.py)

**Задача:** Переписать текст в едином редакционном стиле.

**Что делает:**
- Убирает разговорный стиль
- Приводит к единому формату
- Сохраняет всю техническую информацию
- Улучшает структуру

**Промпт (упрощённо):**
```
Rewrite this technical article in a professional editorial style:

{content}

Requirements:
- Keep all technical details and code
- Use professional tone
- Improve structure and flow
- Remove casual language and filler words
- Preserve all facts and examples

Return the rewritten text.
```

**Результат:**
```python
NormalizationResult(
    normalized_text="[Переписанный текст в профессиональном стиле]",
    changes_made=["Убран разговорный тон", "Улучшена структура", "Добавлены подзаголовки"]
)
```

---

### 6️⃣ QualityValidatorAgent (quality_validator_agent.py)

**Задача:** Проверить что переписанный текст не потерял смысл.

**Что проверяет:**
- Сохранены ли все факты?
- Не изменился ли смысл?
- Остался ли код корректным?
- Нет ли галлюцинаций?

**Промпт (упрощённо):**
```
Compare original and rewritten text. Check for:
1. Lost facts or details
2. Changed meaning
3. Added false information
4. Code correctness

Original: {original}
Rewritten: {rewritten}

Respond with JSON:
{"is_valid": true/false, "score": 0.0-1.0, "issues": ["..."]}
```

**Результат:**
```python
ValidationResult(
    is_valid=True,
    score=0.92,
    issues=[]  # Или ["Потеряна ссылка на GitHub репозиторий"]
)
```

---

## ⚙️ Конфигурация: models.yaml

```yaml
# Активный профиль по умолчанию
active_profile: free_openrouter

profiles:
  # Бесплатный OpenRouter
  free_openrouter:
    provider: openrouter
    agents:
      classifier:
        model: "z-ai/glm-4.5-air:free"
        temperature: 0.3      # Низкая = более детерминированный
        max_tokens: 100       # Короткий ответ
      relevance:
        model: "z-ai/glm-4.5-air:free"
        temperature: 0.4
        max_tokens: 300
      summarizer:
        model: "z-ai/glm-4.5-air:free"
        temperature: 0.5      # Средняя = баланс
        max_tokens: 300
      rewriter:
        model: "z-ai/glm-4.5-air:free"
        temperature: 0.6      # Выше = более креативный
        max_tokens: 200
      style_normalizer:
        model: "z-ai/glm-4.5-air:free"
        temperature: 0.3
        max_tokens: 8000      # Большой = для длинных статей
      quality_validator:
        model: "z-ai/glm-4.5-air:free"
        temperature: 0.2      # Очень низкая = строгий
        max_tokens: 500
```

---

## 🔌 Инфраструктура: LLM Provider

**Файл:** `src/infrastructure/ai/llm_provider.py`

### Иерархия классов:

```
LLMProvider (ABC)              # Абстрактный базовый класс
    │
    ├── OllamaProvider         # Локальный Ollama
    │   └── _create_client()   # → ChatOllama
    │
    └── OpenRouterProvider     # Облачный OpenRouter  
        └── _create_client()   # → ChatOpenAI (с кастомным base_url)
```

### LLMProviderFactory:

```python
class LLMProviderFactory:
    """Фабрика для создания провайдеров."""
    
    _providers = {
        LLMProviderType.OLLAMA: OllamaProvider,
        LLMProviderType.OPENROUTER: OpenRouterProvider,
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        provider_class = cls._providers[config.provider]
        return provider_class(config)
```

### OpenRouterProvider._create_client():

```python
def _create_client(self) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    
    model = self._resolve_model(self.config.model)
    
    client_kwargs = {
        "model": model,
        "openai_api_key": self.config.api_key,
        "openai_api_base": "https://openrouter.ai/api/v1",  # Кастомный URL!
        "temperature": self.config.temperature,
        "max_tokens": self.config.max_tokens,
        "timeout": self.config.timeout,
        "default_headers": {
            "HTTP-Referer": "https://news-aggregator-pro.local",
            "X-Title": "News Aggregator Pro"
        }
    }
    
    # Специальная обработка для GLM моделей (отключаем thinking mode)
    if "glm" in model.lower():
        client_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    
    return ChatOpenAI(**client_kwargs)
```

---

## 📊 Поток данных (Data Flow)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ВХОД                                            │
│  python run_full_pipeline_fixed.py 3 "machine-learning,ai" -v               │
│                                                                              │
│  limit=3           → Обработать 3 статьи                                     │
│  hubs="ml,ai"      → Фильтр по хабам                                        │
│  --verbose         → Подробный вывод                                        │
│  --min-relevance=5 → Порог для Qdrant                                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HabrScraperService._scrape_articles(9, ["machine-learning", "ai"])         │
│                                                                              │
│  Возвращает 9 статей (limit * 3 = запас на дубликаты):                      │
│  [                                                                           │
│    {title: "...", content: "...", url: "habr.com/123", tags: [...], ...},   │
│    {title: "...", content: "...", url: "habr.com/456", tags: [...], ...},   │
│    ...                                                                       │
│  ]                                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ArticleRepositoryImpl.get_existing_urls([url1, url2, ...])                 │
│                                                                              │
│  SQL: SELECT url FROM articles WHERE url IN (...)                           │
│  Возвращает: {"habr.com/111", "habr.com/222"}  ← Уже в БД                   │
│                                                                              │
│  Фильтруем: 9 спарсено - 2 в БД = 7 новых → берём первые 3                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Для каждой из 3 новых статей:                                              │
│                                                                              │
│  article = create_article_from_data(data)                                   │
│  Article(                                                                    │
│    id=UUID("9bd3deb6-..."),                                                 │
│    title="Как обучить нейросеть...",                                        │
│    content="Полный текст 50KB...",                                          │
│    url="habr.com/789",                                                      │
│    source=SourceType.HABR,                                                  │
│    tags=["python", "ml"],                                                   │
│    hubs=["machine-learning"]                                                │
│  )                                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  orchestrator.process_article(article)                                       │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1️⃣ classifier.classify_with_details(title, content)                    │ │
│  │    → HTTP POST https://openrouter.ai/api/v1/chat/completions           │ │
│  │    → Response: {"is_news": false, "confidence": 0.85}                  │ │
│  │    → article.is_news = False                                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 2️⃣ relevance.score_with_details(title, content, tags)                  │ │
│  │    → HTTP POST https://openrouter.ai/api/v1/chat/completions           │ │
│  │    → Response: {"score": 8, "reason": "Практический ML туториал"}      │ │
│  │    → article.relevance_score = 8                                       │ │
│  │    → article.relevance_reason = "Практический ML туториал"             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 3️⃣ summarizer.summarize_with_details(title, content)                   │ │
│  │    → HTTP POST ...                                                     │ │
│  │    → article.editorial_teaser = "Разработчик показал как..."           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 4️⃣ rewriter.rewrite_with_details(title, content)                       │ │
│  │    → HTTP POST ...                                                     │ │
│  │    → article.editorial_title = "Как обучить GPT за $50..."             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 5️⃣ style_normalizer.normalize_with_details(content)                    │ │
│  │    → HTTP POST ... (самый длинный запрос, весь текст)                  │ │
│  │    → article.editorial_rewritten = "[Переписанный текст]"              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 6️⃣ validator.validate(original, rewritten)                             │ │
│  │    → HTTP POST ...                                                     │ │
│  │    → ValidationResult(is_valid=True, score=0.92)                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Возвращает обогащённую статью:                                             │
│  Article(                                                                    │
│    ...оригинальные поля...,                                                 │
│    is_news=False,                        ← от classifier                    │
│    relevance_score=8,                    ← от relevance                     │
│    relevance_reason="...",               ← от relevance                     │
│    editorial_teaser="...",               ← от summarizer                    │
│    editorial_title="...",                ← от rewriter                      │
│    editorial_rewritten="...",            ← от style_normalizer              │
│    embedding_status="processed"                                             │
│  )                                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Сохранение в PostgreSQL                                                     │
│                                                                              │
│  command = CreateArticleCommand(title, content, url, ai_summary, score, ...)│
│  db_article = await repo.create(command)                                    │
│  await session.commit()                                                     │
│                                                                              │
│  SQL: INSERT INTO articles (id, title, content, url, ai_summary,            │
│        relevance_score, ...) VALUES (...)                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Сохранение в Qdrant (только если score >= min_relevance)                   │
│                                                                              │
│  if score >= 5:  # min_relevance=5                                          │
│      qdrant.add_article(                                                    │
│          id=str(db_article.id),                                             │
│          title=db_article.title,                                            │
│          content=db_article.content                                         │
│      )                                                                       │
│                                                                              │
│  Внутри:                                                                    │
│  1. Генерация embedding через sentence-transformers                         │
│  2. Upsert в коллекцию "articles"                                          │
│  3. Теперь статья доступна для семантического поиска                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ВЫХОД                                           │
│                                                                              │
│  ================================================================================│
│  РЕЗУЛЬТАТЫ                                                                  │
│  ================================================================================│
│    Обработано:                                                           3  │
│    В БД:                                                                 3  │
│    В Qdrant:                                                             2  │
│    Низкая релевантность:                                                 1  │
│    Ошибок:                                                               0  │
│    Среднее время:                                                    15.23с │
│    Общее время:                                                      48.72с │
│    Статус:                                                        ✅ УСПЕХ  │
│  ================================================================================│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Резюме

| Компонент | Файл | Задача |
|-----------|------|--------|
| **Pipeline** | `run_full_pipeline_fixed.py` | Координация всех фаз |
| **Scraper** | `scraper_service.py` | Парсинг Habr |
| **Repository** | `article_repository_impl.py` | Работа с PostgreSQL |
| **Orchestrator** | `orchestrator.py` | Координация AI агентов |
| **ClassifierAgent** | `classifier_agent.py` | Новость или статья? |
| **RelevanceAgent** | `relevance_agent.py` | Оценка 0-10 |
| **SummarizerAgent** | `summarizer_agent.py` | Создание тизера |
| **RewriterAgent** | `rewriter_agent.py` | Улучшение заголовка |
| **StyleNormalizer** | `style_normalizer_agent.py` | Переписывание текста |
| **QualityValidator** | `quality_validator_agent.py` | Проверка качества |
| **LLMProvider** | `llm_provider.py` | Абстракция над LLM |
| **QdrantService** | `qdrant_client.py` | Векторный поиск |
| **ModelsConfig** | `models_config.py` | Загрузка `models.yaml` |