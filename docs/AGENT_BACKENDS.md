# Agent Backends: Legacy и LangChain

Проект поддерживает два взаимозаменяемых backend'а для AI-агентов.
Переключение — одной переменной окружения, код пайплайна не меняется.

```bash
# .env
AGENT_BACKEND=legacy      # по умолчанию: собственные HTTP-провайдеры
AGENT_BACKEND=langchain   # LangChain ChatModels
```

## Архитектура

```
AIOrchestrator
    │
    ├── agent_factory.create_agent("classifier")   ← выбирает backend
    │       │
    │       ├── legacy:    src/application/ai_services/agents/
    │       │              (BaseAgent → LLMProviderFactory → HTTP)
    │       │
    │       └── langchain: src/application/ai_services/langchain_agents/
    │                      (LangChainAgent → ChatOpenAI → OpenAI-совместимый API)
    │
    └── конфигурация общая: src/config/models_config.py (ModelsConfig)
```

Оба backend'а:
- читают одну и ту же конфигурацию (`config/models.yaml` + env);
- используют одни и те же промпты и Pydantic-схемы результатов;
- предоставляют одинаковый публичный API (`classify_with_details`,
  `score_with_details`, `summarize_with_details`, `rewrite_with_details`,
  `process`, `get_metrics`).

## LangChain backend

Установка:

```bash
pip install -r requirements-ai.txt   # langchain-core, langchain-openai
```

LangChain-версии есть у четырёх основных текстовых агентов:

| Агент | Legacy | LangChain |
|---|---|---|
| Классификатор | `ClassifierAgent` | `LCClassifierAgent` |
| Релевантность | `RelevanceAgent` | `LCRelevanceAgent` |
| Суммаризатор | `SummarizerAgent` | `LCSummarizerAgent` |
| Рерайтер заголовков | `RewriterAgent` | `LCRewriterAgent` |

Остальные агенты (форматтеры, image-агенты, НейроКот) пока работают только
на legacy backend — фабрика автоматически подставит legacy-версию, даже если
`AGENT_BACKEND=langchain`. Если LangChain не установлен, фабрика тоже тихо
откатывается на legacy и пишет предупреждение в лог.

Преимущества LangChain backend'а:
- нативный structured output (`with_structured_output`) через function
  calling / JSON mode провайдера — меньше ошибок парсинга;
- встроенные retries и таймауты SDK;
- лёгкая интеграция с экосистемой LangChain (LangSmith-трейсинг, цепочки).

Все провайдеры подключаются через их **OpenAI-совместимые endpoints**,
поэтому достаточно одной зависимости `langchain-openai`:

| Провайдер | Endpoint |
|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Ollama | `{OLLAMA_BASE_URL}/v1` |

## OpenRouter

OpenRouter — рекомендуемый облачный провайдер: десятки бесплатных моделей,
автоматическое обнаружение и fallback.

```bash
# .env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...        # https://openrouter.ai/keys

# Опционально:
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free   # закрепить модель
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1          # свой прокси
OPENROUTER_EXCLUDED_MODELS=nemotron,gpt-oss-120b          # исключить модели
```

Как это работает:
1. Список бесплатных моделей запрашивается с API OpenRouter (кэш 1 час).
2. Модели сортируются по размеру под тип задачи: HEAVY-агенты получают
   большие модели первыми, LIGHT-агенты — маленькие.
3. `OPENROUTER_MODEL` (если задан) всегда пробуется первым.
4. Модель с ошибкой 429/402/403 уходит в cooldown (экспоненциальный),
   запрос автоматически повторяется на следующей модели.
5. `OPENROUTER_EXCLUDED_MODELS` фильтрует слабые/сломанные модели
   по подстроке ID.

## Программное использование

```python
# Через фабрику (уважает AGENT_BACKEND)
from src.application.ai_services.agent_factory import create_agent

classifier = create_agent("classifier")
result = classifier.classify_with_details(title="...", content="...")

# Напрямую LangChain-агент
from src.application.ai_services.langchain_agents import LCSummarizerAgent

summarizer = LCSummarizerAgent()
summary = summarizer.summarize_with_details(title="...", content="...")

# Только LangChain ChatModel (для своих цепочек)
from src.application.ai_services.langchain_agents import build_chat_model
from src.config.models_config import get_models_config

llm = build_chat_model(get_models_config().get_llm_config("summarizer"))
response = llm.invoke("Привет!")
```
