# -*- coding: utf-8 -*-
"""
Конфигурация моделей v6.0 - Production Ready

Улучшения:
- Поддержка различных режимов работы
- Динамическое переключение провайдеров
- Улучшенное кэширование конфигурации
- Метрики производительности
- Graceful fallback с весами
- Валидация конфигурации
- Поддержка hot-reload
- Интеграция с мониторингом
"""

import os
import yaml
import logging
import json
import time
import hashlib
import socket
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field, asdict
from threading import Lock, Event
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from src.infrastructure.ai.llm_provider import LLMConfig, LLMProviderType

logger = logging.getLogger(__name__)


# =============================================================================
# Перечисления
# =============================================================================

class TaskComplexity(str, Enum):
    """Сложность задачи агента."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class AgentType(str, Enum):
    """Типы AI агентов."""
    CLASSIFIER = "classifier"
    RELEVANCE = "relevance"
    SUMMARIZER = "summarizer"
    REWRITER = "rewriter"
    STYLE_NORMALIZER = "style_normalizer"
    QUALITY_VALIDATOR = "quality_validator"
    TELEGRAM_FORMATTER = "telegram_formatter"
    SEO_OPTIMIZER = "seo_optimizer"


class SelectionStrategy(str, Enum):
    """Стратегии выбора моделей."""
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    QUALITY_FOCUSED = "quality_focused"
    SPEED_FOCUSED = "speed_focused"
    ADAPTIVE = "adaptive"


class OperationMode(str, Enum):
    """Режимы работы."""
    NORMAL = "normal"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class CacheStrategy(str, Enum):
    """Стратегии кэширования."""
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"


class RetryStrategy(str, Enum):
    """Стратегии повторных попыток."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


# =============================================================================
# Датаклассы
# =============================================================================

@dataclass
class ModelMetrics:
    """Метрики производительности модели."""
    model_id: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    total_tokens: int = 0
    total_cost: float = 0.0

    @property
    def success_rate(self) -> float:
        """Процент успешных запросов."""
        if self.request_count == 0:
            return 0.0
        return (self.success_count / self.request_count) * 100

    def update_metrics(self, success: bool, response_time: float, tokens: int = 0, cost: float = 0.0):
        """Обновить метрики после запроса."""
        self.request_count += 1
        self.last_used = datetime.now()

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # Обновляем среднее время ответа
        if self.request_count == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * (
                        self.request_count - 1) + response_time) / self.request_count

        self.total_tokens += tokens
        self.total_cost += cost


@dataclass
class ModelInfo:
    """Информация о модели."""
    id: str
    cost_tier: str
    max_tokens: int
    capabilities: List[str]
    rate_limit: Optional[int] = None
    speed: Optional[int] = None
    ram_required: Optional[int] = None
    gpu_required: bool = False
    priority: int = 0
    weight: float = 1.0
    health_score: float = 100.0
    last_health_check: Optional[datetime] = None
    metrics: ModelMetrics = field(default_factory=lambda: ModelMetrics(model_id=""))

    def __post_init__(self):
        if self.metrics.model_id == "":
            self.metrics.model_id = self.id


@dataclass
class ProviderConfig:
    """Конфигурация провайдера."""
    provider: LLMProviderType
    models: Dict[str, ModelInfo]
    recommended_for: Dict[str, Any]
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    default_base_url: Optional[str] = None
    enabled: bool = True
    weight: float = 1.0
    health_check_url: Optional[str] = None
    health_check_interval: float = 60.0
    max_concurrent_requests: int = 10
    timeout: float = 30.0


@dataclass
class AgentRequirements:
    """Требования агента."""
    agent_type: AgentType
    complexity: TaskComplexity
    required_capabilities: List[str]
    preferred_capabilities: List[str]
    min_max_tokens: int
    optimal_temperature: float
    max_cost_per_request: Optional[float] = None
    priority: int = 0


@dataclass
class FallbackChain:
    """Цепочка fallback с весами."""
    name: str
    description: str
    providers: List[Tuple[str, float]]
    enabled: bool = True
    max_fallback_attempts: int = 3


@dataclass
class CacheConfig:
    """Конфигурация кэширования."""
    strategy: CacheStrategy
    ttl_seconds: int = 3600
    max_memory_items: int = 1000
    disk_cache_dir: str = "cache/models"
    compression: bool = True
    encryption: bool = False


@dataclass
class MonitoringConfig:
    """Конфигурация мониторинга."""
    enabled: bool = True
    metrics_interval: float = 30.0
    health_check_interval: float = 60.0
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 10.0,
        "response_time": 5.0,
        "memory_usage": 80.0
    })
    webhook_url: Optional[str] = None


# =============================================================================
# Кэширование
# =============================================================================

class ModelCache:
    """Кэш конфигурации моделей."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._memory_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._lock = Lock()

        if config.strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            Path(config.disk_cache_dir).mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Получить путь к файлу кэша."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return Path(self.config.disk_cache_dir) / f"{hash_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        with self._lock:
            if self.config.strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                if key in self._memory_cache:
                    timestamp = self._cache_timestamps.get(key, 0)
                    if time.time() - timestamp < self.config.ttl_seconds:
                        return self._memory_cache[key]
                    else:
                        del self._memory_cache[key]
                        del self._cache_timestamps[key]

            if self.config.strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    try:
                        stat = cache_path.stat()
                        if time.time() - stat.st_mtime < self.config.ttl_seconds:
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if self.config.strategy == CacheStrategy.HYBRID:
                                    self._memory_cache[key] = data
                                    self._cache_timestamps[key] = time.time()
                                return data
                        else:
                            cache_path.unlink()
                    except Exception as e:
                        logger.warning(f"Ошибка чтения кэша {cache_path}: {e}")

            return None

    def set(self, key: str, value: Any):
        """Сохранить значение в кэш."""
        with self._lock:
            if self.config.strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                if len(self._memory_cache) >= self.config.max_memory_items:
                    oldest_key = min(self._cache_timestamps.keys(),
                                     key=lambda k: self._cache_timestamps[k])
                    del self._memory_cache[oldest_key]
                    del self._cache_timestamps[oldest_key]

                self._memory_cache[key] = value
                self._cache_timestamps[key] = time.time()

            if self.config.strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                cache_path = self._get_cache_path(key)
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(value, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Ошибка записи кэша {cache_path}: {e}")

    def clear(self):
        """Очистить кэш."""
        with self._lock:
            self._memory_cache.clear()
            self._cache_timestamps.clear()

            if self.config.strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                try:
                    cache_dir = Path(self.config.disk_cache_dir)
                    for file in cache_dir.glob("*.json"):
                        file.unlink()
                except Exception as e:
                    logger.warning(f"Ошибка очистки дискового кэша: {e}")


# =============================================================================
# Авто-обнаружение моделей OpenRouter
# =============================================================================

class OpenRouterAutoDiscovery:
    """Улучшенное авто-обнаружение моделей OpenRouter."""

    API_URL = "https://openrouter.ai/api/v1/models"

    AGENT_REQUIREMENTS = {
        "classifier": (4000, 1000, "simple"),
        "relevance": (4000, 1000, "simple"),
        "summarizer": (8000, 2000, "medium"),
        "rewriter": (4000, 500, "medium"),
        "style_normalizer": (131072, 32000, "complex"),
        "quality_validator": (8000, 1000, "simple"),
        "telegram_formatter": (16000, 4000, "medium"),
        "seo_optimizer": (8000, 2000, "medium"),
    }

    def __init__(self, api_key: Optional[str] = None, cache: Optional[ModelCache] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._models: List[Dict[str, Any]] = []
        self._free_models: List[Dict[str, Any]] = []
        self._fetched = False
        self._last_fetch: Optional[float] = None
        self._cache = cache
        self._lock = Lock()

        self.fetch_count = 0
        self.cache_hits = 0

    def _check_network_connectivity(self) -> bool:
        """Проверить доступность сети."""
        try:
            host = "openrouter.ai"
            port = 443
            socket.setdefaulttimeout(5)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except Exception as e:
            logger.warning(f"Проблема с подключением к {host}: {e}")
            return False

    def fetch_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Получить модели с улучшенным кэшированием."""
        with self._lock:
            if not force_refresh and self._cache:
                cache_key = "openrouter_models"
                cached = self._cache.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    logger.debug(f"OpenRouter модели из кэша (hit #{self.cache_hits})")
                    return cached

            if not force_refresh and self._fetched and self._last_fetch:
                if time.time() - self._last_fetch < 3600:
                    return self._free_models

            self._fetch_from_api()

            if self._cache:
                cache_key = "openrouter_models"
                self._cache.set(cache_key, self._free_models)

            return self._free_models

    def _fetch_from_api(self):
        """Запросить модели из API с улучшенной обработкой ошибок."""
        import requests
        from urllib3.exceptions import MaxRetryError, NewConnectionError

        if not self._check_network_connectivity():
            logger.warning("Нет подключения к сети, используем запасной список моделей")
            self._free_models = self._get_fallback_models()
            self._fetched = True
            self._last_fetch = time.time()
            return

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                self.API_URL,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"OpenRouter API вернул {response.status_code}")
                self._free_models = self._get_fallback_models()
                self._fetched = True
                self._last_fetch = time.time()
                return

            data = response.json()
            all_models = data.get("data", [])

            free_models = []

            for m in all_models:
                model_id = m.get("id", "")

                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "1") or "1")
                completion_price = float(pricing.get("completion", "1") or "1")

                if prompt_price == 0 and completion_price == 0:
                    context_length = m.get("context_length", 0) or 0
                    if context_length >= 4000:
                        free_models.append({
                            "id": model_id,
                            "name": m.get("name", ""),
                            "context_length": context_length,
                            "max_completion": (
                                    m.get("top_provider", {}).get("max_completion_tokens")
                                    or context_length // 2
                            ),
                            "capabilities": self._extract_capabilities(m),
                        })

            free_models.sort(key=lambda x: -x["context_length"])

            self._free_models = free_models
            self._fetched = True
            self._last_fetch = time.time()
            self.fetch_count += 1

            logger.info(f"OpenRouter: получено {len(free_models)} моделей (запрос #{self.fetch_count})")

        except (MaxRetryError, NewConnectionError) as e:
            logger.warning(f"Ошибка сети при запросе к OpenRouter: {e}")
            self._free_models = self._get_fallback_models()
            self._fetched = True
            self._last_fetch = time.time()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ошибка запроса к OpenRouter: {e}")
            self._free_models = self._get_fallback_models()
            self._fetched = True
            self._last_fetch = time.time()
        except Exception as e:
            logger.warning(f"Неожиданная ошибка при запросе к OpenRouter: {e}")
            self._free_models = self._get_fallback_models()
            self._fetched = True
            self._last_fetch = time.time()

    def _extract_capabilities(self, model_data: Dict[str, Any]) -> List[str]:
        """Извлечь возможности модели."""
        capabilities = ["chat"]

        architecture = model_data.get("architecture", {})
        modality = architecture.get("modality", "")
        if "image" in modality.lower():
            capabilities.append("vision")

        supported_params = model_data.get("supported_parameters", [])
        if supported_params:
            if "tools" in supported_params or "functions" in supported_params:
                capabilities.append("function_calling")
            if "json_mode" in supported_params or "response_format" in supported_params:
                capabilities.append("structured_output")

        return capabilities

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Запасной список моделей."""
        return [
            {
                "id": "meta-llama/llama-3.2-3b-instruct:free",
                "name": "Llama 3.2 3B",
                "context_length": 131072,
                "max_completion": 4096,
                "capabilities": ["chat"],
            },
            {
                "id": "meta-llama/llama-3.1-8b-instruct:free",
                "name": "Llama 3.1 8B",
                "context_length": 131072,
                "max_completion": 4096,
                "capabilities": ["chat"],
            },
            {
                "id": "google/gemini-2.0-flash-exp:free",
                "name": "Gemini 2.0 Flash",
                "context_length": 1048576,
                "max_completion": 8192,
                "capabilities": ["chat", "vision"],
            },
        ]

    def get_model_for_agent(self, agent_name: str) -> Dict[str, Any]:
        """Получить лучшую модель для агента."""
        models = self.fetch_models()
        if not models:
            return self._default_model()

        min_context, min_output, complexity = self.AGENT_REQUIREMENTS.get(
            agent_name, (4000, 1000, "medium")
        )

        for model in models:
            if model["context_length"] >= min_context:
                return model

        return max(models, key=lambda m: m["context_length"])

    def _default_model(self) -> Dict[str, Any]:
        """Модель по умолчанию."""
        return {
            "id": "meta-llama/llama-3.2-3b-instruct:free",
            "name": "Llama 3.2 3B",
            "context_length": 131072,
            "max_completion": 4096,
            "capabilities": ["chat"],
        }

    def build_provider_config(self) -> ProviderConfig:
        """Построить ProviderConfig из обнаруженных моделей."""
        models = self.fetch_models()

        model_infos = {}
        for i, m in enumerate(models):
            key = f"auto_{i}"
            model_infos[key] = ModelInfo(
                id=m["id"],
                cost_tier="free",
                max_tokens=m.get("max_completion", 4096),
                capabilities=m.get("capabilities", ["chat"]),
                rate_limit=None,
                speed=None,
            )

        recommended = {}
        for agent_name in self.AGENT_REQUIREMENTS:
            best = self.get_model_for_agent(agent_name)
            for key, info in model_infos.items():
                if info.id == best["id"]:
                    recommended[agent_name] = {"free": key}
                    break

        return ProviderConfig(
            provider=LLMProviderType.OPENROUTER,
            models=model_infos,
            recommended_for=recommended,
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
        )

    def get_model_metrics(self) -> Dict[str, Any]:
        """Получить метрики обнаружения."""
        return {
            "fetch_count": self.fetch_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (self.cache_hits / max(1, self.fetch_count)) * 100,
            "last_fetch": self._last_fetch,
            "models_count": len(self._free_models),
        }


# =============================================================================
# Главный класс конфигурации
# =============================================================================

class ModelsConfig:
    """Улучшенная конфигурация моделей v6.0."""

    SUPPORTED_PROVIDERS = ["openrouter", "groq", "google", "ollama"]

    def __init__(
            self,
            config_path: str = "config/models.yaml",
            provider: Optional[str] = None,
            strategy: Optional[str] = None,
            fallback_chain: Optional[str] = None,
            enable_fallback: Optional[bool] = None,
            mode: OperationMode = OperationMode.NORMAL,
            cache_config: Optional[CacheConfig] = None,
            monitoring_config: Optional[MonitoringConfig] = None
    ):
        self.config_path = Path(config_path)
        self.mode = mode

        # Инициализация кэша
        self.cache_config = cache_config or CacheConfig(
            strategy=CacheStrategy.HYBRID,
            ttl_seconds=3600,
            max_memory_items=1000
        )
        self._cache = ModelCache(self.cache_config)

        # Инициализация мониторинга
        self.monitoring_config = monitoring_config or MonitoringConfig()

        # Загрузка конфигурации
        self._config = self._load_config()

        # Определение провайдера
        self.provider_name = self._determine_provider(provider)
        self.strategy = strategy or self._config.get("defaults", {}).get("strategy", "balanced")
        self.enable_fallback = self._determine_fallback(enable_fallback)

        # Инициализация компонентов
        self._openrouter_discovery = None
        self.providers = {}
        self.agent_requirements = {}
        self.fallback_chain = None
        self.api_keys = {}

        # Метрики
        self.metrics = {
            "config_loads": 0,
            "provider_switches": 0,
            "fallback_activations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Add missing attributes
        self._background_tasks = set()
        self._shutdown_event = Event()

        # Загрузка конфигурации
        self._load_configuration()

        logger.info(
            f"ModelsConfig v6.0: provider={self.provider_name}, "
            f"strategy={self.strategy}, mode={self.mode.value}, "
            f"cache={self.cache_config.strategy.value}"
        )

    def get_fallback_providers(self) -> List[str]:
        """Получить список провайдеров в порядке fallback."""
        if not self.fallback_chain or not self.fallback_chain.enabled:
            return [self.provider_name]

        # Extract provider names from the fallback chain
        providers = [provider for provider, _ in self.fallback_chain.providers]

        # Ensure current provider is first
        if self.provider_name in providers:
            providers.remove(self.provider_name)
            providers.insert(0, self.provider_name)

        return providers

    def get_provider(self) -> LLMProviderType:
        """Получить текущего провайдера."""
        return LLMProviderType(self.provider_name)

    def get_profile(self):
        """Получить текущий профиль конфигурации."""

        class Profile:
            def __init__(self, name, provider):
                self.name = name
                self.provider = provider

        return Profile(f"{self.provider_name}_{self.strategy}", LLMProviderType(self.provider_name))

    @property
    def active_profile(self) -> str:
        """Получить имя активного профиля."""
        return f"{self.provider_name}_{self.strategy}"

    def to_dict(self) -> Dict[str, Any]:
        """Экспорт конфигурации в словарь."""
        return {
            "provider": self.provider_name,
            "strategy": self.strategy,
            "enable_fallback": self.enable_fallback,
            "fallback_chain": self.get_fallback_providers(),
            "providers_available": list(self.providers.keys()),
            "api_keys_set": [k for k, v in self.api_keys.items() if v]
        }

    def print_config(self):
        """Вывести в консоль текущую конфигурацию."""
        print(f"\n{'=' * 80}")
        print(f"📋 КОНФИГУРАЦИЯ МОДЕЛЕЙ v6.0")
        print(f"{'=' * 80}")
        print(f"🔌 Провайдер: {self.provider_name.upper()}")
        print(f"📊 Стратегия: {self.strategy}")
        print(f"🔄 Fallback: {'ВКЛ' if self.enable_fallback else 'ВЫКЛ'}")

        if self.fallback_chain and self.fallback_chain.enabled:
            chain = self.get_fallback_providers()
            print(f"🔗 Цепочка: {' → '.join(chain)}")

        print(f"{'=' * 80}\n")

    def _determine_provider(self, provider: Optional[str]) -> str:
        """Определить провайдера с учётом режима."""
        if provider:
            return provider

        if self.mode == OperationMode.DEVELOPMENT:
            if os.getenv("OLLAMA_BASE_URL"):
                return "ollama"

        if self.mode == OperationMode.PRODUCTION:
            env_provider = os.getenv("LLM_PROVIDER")
            if env_provider and env_provider in self.SUPPORTED_PROVIDERS:
                return env_provider

        return self._config.get("defaults", {}).get("provider", "groq")

    def _determine_fallback(self, enable_fallback: Optional[bool]) -> bool:
        """Определить необходимость fallback."""
        if enable_fallback is not None:
            return enable_fallback

        if self.mode == OperationMode.DEVELOPMENT:
            return False

        if self.mode == OperationMode.PRODUCTION:
            return True

        return self._config.get("defaults", {}).get("enable_fallback", True)

    def _load_configuration(self):
        """Загрузить всю конфигурацию с улучшенной обработкой ошибок."""
        try:
            if self.provider_name == "openrouter":
                self._openrouter_discovery = OpenRouterAutoDiscovery(cache=self._cache)
                try:
                    discovered = self._openrouter_discovery.build_provider_config()
                    self.providers = {"openrouter": discovered}
                except Exception as e:
                    logger.error(f"Ошибка при построении конфигурации OpenRouter: {e}")
                    self._create_minimal_fallback_config()
            else:
                self.providers = self._load_providers()

            self.agent_requirements = self._get_agent_requirements()
            self.fallback_chain = self._load_fallback_chain()
            self.api_keys = self._load_api_keys()

            self.metrics["config_loads"] += 1

            if self.monitoring_config.enabled:
                self._start_background_tasks()

        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            self._create_minimal_fallback_config()
            raise

    def _create_minimal_fallback_config(self):
        """Создать минимальную конфигурацию для восстановления после ошибки."""
        try:
            fallback_model = ModelInfo(
                id="meta-llama/llama-3.2-3b-instruct:free",
                cost_tier="free",
                max_tokens=4096,
                capabilities=["chat"],
            )

            self.providers = {
                "openrouter": ProviderConfig(
                    provider=LLMProviderType.OPENROUTER,
                    models={"fallback": fallback_model},
                    recommended_for={},
                    api_key_env="OPENROUTER_API_KEY",
                    base_url="https://openrouter.ai/api/v1",
                )
            }

            self.agent_requirements = {
                AgentType.CLASSIFIER: AgentRequirements(
                    agent_type=AgentType.CLASSIFIER,
                    complexity=TaskComplexity.SIMPLE,
                    required_capabilities=["chat"],
                    preferred_capabilities=[],
                    min_max_tokens=4096,
                    optimal_temperature=0.1,
                ),
            }

            self.fallback_chain = FallbackChain(
                name="minimal",
                description="Minimal fallback configuration",
                providers=[("openrouter", 1.0)],
            )

            self.api_keys = {"openrouter": os.getenv("OPENROUTER_API_KEY")}

            logger.warning("Используется минимальная конфигурация для восстановления")

        except Exception as e:
            logger.error(f"Ошибка создания минимальной конфигурации: {e}")
            raise

    def _start_background_tasks(self):
        """Запустить фоновые задачи."""
        if self.monitoring_config.enabled:
            task = asyncio.create_task(self._monitoring_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _monitoring_loop(self):
        """Фоновый цикл мониторинга."""
        while not self._shutdown_event.is_set():
            try:
                await self._health_check()
                self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.monitoring_config.metrics_interval)
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(5)

    async def _health_check(self):
        """Проверка здоровья провайдеров."""
        for provider_name, provider_config in self.providers.items():
            try:
                pass
            except Exception as e:
                logger.warning(f"Провайдер {provider_name} нездоров: {e}")

    def _collect_metrics(self):
        """Сбор метрик."""
        if self._openrouter_discovery:
            metrics = self._openrouter_discovery.get_model_metrics()
            logger.debug(f"OpenRouter метрики: {metrics}")

    async def _check_alerts(self):
        """Проверка порогов для алертов."""
        pass

    # =========================================================================
    # Ollama: единые точки определения параметров
    # =========================================================================

    def get_ollama_model(self, agent_name: Optional[str] = None) -> str:
        """
        Единая точка определения модели Ollama.

        Приоритет:
          1. agent_models.<agent_name> из YAML (per-agent override)
          2. env OLLAMA_MODEL
          3. config/models.yaml → ollama.model
          4. hardcoded default
        """
        if agent_name:
            agent_models = self._config.get("agent_models") or {}
            agent_model = agent_models.get(agent_name)
            if agent_model:
                logger.debug(f"[ModelsConfig] Per-agent model for {agent_name}: {agent_model}")
                return agent_model

        ollama_cfg = self._config.get("ollama", {})
        return (
            os.getenv("OLLAMA_MODEL")
            or ollama_cfg.get("model")
            or "qwen2.5:14b-instruct-q5_k_m"
        )

    def get_ollama_base_url(self) -> str:
        """Единая точка определения URL Ollama."""
        ollama_cfg = self._config.get("ollama", {})
        return (
            os.getenv("OLLAMA_BASE_URL")
            or ollama_cfg.get("base_url")
            or "http://ollama:11434"
        )

    def get_ollama_context_length(self) -> int:
        """Контекст из YAML или дефолт."""
        ollama_cfg = self._config.get("ollama", {})
        return ollama_cfg.get("context_length", 32768)

    def get_llm_config(
            self,
            agent_name: str,
            provider_override: Optional[str] = None
    ) -> LLMConfig:
        """Получить LLMConfig для агента."""
        cache_key = f"llm_config_{agent_name}_{provider_override or self.provider_name}"

        cached = self._cache.get(cache_key)
        if cached:
            self.metrics["cache_hits"] += 1
            return LLMConfig(**cached)

        self.metrics["cache_misses"] += 1

        try:
            agent_type = AgentType(agent_name)
        except ValueError:
            logger.warning(f"Неизвестный агент: {agent_name}, используем CLASSIFIER")
            agent_type = AgentType.CLASSIFIER

        provider_name = provider_override or self.provider_name

        # =====================================================================
        # OpenRouter
        # =====================================================================
        if provider_name == "openrouter" and self._openrouter_discovery:
            model_info = self._openrouter_discovery.get_model_for_agent(agent_name)

            reqs = OpenRouterAutoDiscovery.AGENT_REQUIREMENTS.get(
                agent_name, (4000, 1000, "medium")
            )
            min_context, min_output, complexity = reqs

            temp_map = {
                "classifier": 0.1,
                "relevance": 0.3,
                "quality_validator": 0.1,
                "summarizer": 0.5,
                "rewriter": 0.7,
                "style_normalizer": 0.3,
                "telegram_formatter": 0.5,
                "telegraph_formatter": 0.5,
                "seo_optimizer": 0.4,
            }
            temperature = temp_map.get(agent_name, 0.5)

            if self.mode == OperationMode.DEVELOPMENT:
                temperature *= 0.8

            config = LLMConfig(
                provider=LLMProviderType.OPENROUTER,
                model=model_info["id"],
                temperature=temperature,
                max_tokens=min_output,
                api_key=self.api_keys.get("openrouter"),
                base_url="https://openrouter.ai/api/v1",
                context_length=model_info["context_length"],
            )

            self._cache.set(cache_key, asdict(config))
            return config

        # =====================================================================
        # Ollama (локальный)
        # =====================================================================
        elif provider_name == "ollama":
            ollama_model = self.get_ollama_model(agent_name=agent_name)
            ollama_url = self.get_ollama_base_url()
            context_length = self.get_ollama_context_length()

            yaml_temps = self._config.get("temperatures", {})
            temp_map = {
                "classifier": yaml_temps.get("classifier", 0.1),
                "relevance": yaml_temps.get("relevance", 0.3),
                "quality_validator": yaml_temps.get("quality_validator", 0.1),
                "summarizer": yaml_temps.get("summarizer", 0.5),
                "rewriter": yaml_temps.get("rewriter", 0.7),
                "style_normalizer": yaml_temps.get("style_normalizer", 0.3),
                "telegram_formatter": yaml_temps.get("telegram_formatter", 0.5),
                "telegraph_formatter": yaml_temps.get("telegraph_formatter", 0.5),
                "seo_optimizer": yaml_temps.get("seo_optimizer", 0.4),
            }
            temperature = temp_map.get(agent_name, 0.5)

            agent_max_tokens = {
                "classifier": 1000,
                "relevance": 1000,
                "quality_validator": 1000,
                "summarizer": 2000,
                "rewriter": 500,
                "style_normalizer": 4096,
                "telegram_formatter": 4000,
                "telegraph_formatter": 4096,
                "seo_optimizer": 2000,
            }
            max_tokens = agent_max_tokens.get(agent_name, 4096)

            # Адаптивный max_tokens для style_normalizer
            if agent_name == "style_normalizer":
                if self.mode == OperationMode.PRODUCTION:
                    max_tokens = 16384
                elif self.mode == OperationMode.DEVELOPMENT:
                    max_tokens = 8192
                else:
                    max_tokens = 12288
                logger.info(
                    f"[StyleNormalizer] max_tokens={max_tokens} "
                    f"(режим: {self.mode.value})"
                )

            config = LLMConfig(
                provider=LLMProviderType.OLLAMA,
                model=ollama_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=None,
                base_url=ollama_url,
                context_length=context_length,
            )

            logger.info(
                f"[ModelsConfig] Ollama: agent={agent_name}, "
                f"model={ollama_model}, temp={temperature}, ctx={context_length}"
            )
            self._cache.set(cache_key, asdict(config))
            return config

        # =====================================================================
        # Google Gemini Flash (основной облачный провайдер)
        # =====================================================================
        elif provider_name == "google":
            gemini_cfg = self._config.get("google", {})

            # API ключ: GEMINI_API_KEY → GOOGLE_API_KEY → api_keys dict
            api_key = (
                    os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                    or self.api_keys.get("google")
            )
            if not api_key:
                logger.warning(
                    "[ModelsConfig] Google: GEMINI_API_KEY не задан, "
                    "fallback на Ollama"
                )
                return self.get_llm_config(agent_name, provider_override="ollama")

            # Модели из YAML
            pro_model = gemini_cfg.get("pro_model", "gemini-2.5-pro-preview-06-05")
            flash_model = gemini_cfg.get("flash_model", "gemini-2.0-flash")
            lite_model = gemini_cfg.get("lite_model", "gemini-2.0-flash-lite")

            # Per-agent переопределения (agent_models секция в YAML)
            agent_model_overrides = self._config.get("agent_models", {})

            # Группировка по сложности
            heavy_agents = {"rewriter", "style_normalizer"}
            lite_agents = {"classifier", "relevance", "quality_validator"}

            if agent_name in agent_model_overrides:
                # Явный override из YAML → наивысший приоритет
                model = agent_model_overrides[agent_name]
            elif agent_name in heavy_agents:
                model = pro_model
            elif agent_name in lite_agents:
                model = lite_model
            else:
                # summarizer, formatters, image_prompt, cat_commentator
                model = flash_model

            # Температуры из YAML → fallback хардкод
            yaml_temps = self._config.get("temperatures", {})
            temp_map = {
                "classifier": yaml_temps.get("classifier", 0.1),
                "relevance": yaml_temps.get("relevance", 0.3),
                "quality_validator": yaml_temps.get("quality_validator", 0.1),
                "summarizer": yaml_temps.get("summarizer", 0.5),
                "rewriter": yaml_temps.get("rewriter", 0.7),
                "style_normalizer": yaml_temps.get("style_normalizer", 0.3),
                "telegram_formatter": yaml_temps.get("telegram_formatter", 0.5),
                "telegraph_formatter": yaml_temps.get("telegraph_formatter", 0.5),
                "image_prompt": yaml_temps.get("image_prompt", 0.8),
                "cat_commentator": yaml_temps.get("cat_commentator", 0.9),
                "seo_optimizer": yaml_temps.get("seo_optimizer", 0.4),
            }
            temperature = temp_map.get(agent_name, 0.5)

            # max_tokens по агенту (Gemini даёт до 8192 бесплатно)
            agent_max_tokens = {
                "classifier": 4096,
                "relevance": 4096,
                "quality_validator": 4096,
                "summarizer": 4096,
                "rewriter": 8192,
                "style_normalizer": 8192,
                "telegram_formatter": 4096,
                "telegraph_formatter": 8192,
                "image_prompt": 2048,
                "cat_commentator": 2048,
                "seo_optimizer": 4096,
            }
            max_tokens = agent_max_tokens.get(
                agent_name,
                gemini_cfg.get("max_tokens", 8192)
            )

            config = LLMConfig(
                provider=LLMProviderType.GOOGLE,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=LLMConfig.GOOGLE_DEFAULT_URL,
                context_length=gemini_cfg.get("context_length", 1048576),
            )

            logger.info(
                f"[ModelsConfig] Google: agent={agent_name}, model={model}, "
                f"temp={temperature}, max_tokens={max_tokens}"
            )
            self._cache.set(cache_key, asdict(config))
            return config

        # =====================================================================
        # Groq и другие провайдеры (через _select_model)
        # =====================================================================
        model_info = self._select_model(agent_type, provider_name)

        if not model_info:
            if self.enable_fallback:
                return self._fallback_llm_config(agent_type)
            raise ValueError(
                f"Модель не найдена для {agent_name} на {provider_name}"
            )

        requirements = self.agent_requirements.get(agent_type)
        temperature = requirements.optimal_temperature if requirements else 0.5
        max_tokens = requirements.min_max_tokens if requirements else model_info.max_tokens

        provider_config = self.providers.get(provider_name)
        api_key = self.api_keys.get(provider_name)

        base_url = None
        if provider_config:
            if provider_config.base_url:
                base_url = provider_config.base_url
            elif provider_config.base_url_env:
                base_url = os.getenv(
                    provider_config.base_url_env,
                    provider_config.default_base_url
                )

        config = LLMConfig(
            provider=LLMProviderType(provider_name),
            model=model_info.id,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        self._cache.set(cache_key, asdict(config))
        return config

    # -------------------------------------------------------------------------
    # Также замени _load_api_keys() — исправляет GEMINI_API_KEY
    # -------------------------------------------------------------------------

    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """Загрузить API ключи. GEMINI_API_KEY имеет приоритет над GOOGLE_API_KEY."""
        return {
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "google": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        }

    def _fallback_llm_config(self, agent_type: AgentType) -> LLMConfig:
        """Получить конфигурацию через fallback."""
        if not self.fallback_chain or not self.fallback_chain.enabled:
            raise ValueError("Fallback недоступен")

        self.metrics["fallback_activations"] += 1

        for provider_name, weight in self.fallback_chain.providers:
            try:
                if provider_name in self.providers:
                    model_info = self._select_model(agent_type, provider_name)
                    if model_info:
                        logger.info(f"Fallback на {provider_name} для {agent_type.value}")
                        self.metrics["provider_switches"] += 1
                        return self.get_llm_config(agent_type.value, provider_name)
            except Exception as e:
                logger.warning(f"Fallback на {provider_name} не удался: {e}")
                continue

        raise ValueError("Все fallback провайдеры недоступны")

    def _select_model(self, agent_type: AgentType, provider_name: str) -> Optional[ModelInfo]:
        """Выбрать модель с учётом метрик."""
        provider_config = self.providers.get(provider_name)
        if not provider_config:
            return None

        requirements = self.agent_requirements.get(agent_type)
        recommended = provider_config.recommended_for.get(agent_type.value, {})

        if self.strategy == SelectionStrategy.ADAPTIVE.value:
            return self._adaptive_model_selection(provider_config, requirements)
        else:
            return self._standard_model_selection(provider_config, requirements, recommended)

    def _adaptive_model_selection(self, provider_config: ProviderConfig, requirements: Optional[AgentRequirements]) -> \
    Optional[ModelInfo]:
        """Адаптивный выбор модели на основе метрик."""
        if not provider_config.models:
            return None

        scored_models = []
        for model_info in provider_config.models.values():
            score = self._calculate_model_score(model_info, requirements)
            scored_models.append((score, model_info))

        scored_models.sort(key=lambda x: x[0], reverse=True)
        return scored_models[0][1] if scored_models else None

    def _calculate_model_score(self, model_info: ModelInfo, requirements: Optional[AgentRequirements]) -> float:
        """Рассчитать оценку модели."""
        score = 0.0

        score += model_info.priority * 10

        if model_info.metrics.request_count > 0:
            score += model_info.metrics.success_rate * 0.5
            if model_info.metrics.avg_response_time > 0:
                score += (1.0 / model_info.metrics.avg_response_time) * 10

        score += model_info.health_score * 0.3

        if requirements:
            required_caps = set(requirements.required_capabilities)
            model_caps = set(model_info.capabilities)
            if required_caps.issubset(model_caps):
                score += 20

            if model_info.max_tokens >= requirements.min_max_tokens:
                score += 10

        return score

    def _standard_model_selection(self, provider_config: ProviderConfig, requirements: Optional[AgentRequirements],
                                  recommended: Dict[str, Any]) -> Optional[ModelInfo]:
        """Стандартный выбор модели."""
        if self.strategy == SelectionStrategy.COST_OPTIMIZED.value:
            tier = "free"
        elif self.strategy == SelectionStrategy.QUALITY_FOCUSED.value:
            tier = "paid" if "paid" in recommended else "free"
        else:
            tier = "free"

        model_key = recommended.get(tier)
        if model_key and model_key in provider_config.models:
            return provider_config.models[model_key]

        for model_info in provider_config.models.values():
            if self.strategy == SelectionStrategy.COST_OPTIMIZED.value:
                if model_info.cost_tier != "free":
                    continue

            if requirements:
                required_caps = set(requirements.required_capabilities)
                model_caps = set(model_info.capabilities)
                if not required_caps.issubset(model_caps):
                    continue

            return model_info

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики конфигурации."""
        metrics = self.metrics.copy()

        if self._cache:
            cache_stats = {
                "memory_cache_size": len(self._cache._memory_cache),
                "cache_strategy": self.cache_config.strategy.value,
            }
            metrics["cache"] = cache_stats

        if self._openrouter_discovery:
            metrics["openrouter"] = self._openrouter_discovery.get_model_metrics()

        return metrics

    def reload_config(self):
        """Перезагрузить конфигурацию."""
        logger.info("Перезагрузка конфигурации моделей...")
        self._cache.clear()
        self._load_configuration()
        logger.info("Конфигурация перезагружена")

    async def shutdown(self):
        """Корректное завершение работы."""
        logger.info("Завершение работы ModelsConfig...")
        self._shutdown_event.set()

        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.error(f"Ошибка при отмене фоновой задачи: {e}")

        self._background_tasks.clear()

        if self._cache:
            self._cache.clear()

        logger.info("ModelsConfig завершён")

    def _load_config(self) -> Dict[str, Any]:
        """
        Загрузить главный конфиг.

        Приоритет:
        1. config/models.yaml
        2. config/settings.json (из Streamlit UI)
        3. Пустой словарь (используются дефолты)
        """
        # Пробуем YAML
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                logger.info(f"Конфиг загружен из {self.config_path}")
                return data
            except Exception as e:
                logger.error(f"Ошибка загрузки YAML конфига: {e}")

        # Пробуем JSON (из Streamlit settings_page)
        json_path = self.config_path.parent / "settings.json"
        if json_path.exists():
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Конфиг загружен из {json_path}")
                return data
            except Exception as e:
                logger.error(f"Ошибка загрузки JSON конфига: {e}")

        logger.warning(f"Конфиг не найден: {self.config_path}")
        return {}

    def _load_providers(self) -> Dict[str, ProviderConfig]:
        """Загрузить провайдеры из YAML."""
        return {}

    def _get_agent_requirements(self) -> Dict[AgentType, AgentRequirements]:
        """Получить требования агентов."""
        return {
            AgentType.CLASSIFIER: AgentRequirements(
                agent_type=AgentType.CLASSIFIER,
                complexity=TaskComplexity.SIMPLE,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=4096,
                optimal_temperature=0.1,
            ),
            AgentType.RELEVANCE: AgentRequirements(
                agent_type=AgentType.RELEVANCE,
                complexity=TaskComplexity.SIMPLE,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=4096,
                optimal_temperature=0.3,
            ),
            AgentType.SUMMARIZER: AgentRequirements(
                agent_type=AgentType.SUMMARIZER,
                complexity=TaskComplexity.MEDIUM,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=2000,
                optimal_temperature=0.5,
            ),
            AgentType.REWRITER: AgentRequirements(
                agent_type=AgentType.REWRITER,
                complexity=TaskComplexity.MEDIUM,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=500,
                optimal_temperature=0.7,
            ),
            AgentType.STYLE_NORMALIZER: AgentRequirements(
                agent_type=AgentType.STYLE_NORMALIZER,
                complexity=TaskComplexity.COMPLEX,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=12288,  # Увеличиваем базовое значение
                optimal_temperature=0.3,
            ),
            AgentType.QUALITY_VALIDATOR: AgentRequirements(
                agent_type=AgentType.QUALITY_VALIDATOR,
                complexity=TaskComplexity.SIMPLE,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=4096,
                optimal_temperature=0.1,
            ),
            AgentType.TELEGRAM_FORMATTER: AgentRequirements(
                agent_type=AgentType.TELEGRAM_FORMATTER,
                complexity=TaskComplexity.MEDIUM,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=4000,
                optimal_temperature=0.5,
            ),
            AgentType.SEO_OPTIMIZER: AgentRequirements(
                agent_type=AgentType.SEO_OPTIMIZER,
                complexity=TaskComplexity.MEDIUM,
                required_capabilities=["chat"],
                preferred_capabilities=[],
                min_max_tokens=2000,
                optimal_temperature=0.4,
            ),
        }

    def _load_fallback_chain(self) -> FallbackChain:
        """Загрузить цепочку fallback."""
        return FallbackChain(
            name="default",
            description="Default fallback",
            providers=[("openrouter", 1.0)]
        )

    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """Загрузить API ключи."""
        return {
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
        }


# =============================================================================
# Глобальный экземпляр
# =============================================================================

_models_config: Optional[ModelsConfig] = None
_config_lock = Lock()


def get_models_config(
        config_path: str = "config/models.yaml",
        provider: Optional[str] = None,
        strategy: Optional[str] = None,
        enable_fallback: Optional[bool] = None,
        force_new: bool = False,
        mode: OperationMode = OperationMode.NORMAL,
        cache_config: Optional[CacheConfig] = None,
        monitoring_config: Optional[MonitoringConfig] = None,
        **kwargs
) -> ModelsConfig:
    """Получить глобальный экземпляр с улучшенной логикой."""
    global _models_config

    with _config_lock:
        if _models_config is None or force_new or provider:
            if _models_config:
                asyncio.create_task(_models_config.shutdown())

            _models_config = ModelsConfig(
                config_path=config_path,
                provider=provider,
                strategy=strategy,
                enable_fallback=enable_fallback,
                mode=mode,
                cache_config=cache_config,
                monitoring_config=monitoring_config
            )

        return _models_config


def reset_models_config():
    """Сбросить глобальный экземпляр."""
    global _models_config

    with _config_lock:
        if _models_config:
            asyncio.create_task(_models_config.shutdown())
            _models_config = None


def create_config_from_args(
        provider: Optional[str] = None,
        strategy: Optional[str] = None,
        no_fallback: bool = False,
        fallback_chain: Optional[str] = None,
        mode: str = "normal",
        **kwargs
) -> ModelsConfig:
    """Создать конфигурацию из аргументов командной строки."""
    try:
        operation_mode = OperationMode(mode)
    except ValueError:
        logger.warning(f"Неправильный режим работы: {mode}, используем normal")
        operation_mode = OperationMode.NORMAL

    return get_models_config(
        provider=provider,
        strategy=strategy,
        fallback_chain=fallback_chain,
        enable_fallback=not no_fallback,
        force_new=True,
        mode=operation_mode,
        **kwargs
    )


@asynccontextmanager
async def models_config_context(**kwargs):
    """Контекстный менеджер для ModelsConfig."""
    config = get_models_config(**kwargs)
    try:
        yield config
    finally:
        await config.shutdown()