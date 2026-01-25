# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/infrastructure/ai/llm_provider.py
# =============================================================================
"""
LLM Provider - Multi-Provider система с автоматическим fallback.

Версия 3.0.0:
- OpenRouter (бесплатные модели, 50 req/day)
- Groq (30 req/min, очень быстрый)
- Google Gemini (60 req/min, 1500 req/day)
- HuggingFace (fallback)
- Ollama (локальный)

При выборе провайдера вручную — остальные используются как fallback при 429.

Примеры:
    >>> # Автоматический выбор с fallback по всем провайдерам
    >>> llm = get_llm_provider("auto")

    >>> # Groq как основной, остальные как fallback
    >>> llm = get_llm_provider("groq")

    >>> # OpenRouter как основной
    >>> llm = get_llm_provider("openrouter", model="meta-llama/llama-3.3-70b-instruct:free")
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Type, TypeVar, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import re
import time
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LLMProviderType(str, Enum):
    """Поддерживаемые провайдеры."""
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"  # Автоматический выбор


@dataclass
class LLMConfig:
    """Конфигурация LLM провайдера."""
    provider: LLMProviderType = LLMProviderType.OLLAMA
    model: str = "qwen2.5:14b-instruct-q5_k_m"
    temperature: float = 0.7
    max_tokens: int = 2000
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120
    use_fallback: bool = True  # Использовать fallback при ошибках

    OLLAMA_DEFAULT_URL = "http://ollama:11434"
    OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"

    def get_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        if self.provider == LLMProviderType.OLLAMA:
            return self.OLLAMA_DEFAULT_URL
        elif self.provider == LLMProviderType.OPENROUTER:
            return self.OPENROUTER_DEFAULT_URL
        return ""


# =============================================================================
# Базовый класс провайдера
# =============================================================================

class LLMProvider(ABC):
    """Абстрактный базовый класс для LLM провайдеров."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[BaseChatModel] = None

    @property
    def client(self) -> BaseChatModel:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self) -> BaseChatModel:
        pass

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        client = self.client
        if temperature is not None or max_tokens is not None:
            client = self._get_client_with_overrides(temperature, max_tokens)

        try:
            response = client.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Ошибка генерации LLM: {e}", exc_info=True)
            raise

    def generate_structured(
            self,
            prompt: str,
            output_schema: Type[T],
            system_prompt: Optional[str] = None
    ) -> T:
        parser = PydanticOutputParser(pydantic_object=output_schema)
        format_instructions = parser.get_format_instructions()
        enhanced_prompt = f"{prompt}\n\n{format_instructions}"

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=enhanced_prompt))

        try:
            response = self.client.invoke(messages)
            return parser.parse(response.content)
        except Exception as e:
            logger.error(f"Ошибка структурированной генерации: {e}", exc_info=True)
            raise

    def _get_client_with_overrides(
            self,
            temperature: Optional[float],
            max_tokens: Optional[int]
    ) -> BaseChatModel:
        return self.client

    def health_check(self) -> bool:
        try:
            self.generate("Скажи 'ok'", max_tokens=10)
            return True
        except Exception as e:
            logger.warning(f"Health check не прошёл: {e}")
            return False


# =============================================================================
# Ollama Provider (локальный)
# =============================================================================

class OllamaProvider(LLMProvider):
    """Локальный Ollama провайдер."""

    def _create_client(self) -> BaseChatModel:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=self.config.model,
            base_url=self.config.get_base_url(),
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            timeout=self.config.timeout
        )

    def _get_client_with_overrides(self, temperature, max_tokens) -> BaseChatModel:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=self.config.model,
            base_url=self.config.get_base_url(),
            temperature=temperature if temperature is not None else self.config.temperature,
            num_predict=max_tokens if max_tokens is not None else self.config.max_tokens,
            timeout=self.config.timeout
        )

    def list_models(self) -> list[str]:
        try:
            import ollama
            client = ollama.Client(host=self.config.get_base_url())
            models = client.list()
            return [m['name'] for m in models.get('models', [])]
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей Ollama: {e}")
            return []


# =============================================================================
# OpenRouter Provider
# =============================================================================

class OpenRouterProvider(LLMProvider):
    """OpenRouter провайдер с бесплатными моделями."""

    MODELS = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
        "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
        "mistral-7b": "mistralai/mistral-7b-instruct",
        "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
        # Free models
        "llama-3.3-70b-free": "meta-llama/llama-3.3-70b-instruct:free",
        "gemma-3-27b-free": "google/gemma-3-27b-it:free",
        "mistral-24b-free": "mistralai/mistral-small-3.1-24b-instruct:free",
    }

    FREE_MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "qwen/qwen-2.5-vl-7b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ]

    def __init__(self, config: LLMConfig):
        if not config.api_key:
            config.api_key = os.getenv("OPENROUTER_API_KEY")
        if not config.api_key:
            raise ValueError("OPENROUTER_API_KEY не установлен")
        super().__init__(config)

    def _create_client(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        model = self._resolve_model(self.config.model)

        client_kwargs = {
            "model": model,
            "openai_api_key": self.config.api_key,
            "openai_api_base": self.config.get_base_url(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "default_headers": {
                "HTTP-Referer": "https://news-aggregator-pro.local",
                "X-Title": "News Aggregator Pro"
            }
        }

        if "glm" in model.lower() or "z-ai" in model.lower():
            client_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        return ChatOpenAI(**client_kwargs)

    def _get_client_with_overrides(self, temperature, max_tokens) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        model = self._resolve_model(self.config.model)

        client_kwargs = {
            "model": model,
            "openai_api_key": self.config.api_key,
            "openai_api_base": self.config.get_base_url(),
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "timeout": self.config.timeout,
            "default_headers": {
                "HTTP-Referer": "https://news-aggregator-pro.local",
                "X-Title": "News Aggregator Pro"
            }
        }

        if "glm" in model.lower() or "z-ai" in model.lower():
            client_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        return ChatOpenAI(**client_kwargs)

    def _resolve_model(self, model: str) -> str:
        if model == "auto" or not model:
            return self.FREE_MODELS[0]
        return self.MODELS.get(model, model)


# =============================================================================
# Groq Provider (очень быстрый, 30 req/min бесплатно)
# =============================================================================

class GroqProvider(LLMProvider):
    """
    Groq провайдер - самый быстрый inference.

    Бесплатно: 30 req/min, 14400 req/day
    Модели: llama-3.1-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768

    Получить ключ: https://console.groq.com
    """

    DEFAULT_MODEL = "llama-3.1-70b-versatile"

    MODELS = {
        "llama-70b": "llama-3.1-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "mixtral": "mixtral-8x7b-32768",
        "gemma-9b": "gemma2-9b-it",
        "llama-3.3-70b": "llama-3.3-70b-versatile",
    }

    def __init__(self, config: LLMConfig):
        if not config.api_key:
            config.api_key = os.getenv("GROQ_API_KEY")
        if not config.api_key:
            raise ValueError(
                "GROQ_API_KEY не установлен. "
                "Получить бесплатно: https://console.groq.com"
            )
        super().__init__(config)

    def _create_client(self) -> BaseChatModel:
        from langchain_groq import ChatGroq

        model = self.config.model
        if model == "auto" or not model:
            model = self.DEFAULT_MODEL
        model = self.MODELS.get(model, model)

        return ChatGroq(
            model=model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            max_retries=2,
        )

    def _get_client_with_overrides(self, temperature, max_tokens) -> BaseChatModel:
        from langchain_groq import ChatGroq

        model = self.config.model
        if model == "auto" or not model:
            model = self.DEFAULT_MODEL
        model = self.MODELS.get(model, model)

        return ChatGroq(
            model=model,
            api_key=self.config.api_key,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            timeout=self.config.timeout,
            max_retries=2,
        )


# =============================================================================
# Google Gemini Provider (60 req/min бесплатно)
# =============================================================================

class GoogleProvider(LLMProvider):
    """
    Google Gemini провайдер.

    Бесплатно: 60 req/min, 1500 req/day
    Модели: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash-exp

    Получить ключ: https://aistudio.google.com/apikey
    """

    DEFAULT_MODEL = "gemini-1.5-flash"

    MODELS = {
        "gemini-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-1.5-pro",
        "gemini-2": "gemini-2.0-flash-exp",
        "gemini-2.5-flash": "gemini-2.5-flash-preview-05-20",
    }

    def __init__(self, config: LLMConfig):
        if not config.api_key:
            config.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not config.api_key:
            raise ValueError(
                "GOOGLE_API_KEY не установлен. "
                "Получить бесплатно: https://aistudio.google.com/apikey"
            )
        super().__init__(config)

    def _create_client(self) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = self.config.model
        if model == "auto" or not model:
            model = self.DEFAULT_MODEL
        model = self.MODELS.get(model, model)

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )

    def _get_client_with_overrides(self, temperature, max_tokens) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = self.config.model
        if model == "auto" or not model:
            model = self.DEFAULT_MODEL
        model = self.MODELS.get(model, model)

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.config.api_key,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            timeout=self.config.timeout,
        )


# =============================================================================
# HuggingFace Provider (fallback)
# =============================================================================

class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace Inference API провайдер.

    Получить токен: https://huggingface.co/settings/tokens
    """

    DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

    def __init__(self, config: LLMConfig):
        if not config.api_key:
            config.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not config.api_key:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN не установлен. "
                "Получить: https://huggingface.co/settings/tokens"
            )
        super().__init__(config)

    def _create_client(self) -> BaseChatModel:
        from langchain_huggingface import HuggingFaceEndpoint

        model = self.config.model
        if model == "auto" or not model:
            model = self.DEFAULT_MODEL

        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=self.config.api_key,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )


# =============================================================================
# Provider Stats для отслеживания ошибок
# =============================================================================

@dataclass
class ProviderStats:
    """Статистика провайдера."""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None


# =============================================================================
# Multi-Provider Wrapper с Fallback
# =============================================================================

class MultiProviderWrapper(LLMProvider):
    """
    Обёртка над провайдером с автоматическим fallback на другие провайдеры.

    При выборе основного провайдера (например, Groq), остальные доступные
    провайдеры используются как fallback при ошибках 429/502/503.

    Порядок fallback: Groq → Google → OpenRouter → HuggingFace
    """

    # Порядок fallback провайдеров
    FALLBACK_ORDER = [
        LLMProviderType.GROQ,
        LLMProviderType.GOOGLE,
        LLMProviderType.OPENROUTER,
        LLMProviderType.HUGGINGFACE,
    ]

    # Дефолтные модели
    DEFAULT_MODELS = {
        LLMProviderType.GROQ: "llama-3.1-70b-versatile",
        LLMProviderType.GOOGLE: "gemini-1.5-flash",
        LLMProviderType.OPENROUTER: "meta-llama/llama-3.3-70b-instruct:free",
        LLMProviderType.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.3",
    }

    # Cooldown в секундах
    COOLDOWN_SECONDS = {
        LLMProviderType.GROQ: 65,
        LLMProviderType.GOOGLE: 65,
        LLMProviderType.OPENROUTER: 120,
        LLMProviderType.HUGGINGFACE: 300,
    }

    # Классы провайдеров
    PROVIDER_CLASSES = {
        LLMProviderType.GROQ: GroqProvider,
        LLMProviderType.GOOGLE: GoogleProvider,
        LLMProviderType.OPENROUTER: OpenRouterProvider,
        LLMProviderType.HUGGINGFACE: HuggingFaceProvider,
    }

    def __init__(self, config: LLMConfig, primary_provider: Optional[LLMProviderType] = None):
        """
        Инициализация multi-provider wrapper.

        Args:
            config: Базовая конфигурация
            primary_provider: Основной провайдер (None = auto)
        """
        self.base_config = config
        self.primary_provider = primary_provider
        self.providers: Dict[LLMProviderType, LLMProvider] = {}
        self.stats: Dict[LLMProviderType, ProviderStats] = {}
        self._current_provider: Optional[LLMProviderType] = None

        # Определяем порядок провайдеров
        if primary_provider and primary_provider in self.FALLBACK_ORDER:
            # Основной провайдер первым, остальные как fallback
            self.provider_order = [primary_provider] + [
                p for p in self.FALLBACK_ORDER if p != primary_provider
            ]
        else:
            self.provider_order = self.FALLBACK_ORDER.copy()

        self._init_providers()

    def _init_providers(self):
        """Инициализировать доступные провайдеры."""
        for provider_type in self.provider_order:
            try:
                provider = self._create_single_provider(provider_type)
                self.providers[provider_type] = provider
                self.stats[provider_type] = ProviderStats()
                logger.info(f"✓ {provider_type.value} инициализирован")
            except ValueError as e:
                logger.warning(f"✗ {provider_type.value} недоступен: {e}")
            except ImportError as e:
                logger.warning(f"✗ {provider_type.value} не установлен: {e}")
            except Exception as e:
                logger.warning(f"✗ {provider_type.value} ошибка: {e}")

        if not self.providers:
            raise ValueError(
                "Нет доступных провайдеров! Установите API ключи:\n"
                "- GROQ_API_KEY (https://console.groq.com)\n"
                "- GOOGLE_API_KEY (https://aistudio.google.com/apikey)\n"
                "- OPENROUTER_API_KEY (https://openrouter.ai/keys)"
            )

        logger.info(f"MultiProvider: {len(self.providers)} провайдеров готово")

    def _create_single_provider(self, provider_type: LLMProviderType) -> LLMProvider:
        """Создать один провайдер."""
        provider_class = self.PROVIDER_CLASSES.get(provider_type)
        if not provider_class:
            raise ValueError(f"Неизвестный провайдер: {provider_type}")

        # Используем модель из конфига если это основной провайдер
        if provider_type == self.primary_provider and self.base_config.model != "auto":
            model = self.base_config.model
        else:
            model = self.DEFAULT_MODELS.get(provider_type, "auto")

        config = LLMConfig(
            provider=provider_type,
            model=model,
            temperature=self.base_config.temperature,
            max_tokens=self.base_config.max_tokens,
            timeout=self.base_config.timeout,
        )

        return provider_class(config)

    def _get_available_provider(self) -> Optional[LLMProviderType]:
        """Получить следующий доступный провайдер."""
        now = datetime.now()

        for provider_type in self.provider_order:
            if provider_type not in self.providers:
                continue

            stats = self.stats.get(provider_type)
            if stats and stats.cooldown_until and stats.cooldown_until > now:
                logger.debug(f"{provider_type.value} в cooldown")
                continue

            return provider_type

        # Все в cooldown — возвращаем первый доступный
        for provider_type in self.provider_order:
            if provider_type in self.providers:
                return provider_type

        return None

    def _report_error(self, provider_type: LLMProviderType, error: Exception):
        """Зарегистрировать ошибку."""
        stats = self.stats.setdefault(provider_type, ProviderStats())
        stats.requests_failed += 1
        stats.last_error = str(error)
        stats.last_error_time = datetime.now()

        error_str = str(error).lower()

        # Устанавливаем cooldown в зависимости от ошибки
        if "429" in error_str or "rate limit" in error_str:
            cooldown = self.COOLDOWN_SECONDS.get(provider_type, 60)
            if "per-day" in error_str or "daily" in error_str:
                cooldown = 3600  # 1 час для дневного лимита
            stats.cooldown_until = datetime.now() + timedelta(seconds=cooldown)
            logger.warning(f"{provider_type.value} rate limit, cooldown {cooldown}s")
        elif "404" in error_str:
            stats.cooldown_until = datetime.now() + timedelta(seconds=300)
            logger.warning(f"{provider_type.value} 404 error, cooldown 5min")
        elif "502" in error_str or "503" in error_str:
            stats.cooldown_until = datetime.now() + timedelta(seconds=30)
            logger.warning(f"{provider_type.value} server error, cooldown 30s")

    def _report_success(self, provider_type: LLMProviderType):
        """Зарегистрировать успех."""
        stats = self.stats.setdefault(provider_type, ProviderStats())
        stats.requests_total += 1
        stats.requests_success += 1

    def _create_client(self) -> BaseChatModel:
        """Получить клиент основного провайдера."""
        provider_type = self._get_available_provider()
        if provider_type and provider_type in self.providers:
            return self.providers[provider_type].client
        raise ValueError("Нет доступных провайдеров")

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> str:
        """Генерация с автоматическим fallback."""
        last_error = None
        attempts = 0
        max_attempts = len(self.providers) * 2

        while attempts < max_attempts:
            provider_type = self._get_available_provider()
            if not provider_type:
                break

            provider = self.providers[provider_type]
            attempts += 1

            try:
                logger.debug(f"Попытка {attempts}: {provider_type.value}")
                result = provider.generate(prompt, system_prompt, temperature, max_tokens)
                self._report_success(provider_type)
                self._current_provider = provider_type
                return result

            except Exception as e:
                self._report_error(provider_type, e)
                last_error = e

                # Проверяем, нужен ли fallback
                error_str = str(e).lower()
                if any(x in error_str for x in ["429", "rate limit", "502", "503", "404"]):
                    logger.warning(f"{provider_type.value} ошибка, пробуем fallback...")
                    time.sleep(1)
                    continue
                else:
                    # Другие ошибки — сразу выбрасываем
                    raise

        if last_error:
            raise last_error
        raise Exception("Все провайдеры недоступны")

    def generate_structured(
            self,
            prompt: str,
            output_schema: Type[T],
            system_prompt: Optional[str] = None
    ) -> T:
        """Структурированная генерация с fallback."""
        last_error = None
        attempts = 0
        max_attempts = len(self.providers) * 2

        while attempts < max_attempts:
            provider_type = self._get_available_provider()
            if not provider_type:
                break

            provider = self.providers[provider_type]
            attempts += 1

            try:
                logger.debug(f"Structured попытка {attempts}: {provider_type.value}")
                result = provider.generate_structured(prompt, output_schema, system_prompt)
                self._report_success(provider_type)
                self._current_provider = provider_type
                return result

            except Exception as e:
                self._report_error(provider_type, e)
                last_error = e

                error_str = str(e).lower()
                if any(x in error_str for x in ["429", "rate limit", "502", "503", "404"]):
                    logger.warning(f"{provider_type.value} structured ошибка, fallback...")
                    time.sleep(1)
                    continue
                else:
                    raise

        if last_error:
            raise last_error
        raise Exception("Все провайдеры недоступны")

    def get_current_provider(self) -> Optional[str]:
        """Имя текущего используемого провайдера."""
        return self._current_provider.value if self._current_provider else None

    def get_stats(self) -> Dict[str, Any]:
        """Статистика по провайдерам."""
        return {
            "available": [p.value for p in self.providers.keys()],
            "order": [p.value for p in self.provider_order if p in self.providers],
            "current": self.get_current_provider(),
            "stats": {
                pt.value: {
                    "success": s.requests_success,
                    "failed": s.requests_failed,
                    "in_cooldown": bool(s.cooldown_until and s.cooldown_until > datetime.now())
                }
                for pt, s in self.stats.items()
            }
        }


# =============================================================================
# Factory
# =============================================================================

class LLMProviderFactory:
    """Фабрика LLM провайдеров."""

    _providers = {
        LLMProviderType.OLLAMA: OllamaProvider,
        LLMProviderType.OPENROUTER: OpenRouterProvider,
        LLMProviderType.GROQ: GroqProvider,
        LLMProviderType.GOOGLE: GoogleProvider,
        LLMProviderType.HUGGINGFACE: HuggingFaceProvider,
    }

    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        """Создать провайдер."""
        # AUTO = MultiProvider
        if config.provider == LLMProviderType.AUTO:
            return MultiProviderWrapper(config, primary_provider=None)

        # Конкретный провайдер с fallback
        if config.use_fallback and config.provider in [
            LLMProviderType.GROQ,
            LLMProviderType.GOOGLE,
            LLMProviderType.OPENROUTER,
            LLMProviderType.HUGGINGFACE
        ]:
            return MultiProviderWrapper(config, primary_provider=config.provider)

        # Без fallback (Ollama или явный запрос)
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Неподдерживаемый провайдер: {config.provider}")

        logger.info(f"Создание провайдера: {config.provider.value}, model={config.model}")
        return provider_class(config)


def get_llm_provider(
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        use_fallback: bool = True,
        **kwargs
) -> LLMProvider:
    """
    Создать LLM провайдер.

    Args:
        provider: Провайдер ("auto", "groq", "google", "openrouter", "huggingface", "ollama")
        model: Модель (опционально)
        api_key: API ключ (опционально, берётся из env)
        use_fallback: Использовать fallback на другие провайдеры (default: True)
        **kwargs: Дополнительные параметры

    Returns:
        LLM провайдер

    Examples:
        >>> # Автоматический выбор с fallback
        >>> llm = get_llm_provider("auto")

        >>> # Groq как основной, остальные как fallback
        >>> llm = get_llm_provider("groq")

        >>> # Groq без fallback
        >>> llm = get_llm_provider("groq", use_fallback=False)

        >>> # OpenRouter с конкретной моделью
        >>> llm = get_llm_provider("openrouter", model="meta-llama/llama-3.3-70b-instruct:free")
    """
    provider_type = LLMProviderType(provider.lower())

    # Дефолтные модели
    default_models = {
        LLMProviderType.OLLAMA: "qwen2.5:14b-instruct-q5_k_m",
        LLMProviderType.OPENROUTER: "meta-llama/llama-3.3-70b-instruct:free",
        LLMProviderType.GROQ: "llama-3.1-70b-versatile",
        LLMProviderType.GOOGLE: "gemini-1.5-flash",
        LLMProviderType.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.3",
        LLMProviderType.AUTO: "auto",
    }

    if not model:
        model = default_models.get(provider_type, "auto")

    config = LLMConfig(
        provider=provider_type,
        model=model,
        api_key=api_key,
        use_fallback=use_fallback,
        **kwargs
    )

    return LLMProviderFactory.create(config)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 60)
    print("Multi-Provider LLM Test")
    print("=" * 60)

    try:
        # Тест auto
        print("\n1. Тест AUTO провайдера:")
        llm = get_llm_provider("auto")
        if hasattr(llm, 'get_stats'):
            print(f"   Доступные: {llm.get_stats()['available']}")

        response = llm.generate("Скажи 'привет' одним словом")
        print(f"   Ответ: {response}")
        if hasattr(llm, 'get_current_provider'):
            print(f"   Использован: {llm.get_current_provider()}")

    except Exception as e:
        print(f"Ошибка: {e}")