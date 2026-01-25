# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/config/models_config.py
# =============================================================================
"""
Конфигурация моделей с поддержкой профилей и провайдеров.

Поддерживает:
- Несколько профилей (balanced, fast, cloud_balanced и др.)
- Переключение через env-переменные
- YAML конфигурацию
- Дефолтные значения

Переменные окружения:
- LLM_PROVIDER: Переопределить провайдер (ollama/openrouter)
- LLM_PROFILE: Переопределить профиль
- OPENROUTER_API_KEY: API ключ для OpenRouter
- OLLAMA_BASE_URL: Кастомный URL Ollama
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.infrastructure.ai.llm_provider import LLMConfig, LLMProviderType


class AgentType(str, Enum):
    """Типы AI агентов."""
    CLASSIFIER = "classifier"
    RELEVANCE = "relevance"
    SUMMARIZER = "summarizer"
    REWRITER = "rewriter"
    STYLE_NORMALIZER = "style_normalizer"
    QUALITY_VALIDATOR = "quality_validator"


@dataclass
class AgentConfig:
    """Конфигурация одного агента."""
    model: str
    temperature: float
    max_tokens: int
    provider: LLMProviderType = LLMProviderType.OLLAMA

    def to_llm_config(
            self,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None
    ) -> LLMConfig:
        """Преобразовать в LLMConfig."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=base_url,
            api_key=api_key
        )


@dataclass
class ProfileConfig:
    """Профиль конфигурации со всеми агентами."""
    name: str
    provider: LLMProviderType
    agents: Dict[AgentType, AgentConfig] = field(default_factory=dict)


class ModelsConfig:
    """
    Главный класс конфигурации моделей.

    Приоритет загрузки:
    1. Переменные окружения (высший приоритет)
    2. YAML файл конфигурации
    3. Встроенные дефолты (низший приоритет)
    """

    # Встроенные профили
    DEFAULT_PROFILES = {
        "balanced": {
            "provider": "ollama",
            "agents": {
                "classifier": {"model": "qwen2.5:14b-instruct-q5_k_m", "temperature": 0.3, "max_tokens": 100},
                "relevance": {"model": "qwen2.5:14b-instruct-q5_k_m", "temperature": 0.4, "max_tokens": 300},
                "summarizer": {"model": "qwen2.5:14b-instruct-q5_k_m", "temperature": 0.5, "max_tokens": 300},
                "rewriter": {"model": "qwen2.5:14b-instruct-q5_k_m", "temperature": 0.6, "max_tokens": 200},
                "style_normalizer": {"model": "qwen2.5:14b-instruct-q5_k_m", "temperature": 0.3, "max_tokens": 8000},
                "quality_validator": {"model": "qwen2.5:14b-instruct-q5_k_m", "temperature": 0.2, "max_tokens": 500},
            }
        },
        "fast": {
            "provider": "ollama",
            "agents": {
                "classifier": {"model": "mistral:latest", "temperature": 0.3, "max_tokens": 100},
                "relevance": {"model": "mistral:latest", "temperature": 0.4, "max_tokens": 300},
                "summarizer": {"model": "mistral:latest", "temperature": 0.5, "max_tokens": 300},
                "rewriter": {"model": "mistral:latest", "temperature": 0.6, "max_tokens": 200},
                "style_normalizer": {"model": "qwen2.5:7b", "temperature": 0.3, "max_tokens": 4000},
                "quality_validator": {"model": "mistral:latest", "temperature": 0.2, "max_tokens": 500},
            }
        },
        # =========================================================================
        # БЕСПЛАТНЫЙ ПРОФИЛЬ - OpenRouter с LiquidAI (FREE!)
        # =========================================================================
        "free_openrouter": {
            "provider": "openrouter",
            "agents": {
                "classifier": {"model": "liquid/lfm-2.5-1.2b-instruct", "temperature": 0.3, "max_tokens": 100},
                "relevance": {"model": "liquid/lfm-2.5-1.2b-instruct", "temperature": 0.4, "max_tokens": 300},
                "summarizer": {"model": "liquid/lfm-2.5-1.2b-instruct", "temperature": 0.5, "max_tokens": 400},
                "rewriter": {"model": "liquid/lfm-2.5-1.2b-instruct", "temperature": 0.6, "max_tokens": 300},
                "style_normalizer": {"model": "liquid/lfm-2.5-1.2b-instruct", "temperature": 0.3, "max_tokens": 4000},
                "quality_validator": {"model": "liquid/lfm-2.5-1.2b-instruct", "temperature": 0.2, "max_tokens": 500},
            }
        },
        "cloud_balanced": {
            "provider": "openrouter",
            "agents": {
                "classifier": {"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 100},
                "relevance": {"model": "gpt-4o-mini", "temperature": 0.4, "max_tokens": 300},
                "summarizer": {"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 300},
                "rewriter": {"model": "gpt-4o-mini", "temperature": 0.6, "max_tokens": 200},
                "style_normalizer": {"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 4000},
                "quality_validator": {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 500},
            }
        },
        "cloud_quality": {
            "provider": "openrouter",
            "agents": {
                "classifier": {"model": "gpt-4o", "temperature": 0.2, "max_tokens": 100},
                "relevance": {"model": "gpt-4o", "temperature": 0.3, "max_tokens": 500},
                "summarizer": {"model": "claude-3.5-sonnet", "temperature": 0.4, "max_tokens": 500},
                "rewriter": {"model": "claude-3.5-sonnet", "temperature": 0.5, "max_tokens": 300},
                "style_normalizer": {"model": "claude-3.5-sonnet", "temperature": 0.2, "max_tokens": 8000},
                "quality_validator": {"model": "gpt-4o", "temperature": 0.1, "max_tokens": 500},
            }
        },
    }

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config_path = config_path
        self._raw_config = self._load_config()

        # Получить активный профиль из env или конфига
        self.active_profile = os.getenv(
            "LLM_PROFILE",
            self._raw_config.get("active_profile", "balanced")
        )

        # Переопределение провайдера из env
        self._provider_override = os.getenv("LLM_PROVIDER")

        # API ключи и URL из env
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    def _load_config(self) -> Dict[str, Any]:
        """Загрузить конфигурацию из YAML файла."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:  # ИСПРАВЛЕНО: было "FileNotFoundОшибка"
            print(f"⚠️  Конфиг {self.config_path} не найден, используем дефолты")
            return {}
        except Exception as e:
            print(f"⚠️  Ошибка чтения конфига: {e}, используем дефолты")
            return {}

    def get_profile(self) -> ProfileConfig:
        """Получить текущий профиль конфигурации."""
        # Попробовать загрузить из YAML
        profiles = self._raw_config.get("profiles", {})

        if self.active_profile in profiles:
            profile_data = profiles[self.active_profile]
        elif self.active_profile in self.DEFAULT_PROFILES:
            profile_data = self.DEFAULT_PROFILES[self.active_profile]
        else:
            print(f"⚠️  Профиль '{self.active_profile}' не найден, используем 'balanced'")
            profile_data = self.DEFAULT_PROFILES["balanced"]

        # Распарсить провайдер
        provider_str = self._provider_override or profile_data.get("provider", "ollama")
        provider = LLMProviderType(provider_str.lower())

        # Распарсить агентов
        agents = {}
        for agent_name, agent_data in profile_data.get("agents", {}).items():  # ИСПРАВЛЕНО: было "элементов()"
            agent_type = AgentType(agent_name)
            agents[agent_type] = AgentConfig(
                model=agent_data["model"],
                temperature=agent_data["temperature"],
                max_tokens=agent_data["max_tokens"],
                provider=provider
            )

        return ProfileConfig(
            name=self.active_profile,
            provider=provider,
            agents=agents
        )

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Получить конфигурацию для конкретного агента."""
        profile = self.get_profile()
        agent_type = AgentType(agent_name)

        if agent_type not in profile.agents:
            # Вернуть дефолтный конфиг
            return AgentConfig(
                model="mistral:latest",
                temperature=0.5,
                max_tokens=500,
                provider=profile.provider
            )

        return profile.agents[agent_type]

    def get_llm_config(self, agent_name: str) -> LLMConfig:
        """Получить LLMConfig для конкретного агента."""
        agent_config = self.get_agent_config(agent_name)
        profile = self.get_profile()

        # Определить base_url и api_key по провайдеру
        if profile.provider == LLMProviderType.OPENROUTER:
            base_url = LLMConfig.OPENROUTER_DEFAULT_URL
            api_key = self.openrouter_api_key
        else:
            base_url = self.ollama_base_url
            api_key = None

        return agent_config.to_llm_config(base_url=base_url, api_key=api_key)

    def get_provider(self) -> LLMProviderType:
        """Получить текущий тип провайдера."""
        return self.get_profile().provider

    # Методы для обратной совместимости
    def get_model(self, agent_name: str) -> str:
        """Получить модель для агента."""
        return self.get_agent_config(agent_name).model

    def get_temperature(self, agent_name: str) -> float:
        """Получить температуру для агента."""
        return self.get_agent_config(agent_name).temperature

    def get_max_tokens(self, agent_name: str) -> int:
        """Получить max_tokens для агента."""
        return self.get_agent_config(agent_name).max_tokens

    def print_config(self):
        """Вывести текущую конфигурацию."""
        profile = self.get_profile()

        print(f"\n{'=' * 70}")
        print(f"📋 АКТИВНЫЙ ПРОФИЛЬ: {profile.name}")
        print(f"🔌 ПРОВАЙДЕР: {profile.provider.value}")
        print(f"{'=' * 70}\n")

        for agent_type, agent_config in profile.agents.items():  # ИСПРАВЛЕНО: было "элементов()"
            print(
                f"🤖 {agent_type.value:20} → {agent_config.model:30} "
                f"(T={agent_config.temperature}, tokens={agent_config.max_tokens})"
            )

        print(f"\n{'=' * 70}\n")

    @classmethod
    def get_available_profiles(cls) -> list[str]:
        """Получить список доступных встроенных профилей."""
        return list(cls.DEFAULT_PROFILES.keys())


# Глобальный инстанс для удобства
_config_instance: Optional[ModelsConfig] = None


def get_models_config(config_path: str = "config/models.yaml") -> ModelsConfig:
    """Получить или создать глобальный инстанс ModelsConfig."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ModelsConfig(config_path)
    return _config_instance