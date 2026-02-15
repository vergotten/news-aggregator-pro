"""
Модуль провайдеров LLM с автоматическим выбором моделей v5.0

Динамическая приоритизация по размеру модели:
- HEAVY задачи → большие модели первые (сортировка по убыванию размера)
- LIGHT задачи → маленькие модели первые (сортировка по возрастанию размера)
- MEDIUM задачи → по контексту (по умолчанию)

БЕЗ ХАРДКОДА моделей - размер извлекается из названия модели.
"""

import json
import logging
import os
import re
import time
import random
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Dict, Any, Optional, Type, TypeVar, Union, List

from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


# =============================================================================
# Перечисления и конфигурация
# =============================================================================

class LLMProviderType(str, Enum):
    """
    Типы поддерживаемых LLM провайдеров.
    """
    OPENROUTER = "openrouter"
    GROQ = "groq"
    GOOGLE = "google"
    OLLAMA = "ollama"


class TaskType(str, Enum):
    """
    Типы задач для выбора оптимальной модели.

    HEAVY: Сложные задачи → большие модели первые (по убыванию размера)
    LIGHT: Простые задачи → маленькие модели первые (по возрастанию размера)
    MEDIUM: Средние задачи → по контексту (по умолчанию)
    """
    HEAVY = "heavy"
    MEDIUM = "medium"
    LIGHT = "light"


@dataclass
class LLMConfig:
    """
    Конфигурация LLM провайдера.
    """
    provider: LLMProviderType
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    context_length: int = 131072

    OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"
    GROQ_DEFAULT_URL = "https://api.groq.com/openai/v1"
    GOOGLE_DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta"
    OLLAMA_DEFAULT_URL = "http://ollama:11434"

    def __post_init__(self):
        if isinstance(self.provider, str):
            self.provider = LLMProviderType(self.provider.lower())

    def get_base_url(self) -> str:
        if self.base_url:
            return self.base_url

        defaults = {
            LLMProviderType.OPENROUTER: self.OPENROUTER_DEFAULT_URL,
            LLMProviderType.GROQ: self.GROQ_DEFAULT_URL,
            LLMProviderType.GOOGLE: self.GOOGLE_DEFAULT_URL,
            LLMProviderType.OLLAMA: self.OLLAMA_DEFAULT_URL,
        }
        return defaults.get(self.provider, "")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        try:
            provider = data.get("provider", "ollama")
            if isinstance(provider, str):
                provider = LLMProviderType(provider.lower())

            # Для ollama: если model не задан явно, берём из env → дефолт
            default_model = (
                os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q5_k_m")
                if provider == LLMProviderType.OLLAMA
                else "qwen2.5:14b-instruct-q5_k_m"
            )

            return cls(
                provider=provider,
                model=data.get("model") or default_model,
                temperature=data.get("temperature", 0.7),
                max_tokens=data.get("max_tokens", 4096),
                api_key=data.get("api_key"),
                base_url=data.get("base_url"),
                context_length=data.get("context_length", 131072),
            )
        except Exception as e:
            logger.error(f"Ошибка создания LLMConfig: {e}")
            return cls(
                provider=LLMProviderType.OLLAMA,
                model=os.getenv("OLLAMA_MODEL", "glm-4.7-flash:q4_K_M"),
            )


# =============================================================================
# Обнаружение моделей OpenRouter
# =============================================================================

@dataclass
class FreeModel:
    """
    Информация о бесплатной модели OpenRouter.
    """
    id: str
    name: str
    context_length: int
    max_output: int
    capabilities: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"FreeModel({self.id}, ctx={self.context_length})"


class OpenRouterModelDiscovery:
    """
    Динамическое обнаружение бесплатных моделей OpenRouter.

    Поддерживает сортировку по размеру модели для TaskType.
    """

    _instance = None
    _lock = Lock()

    API_URL = "https://openrouter.ai/api/v1/models"
    CACHE_TTL_SECONDS = 3600

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self._models: List[FreeModel] = []
        self._last_fetch: Optional[datetime] = None
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        self._initialized = True
        logger.info("OpenRouterModelDiscovery инициализирован")

    def get_model_size(self, model_id: str) -> int:
        """
        Извлечь размер модели из её ID (в миллиардах параметров).

        Примеры:
            qwen/qwen3-235b → 235
            meta-llama/llama-3.3-70b-instruct → 70
            google/gemma-3-4b-it → 4
            deepseek/deepseek-chat → 100 (дефолт для известных больших)

        Returns:
            Размер в миллиардах (int)
        """
        model_lower = model_id.lower()

        # Ищем паттерны размера: 235b, 70b, 4b, 3.5b и т.д.
        patterns = [
            r'(\d+\.?\d*)b(?:-|:|$|_|\.)',  # 70b-, 235b:, 4b_, 70b.
            r'-(\d+\.?\d*)b',                # -70b
            r'(\d+\.?\d*)b-',                # 70b-
            r'[/-](\d+)b',                   # /70b или -70b
        ]

        for pattern in patterns:
            match = re.search(pattern, model_lower)
            if match:
                try:
                    size = float(match.group(1))
                    return int(size)
                except:
                    pass

        # Известные большие модели без размера в названии
        big_models = {
            'deepseek-chat': 100,
            'deepseek-v3': 100,
            'gpt-4': 100,
            'claude': 100,
            'gemini-pro': 50,
            'gemini-2': 50,
            'gemini-flash': 30,
        }
        for name, size in big_models.items():
            if name in model_lower:
                return size

        # Дефолт для неизвестных
        return 20

    def get_models_sorted_by_size(
            self,
            ascending: bool = True,
            min_context: int = 4000
    ) -> List[FreeModel]:
        """
        Получить модели отсортированные по размеру.

        Args:
            ascending: True = маленькие первые, False = большие первые
            min_context: Минимальный контекст

        Returns:
            Отсортированный список моделей
        """
        models = self.get_free_models(min_context=min_context)

        if not models:
            return []

        # Сортируем по размеру
        sorted_models = sorted(
            models,
            key=lambda m: self.get_model_size(m.id),
            reverse=not ascending
        )

        return sorted_models

    def get_free_models(
            self,
            min_context: int = 4000,
            force_refresh: bool = False
    ) -> List[FreeModel]:
        """
        Получить список бесплатных моделей.
        """
        try:
            if not force_refresh and self._models and self._last_fetch:
                age = (datetime.now() - self._last_fetch).total_seconds()
                if age < self.CACHE_TTL_SECONDS:
                    logger.debug(f"Используем кэш моделей: {len(self._models)} моделей")
                    return [m for m in self._models if m.context_length >= min_context]

            logger.info("Запрос списка моделей с OpenRouter API...")
            self._fetch_models()

            filtered = [m for m in self._models if m.context_length >= min_context]
            logger.info(f"Получено {len(self._models)} моделей, отфильтровано: {len(filtered)}")

            return filtered

        except Exception as e:
            logger.error(f"Ошибка получения моделей: {e}")
            return []

    def get_best_model(self, min_context: int = 4000) -> Optional[FreeModel]:
        """Получить лучшую бесплатную модель."""
        try:
            models = self.get_free_models(min_context=min_context)
            return models[0] if models else None
        except Exception as e:
            logger.error(f"Ошибка получения лучшей модели: {e}")
            return None

    def _fetch_models(self) -> None:
        """Запросить модели с API OpenRouter."""
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                logger.info(f"Попытка {attempt + 1}/{max_retries} запроса моделей...")

                response = requests.get(
                    self.API_URL,
                    headers=headers,
                    timeout=60
                )

                if response.status_code != 200:
                    if response.status_code == 408 and attempt < max_retries - 1:
                        logger.warning(f"Таймаут, повтор через {retry_delay}с...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                    logger.error(f"Ошибка API: {response.status_code}")
                    if attempt == max_retries - 1:
                        self._use_fallback_models()
                    return

                data = response.json()
                models = []

                for item in data.get("data", []):
                    try:
                        pricing = item.get("pricing", {})
                        prompt_price = float(pricing.get("prompt", "1") or "1")
                        completion_price = float(pricing.get("completion", "1") or "1")

                        if prompt_price == 0 and completion_price == 0:
                            model = FreeModel(
                                id=item.get("id", ""),
                                name=item.get("name", ""),
                                context_length=item.get("context_length", 4096),
                                max_output=item.get("top_provider", {}).get("max_completion_tokens", 4096),
                                capabilities=self._extract_capabilities(item),
                            )
                            models.append(model)
                    except Exception as e:
                        logger.warning(f"Ошибка обработки модели {item.get('id')}: {e}")

                models.sort(key=lambda m: (-m.context_length, m.name))

                self._models = models
                self._last_fetch = datetime.now()

                logger.info(f"Успешно получено {len(models)} моделей")
                return

            except requests.exceptions.RequestException as e:
                if "timeout" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Сетевая ошибка, повтор через {retry_delay}с...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                logger.error(f"Сетевая ошибка: {e}")
                if attempt == max_retries - 1:
                    self._use_fallback_models()
                return
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {e}")
                if attempt == max_retries - 1:
                    self._use_fallback_models()
                return

        logger.error("Все попытки запроса моделей провалены")
        self._use_fallback_models()

    def _extract_capabilities(self, item: Dict) -> List[str]:
        """Извлечь возможности модели из данных API."""
        try:
            caps = ["chat"]

            arch = item.get("architecture", {})
            modality = arch.get("modality", "")
            if "image" in modality.lower():
                caps.append("vision")

            params = item.get("supported_parameters", [])
            if params and ("tools" in params or "functions" in params):
                caps.append("function_calling")

            return caps
        except Exception as e:
            logger.warning(f"Ошибка извлечения возможностей: {e}")
            return ["chat"]

    def _use_fallback_models(self) -> None:
        """Использовать предопределенный список моделей при ошибке API."""
        logger.warning("Используем резервный список моделей")
        self._models = [
            FreeModel("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash", 1048576, 8192, ["chat", "vision"]),
            FreeModel("meta-llama/llama-4-scout:free", "Llama 4 Scout", 524288, 16384, ["chat"]),
            FreeModel("meta-llama/llama-4-maverick:free", "Llama 4 Maverick", 131072, 16384, ["chat", "vision"]),
            FreeModel("deepseek/deepseek-chat-v3-0324:free", "DeepSeek Chat V3", 131072, 8192, ["chat"]),
            FreeModel("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1", 131072, 8192, ["chat"]),
            FreeModel("qwen/qwen2.5-72b-instruct:free", "Qwen 2.5 72B", 131072, 8192, ["chat"]),
            FreeModel("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B", 131072, 8192, ["chat"]),
            FreeModel("meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B", 131072, 4096, ["chat"]),
        ]
        self._last_fetch = datetime.now()


# =============================================================================
# Отслеживание состояния моделей
# =============================================================================

@dataclass
class ModelStatus:
    """Статус модели для отслеживания rate-limit."""
    model_id: str
    errors: int = 0
    cooldown_until: Optional[datetime] = None
    last_success: Optional[datetime] = None

    @property
    def is_available(self) -> bool:
        if self.cooldown_until is None:
            return True
        return datetime.now() >= self.cooldown_until

    def record_error(self) -> None:
        try:
            self.errors += 1
            cooldown = min(30 * (2 ** (self.errors - 1)), 600)
            jitter = cooldown * 0.2 * (random.random() * 2 - 1)
            cooldown += jitter

            self.cooldown_until = datetime.now() + timedelta(seconds=cooldown)
            logger.warning(f"Модель {self.model_id}: cooldown {cooldown:.0f}с (ошибок: {self.errors})")
        except Exception as e:
            logger.error(f"Ошибка записи ошибки модели: {e}")

    def record_success(self) -> None:
        try:
            self.errors = 0
            self.cooldown_until = None
            self.last_success = datetime.now()
        except Exception as e:
            logger.error(f"Ошибка записи успеха модели: {e}")


class ModelStatusTracker:
    """Синглтон для отслеживания статуса всех моделей."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._statuses = {}
        return cls._instance

    def get(self, model_id: str) -> ModelStatus:
        if model_id not in self._statuses:
            self._statuses[model_id] = ModelStatus(model_id=model_id)
        return self._statuses[model_id]

    def is_available(self, model_id: str) -> bool:
        try:
            return self.get(model_id).is_available
        except Exception as e:
            logger.error(f"Ошибка проверки доступности модели {model_id}: {e}")
            return False

    def record_error(self, model_id: str) -> None:
        try:
            self.get(model_id).record_error()
        except Exception as e:
            logger.error(f"Ошибка записи ошибки модели {model_id}: {e}")

    def record_success(self, model_id: str) -> None:
        try:
            self.get(model_id).record_success()
        except Exception as e:
            logger.error(f"Ошибка записи успеха модели {model_id}: {e}")

    def get_available_models(self, model_ids: List[str]) -> List[str]:
        try:
            return [m for m in model_ids if self.is_available(m)]
        except Exception as e:
            logger.error(f"Ошибка получения доступных моделей: {e}")
            return []


# =============================================================================
# Базовый провайдер
# =============================================================================

class LLMProvider(ABC):
    """Абстрактный базовый класс для LLM провайдеров."""

    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        try:
            if isinstance(config, dict):
                self.config = LLMConfig.from_dict(config)
            else:
                self.config = config

            self.provider_name = self.config.provider.value
            self.model = self.config.model
            self.api_key = self.config.api_key
            self.base_url = self.config.get_base_url()

            self._request_count = 0
            self._error_count = 0

            logger.info(f"Провайдер {self.provider_name} инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации провайдера: {e}")
            raise

    @abstractmethod
    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> str:
        pass

    def generate_structured(
            self,
            prompt: str,
            output_schema: Type[T],
            system_prompt: Optional[str] = None,
            max_retries: int = 3,
            **kwargs
    ) -> T:
        """Генерация структурированного ответа."""
        last_error = None

        for attempt in range(max_retries):
            try:
                example_json = self._build_example_json(output_schema)

                json_prompt = f"""{prompt}

ВАЖНО: Ответь ТОЛЬКО валидным JSON объектом. Никакого текста до или после JSON.
Заполни все поля своими реальными значениями по результатам анализа.

Формат ответа (заполни своими значениями):
{example_json}

Начни ответ с {{ и закончи }}. Без markdown, без ```. Только JSON."""

                temp = 0.1 if attempt >= max_retries - 1 else 0.2

                response = self.generate(
                    json_prompt,
                    system_prompt,
                    temperature=temp,
                    **kwargs
                )

                json_data = self._extract_json(response)
                if json_data:
                    json_data = self._clean_schema_artifacts(json_data, output_schema)
                    return output_schema(**json_data)

                last_error = ValueError("Не найден валидный JSON в ответе")

            except Exception as e:
                last_error = e
                logger.warning(f"Попытка {attempt + 1} структурированного вывода не удалась: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        raise ValueError(f"Не удалось получить структурированный ответ: {last_error}")

    def _build_example_json(self, schema: Type[BaseModel]) -> str:
        """Построить JSON-пример на основе Pydantic схемы."""
        try:
            try:
                fields = schema.model_fields
            except AttributeError:
                fields = schema.__fields__

            example = {}
            for field_name, field_info in fields.items():
                try:
                    annotation = field_info.annotation
                    description = field_info.description or ""
                except AttributeError:
                    annotation = field_info.outer_type_
                    description = field_info.field_info.description or ""

                example[field_name] = self._example_value_for_type(
                    field_name, annotation, description
                )

            return json.dumps(example, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка построения примера JSON: {e}")
            return "{}"

    def _example_value_for_type(self, name: str, annotation, description: str) -> Any:
        """Сгенерировать пример значения для поля."""
        try:
            type_str = str(annotation).lower() if annotation else ""

            if annotation is int or "int" in type_str:
                if "score" in name or "оценк" in name.lower():
                    return 7
                return 5

            if annotation is float or "float" in type_str:
                return 0.8

            if annotation is bool or "bool" in type_str:
                return True

            if "list" in type_str:
                if "categor" in name or "катег" in name.lower():
                    return ["AI/ML", "DevOps"]
                if "tag" in name:
                    return ["python", "machine-learning"]
                return ["пример1", "пример2"]

            if annotation is str or "str" in type_str:
                if "reason" in name or "причин" in name.lower():
                    return "ваше объяснение здесь"
                if "audience" in name or "аудитор" in name.lower():
                    return "developers"
                if "categor" in name:
                    return "Technology"
                if description:
                    return f"<{description[:40]}>"
                return f"<заполните {name}>"

            return f"<{name}>"
        except Exception as e:
            logger.error(f"Ошибка генерации примера значения: {e}")
            return "<значение>"

    def _clean_schema_artifacts(self, data: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """Очистить данные от артефактов JSON Schema."""
        try:
            try:
                expected_fields = set(schema.model_fields.keys())
            except AttributeError:
                expected_fields = set(schema.__fields__.keys())

            schema_meta_fields = {
                "description", "title", "type", "properties",
                "required", "$defs", "definitions", "additionalProperties"
            }

            data_keys = set(data.keys())
            is_schema_response = (
                    data_keys & schema_meta_fields and
                    not (data_keys & expected_fields)
            )

            if is_schema_response:
                if "properties" in data:
                    props = data["properties"]
                    extracted = {}
                    for key, val in props.items():
                        if key in expected_fields:
                            if isinstance(val, dict) and "default" in val:
                                extracted[key] = val["default"]
                            elif isinstance(val, dict) and "example" in val:
                                extracted[key] = val["example"]
                            else:
                                extracted[key] = val
                    if extracted:
                        logger.warning("Модель вернула JSON Schema, извлекаем данные")
                        return extracted

                logger.warning("Модель вернула JSON Schema, не удалось извлечь данные")

            cleaned = {
                k: v for k, v in data.items()
                if k not in schema_meta_fields or k in expected_fields
            }

            if "content_structure" in cleaned and isinstance(cleaned["content_structure"], str):
                try:
                    if cleaned["content_structure"].strip().startswith("{"):
                        cleaned["content_structure"] = json.loads(cleaned["content_structure"])
                    else:
                        sections = cleaned["content_structure"].split(",")
                        cleaned["content_structure"] = {
                            "sections": [section.strip() for section in sections]
                        }
                except:
                    cleaned["content_structure"] = {"raw": cleaned["content_structure"]}

            return cleaned
        except Exception as e:
            logger.error(f"Ошибка очистки данных: {e}")
            return data

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Извлечь JSON из текстового ответа."""
        if not text:
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        for block in json_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        code_blocks = re.findall(r'```\s*([\s\S]*?)\s*```', text)
        for block in code_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        start = text.find('{')
        if start != -1:
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        try:
            cleaned = re.sub(r'^[^{]*', '', text)
            cleaned = re.sub(r'}[^}]*$', '}', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "requests": self._request_count,
            "errors": self._error_count,
        }


# =============================================================================
# OpenRouter провайдер с автоматическим fallback и приоритизацией по задачам
# =============================================================================

class OpenRouterProvider(LLMProvider):
    """
    Провайдер OpenRouter с динамической приоритизацией моделей.

    Поддерживает TaskType для автоматической сортировки:
    - HEAVY → большие модели первые
    - LIGHT → маленькие модели первые
    """

    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        super().__init__(config)

        try:
            self.api_base = self.base_url or LLMConfig.OPENROUTER_DEFAULT_URL
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/news-aggregator",
                "X-Title": "News Aggregator Pro"
            }

            self._discovery = OpenRouterModelDiscovery()
            self._tracker = ModelStatusTracker()

            self._fallback_count = 0
            self._current_model = self.model

            logger.info(f"OpenRouter провайдер инициализирован с моделью: {self.model}")
        except Exception as e:
            logger.error(f"Ошибка инициализации OpenRouter провайдера: {e}")
            raise

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> str:
        """Сгенерировать текст с автоматическим fallback."""
        return self._generate_internal(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            task_type=None
        )

    def generate_for_task(
            self,
            prompt: str,
            task_type: TaskType,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> str:
        """
        Генерация с учётом типа задачи.

        Args:
            prompt: Промпт
            task_type: Тип задачи (HEAVY, MEDIUM, LIGHT)
            system_prompt: Системный промпт
            temperature: Температура
            max_tokens: Макс. токенов

        Returns:
            Сгенерированный текст
        """
        return self._generate_internal(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            task_type=task_type
        )

    def _generate_internal(
            self,
            prompt: str,
            system_prompt: Optional[str],
            temperature: Optional[float],
            max_tokens: Optional[int],
            task_type: Optional[TaskType]
    ) -> str:
        """Внутренний метод генерации."""
        self._request_count += 1

        try:
            temp = temperature or self.config.temperature
            tokens = max_tokens or self.config.max_tokens

            models_to_try = self._get_models_to_try(task_type=task_type)

            if not models_to_try:
                raise Exception("Нет доступных моделей - все в cooldown")

            task_name = task_type.value if task_type else "default"
            logger.info(f"🎯 [{task_name}] Попытка с {len(models_to_try)} моделями")

            tried = []
            last_error = None
            base_delay = 1
            max_delay = 10

            for i, model_id in enumerate(models_to_try):
                if model_id in tried:
                    continue

                tried.append(model_id)

                try:
                    if i > 0:
                        delay = min(base_delay * (2 ** (i - 1)), max_delay)
                        logger.debug(f"Задержка {delay:.2f}с перед попыткой {i + 1}")
                        time.sleep(delay)

                    result = self._make_request(
                        model_id, prompt, system_prompt, temp, tokens
                    )

                    self._tracker.record_success(model_id)
                    self._current_model = model_id

                    size = self._discovery.get_model_size(model_id)
                    logger.info(f"✅ [{task_name}] Успех: {model_id} ({size}B)")
                    return result

                except Exception as e:
                    last_error = e
                    error_str = str(e)

                    if any(code in error_str for code in ["429", "402", "403"]):
                        self._tracker.record_error(model_id)
                        logger.warning(f"⚠️ {model_id} rate-limited")
                        continue

                    if any(code in error_str for code in ["500", "502", "503"]):
                        logger.warning(f"⚠️ {model_id} server error")
                        time.sleep(2)
                        continue

                    logger.warning(f"⚠️ {model_id}: {str(e)[:60]}")
                    continue

            self._error_count += 1
            raise Exception(f"Все модели отказали [{task_name}]. Последняя: {last_error}")

        except Exception as e:
            self._error_count += 1
            logger.error(f"❌ Критическая ошибка: {e}")
            raise

    def _get_models_to_try(self, task_type: Optional[TaskType] = None) -> List[str]:
        """
        Собрать список моделей отсортированных по приоритету для задачи.

        Args:
            task_type: Тип задачи
                - HEAVY → большие модели первые
                - LIGHT → маленькие модели первые
                - MEDIUM/None → по контексту (дефолт)

        Returns:
            Список ID моделей в порядке приоритета
        """
        try:
            # Определяем порядок сортировки
            if task_type == TaskType.HEAVY:
                ascending = False  # Большие первые
                task_name = "HEAVY"
            elif task_type == TaskType.LIGHT:
                ascending = True   # Маленькие первые
                task_name = "LIGHT"
            else:
                ascending = None   # По контексту
                task_name = "MEDIUM"

            # Получаем отсортированные модели
            if ascending is not None:
                all_models = self._discovery.get_models_sorted_by_size(ascending=ascending)
            else:
                all_models = self._discovery.get_free_models()

            # Фильтруем доступные (не в cooldown)
            models = []
            for m in all_models:
                if self._tracker.is_available(m.id):
                    models.append(m.id)

            # Логируем приоритет
            logger.info(f"📋 [{task_name}] Модели по приоритету ({len(models)} доступно):")
            for i, model_id in enumerate(models[:5]):
                size = self._discovery.get_model_size(model_id)
                logger.info(f"   {i+1}. {model_id} ({size}B)")
            if len(models) > 5:
                logger.info(f"   ... и ещё {len(models) - 5}")

            return models

        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            return [self.model]

    def _make_request(
            self,
            model: str,
            prompt: str,
            system_prompt: Optional[str],
            temperature: float,
            max_tokens: int
    ) -> str:
        """Выполнить HTTP запрос к API."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=180
            )

            if response.status_code != 200:
                error_msg = f"OpenRouter {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                raise Exception(error_msg)

            result = response.json()
            if "error" in result:
                error_msg = f"Ошибка OpenRouter: {result['error']}"
                logger.error(error_msg)
                raise Exception(error_msg)

            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            raise Exception(f"Таймаут запроса к модели {model}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Сетевая ошибка при запросе к {model}: {e}")
        except Exception as e:
            raise

    def get_metrics(self) -> Dict[str, Any]:
        try:
            metrics = super().get_metrics()
            metrics.update({
                "current_model": self._current_model,
                "fallbacks": self._fallback_count,
            })
            return metrics
        except Exception as e:
            logger.error(f"Ошибка получения метрик: {e}")
            return {}

    def get_available_models(self) -> List[str]:
        try:
            free_models = self._discovery.get_free_models()
            return [m.id for m in free_models if self._tracker.is_available(m.id)]
        except Exception as e:
            logger.error(f"Ошибка получения доступных моделей: {e}")
            return []

    def print_models(self, task_type: Optional[TaskType] = None) -> None:
        """Вывести список моделей с их статусом."""
        try:
            if task_type == TaskType.HEAVY:
                models = self._discovery.get_models_sorted_by_size(ascending=False)
                title = "HEAVY (большие первые)"
            elif task_type == TaskType.LIGHT:
                models = self._discovery.get_models_sorted_by_size(ascending=True)
                title = "LIGHT (маленькие первые)"
            else:
                models = self._discovery.get_free_models()
                title = "по контексту"

            print(f"\n{'=' * 70}")
            print(f"🆓 МОДЕЛИ [{title}] ({len(models)} доступно)")
            print(f"{'=' * 70}")

            for i, m in enumerate(models[:15], 1):
                status = "✓" if self._tracker.is_available(m.id) else "⏳"
                size = self._discovery.get_model_size(m.id)
                ctx = f"{m.context_length // 1000}k"
                caps = ""
                if "vision" in m.capabilities:
                    caps += "👁"
                if "function_calling" in m.capabilities:
                    caps += "🔧"
                print(f"{i:2}. {status} {m.id:<50} {size:>3}B ctx={ctx} {caps}")

            print(f"{'=' * 70}\n")
        except Exception as e:
            logger.error(f"Ошибка вывода моделей: {e}")


# =============================================================================
# Остальные провайдеры
# =============================================================================

class GroqProvider(LLMProvider):
    """Провайдер Groq с OpenAI-совместимым API."""

    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        super().__init__(config)
        try:
            self.api_base = self.base_url or LLMConfig.GROQ_DEFAULT_URL
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            logger.info("Groq провайдер инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Groq провайдера: {e}")
            raise

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> str:
        try:
            self._request_count += 1

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
            }

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=60
            )

            if response.status_code != 200:
                error_msg = f"Ошибка Groq: {response.status_code} {response.text}"
                logger.error(error_msg)
                self._error_count += 1
                raise Exception(error_msg)

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            self._error_count += 1
            logger.error(f"Ошибка генерации Groq: {e}")
            raise


class GoogleProvider(LLMProvider):
    """Провайдер Google Gemini с нативным API."""

    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        super().__init__(config)
        try:
            self.api_base = self.base_url or LLMConfig.GOOGLE_DEFAULT_URL
            logger.info("Google провайдер инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Google провайдера: {e}")
            raise

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> str:
        try:
            self._request_count += 1

            url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"

            contents = []
            if system_prompt:
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {system_prompt}"}]
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": "Understood."}]
                })
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })

            data = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature or self.config.temperature,
                    "maxOutputTokens": max_tokens or self.config.max_tokens,
                }
            }

            response = requests.post(url, json=data, timeout=120)

            if response.status_code != 200:
                error_msg = f"Ошибка Google: {response.status_code} {response.text}"
                logger.error(error_msg)
                self._error_count += 1
                raise Exception(error_msg)

            result = response.json()
            if "candidates" not in result:
                raise Exception(f"Google не вернул кандидатов: {result}")

            return result["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            self._error_count += 1
            logger.error(f"Ошибка генерации Google: {e}")
            raise


class OllamaProvider(LLMProvider):
    """Провайдер Ollama для локальных моделей."""

    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        super().__init__(config)
        try:
            self.api_base = self.base_url or LLMConfig.OLLAMA_DEFAULT_URL
            # Модель уже должна быть корректно заполнена через LLMConfig/ModelsConfig.
            # Fallback на env только если config.model пуст.
            self.model = self.config.model or os.getenv("OLLAMA_MODEL", "glm-4.7-flash:q4_K_M")
            logger.info(f"Ollama провайдер инициализирован с моделью: {self.model}")
        except Exception as e:
            logger.error(f"Ошибка инициализации Ollama провайдера: {e}")
            raise

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> str:
        try:
            self._request_count += 1

            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            data = {
                "model": self.model,
                "prompt": full_prompt,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                },
                "stream": False
            }

            response = requests.post(
                f"{self.api_base}/api/generate",
                json=data,
                timeout=3600
            )

            if response.status_code != 200:
                error_msg = f"Ошибка Ollama: {response.status_code} {response.text}"
                logger.error(error_msg)
                self._error_count += 1
                raise Exception(error_msg)

            return response.json()["response"]

        except Exception as e:
            self._error_count += 1
            logger.error(f"Ошибка генерации Ollama: {e}")
            raise


# =============================================================================
# Фабрика провайдеров
# =============================================================================

class LLMProviderFactory:
    """Фабрика для создания LLM провайдеров."""

    _providers = {
        "groq": GroqProvider,
        "openrouter": OpenRouterProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
    }

    @classmethod
    def create(cls, config: Union[Dict[str, Any], LLMConfig]) -> LLMProvider:
        try:
            if isinstance(config, dict):
                config = LLMConfig.from_dict(config)

            provider_name = config.provider.value

            if provider_name not in cls._providers:
                raise ValueError(f"Неизвестный провайдер: {provider_name}")

            return cls._providers[provider_name](config)
        except Exception as e:
            logger.error(f"Ошибка создания провайдера: {e}")
            raise

    @classmethod
    def create_auto(
            cls,
            provider: str = "openrouter",
            min_context: int = 4000,
            max_tokens: int = 4096,
            temperature: float = 0.7
    ) -> LLMProvider:
        try:
            if provider == "openrouter":
                discovery = OpenRouterModelDiscovery()
                best_model = discovery.get_best_model(min_context=min_context)

                if best_model:
                    model_id = best_model.id
                    logger.info(f"Авто-выбрана модель: {model_id} (контекст: {best_model.context_length})")
                else:
                    model_id = "meta-llama/llama-3.1-8b-instruct:free"
                    logger.warning(f"Модели не найдены, используем fallback: {model_id}")

                config = LLMConfig(
                    provider=LLMProviderType.OPENROUTER,
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    context_length=best_model.context_length if best_model else 131072,
                )
            else:
                # Модель определяется через env → разумный дефолт (без хардкода qwen)
                env_models = {
                    "groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    "google": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
                    "ollama": os.getenv("OLLAMA_MODEL", "glm-4.7-flash:q4_K_M"),
                }

                config = LLMConfig(
                    provider=LLMProviderType(provider),
                    model=env_models.get(provider, "glm-4.7-flash:q4_K_M"),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=os.getenv(f"{provider.upper()}_API_KEY"),
                )

            return cls.create(config)
        except Exception as e:
            logger.error(f"Ошибка создания авто-провайдера: {e}")
            raise

    @classmethod
    def available_providers(cls) -> List[str]:
        return list(cls._providers.keys())


# =============================================================================
# Вспомогательные функции
# =============================================================================

def get_free_models(min_context: int = 4000) -> List[FreeModel]:
    """Получить список бесплатных моделей OpenRouter."""
    try:
        discovery = OpenRouterModelDiscovery()
        return discovery.get_free_models(min_context=min_context)
    except Exception as e:
        logger.error(f"Ошибка получения бесплатных моделей: {e}")
        return []


def print_free_models(task_type: Optional[TaskType] = None) -> None:
    """Вывести список бесплатных моделей OpenRouter."""
    try:
        discovery = OpenRouterModelDiscovery()

        if task_type == TaskType.HEAVY:
            models = discovery.get_models_sorted_by_size(ascending=False)
            title = "HEAVY - большие первые"
        elif task_type == TaskType.LIGHT:
            models = discovery.get_models_sorted_by_size(ascending=True)
            title = "LIGHT - маленькие первые"
        else:
            models = discovery.get_free_models()
            title = "по контексту"

        print(f"\n{'=' * 70}")
        print(f"🆓 БЕСПЛАТНЫЕ МОДЕЛИ [{title}] ({len(models)} найдено)")
        print(f"{'=' * 70}")

        for i, m in enumerate(models[:20], 1):
            size = discovery.get_model_size(m.id)
            ctx = f"{m.context_length // 1000}k" if m.context_length >= 1000 else str(m.context_length)
            caps = ""
            if "vision" in m.capabilities:
                caps += "👁"
            if "function_calling" in m.capabilities:
                caps += "🔧"
            print(f"{i:2}. {m.id:<55} {size:>3}B ctx={ctx:<6} {caps}")

        print(f"{'=' * 70}\n")
    except Exception as e:
        logger.error(f"Ошибка вывода моделей: {e}")


# =============================================================================
# CLI для тестирования
# =============================================================================

if __name__ == "__main__":
    import sys

    try:
        print("\n=== HEAVY (большие модели первые) ===")
        print_free_models(TaskType.HEAVY)

        print("\n=== LIGHT (маленькие модели первые) ===")
        print_free_models(TaskType.LIGHT)

        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            print("\n🧪 Тестируем...")

            try:
                provider = LLMProviderFactory.create_auto(min_context=8000)
                print(f"Выбрана: {provider.model}")

                response = provider.generate("Скажи 'Привет мир' ровно двумя словами.")
                print(f"Ответ: {response}")
            except Exception as e:
                print(f"Ошибка: {e}")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")