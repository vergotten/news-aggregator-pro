# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/infrastructure/ai/openrouter_models.py
# =============================================================================
"""
OpenRouter — динамическое обнаружение и выбор моделей.

Никаких захардкоженных ID или рейтингов — всё вычисляется из данных API:
- Размер параметров извлекается из названия модели (70b, 8b, 3b)
- Качество = f(параметры, контекст, max_output, capabilities)
- Каждый запуск получает свежий список и ранжирует автоматически
- Модели фильтруются по контексту, output, capabilities (не по задачам)

Использование:
    from src.infrastructure.ai.openrouter_models import (
        OpenRouterModels,
        RateLimitTracker,
        SmartModelSelector,
    )

    # Получить бесплатные модели (ранжированные динамически)
    client = OpenRouterModels(api_key="sk-or-...")
    free = client.get_free_models(min_context=8000, min_output=2000)

    # Умный выбор с fallback
    selector = SmartModelSelector(client)
    model_id = selector.select(min_context=131072, min_output=8000)

    # Rate-limit трекер (синглтон, общий для всех агентов)
    tracker = RateLimitTracker()
    tracker.record_error("model-id", 429)
"""

import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Датакласс: информация о модели
# =============================================================================

@dataclass
class OpenRouterModel:
    """
    Модель OpenRouter — все данные получаются из API, ничего захардкожено.

    Атрибуты:
        id: полный ID ('meta-llama/llama-3.2-3b-instruct:free')
        name: человекочитаемое название
        context_length: размер контекста в токенах
        max_completion_tokens: максимум токенов ответа
        prompt_price: цена за prompt (USD/1M токенов, 0 = бесплатно)
        completion_price: цена за completion
        capabilities: возможности (['chat', 'vision', 'function_calling'])
        architecture: модальность ('text->text', 'text+image->text')
        top_provider: имя провайдера (если известно)
        parameters_b: размер модели в миллиардах параметров (из названия)
        quality_score: вычисленный рейтинг качества (чем выше тем лучше)
    """
    id: str
    name: str
    context_length: int = 4096
    max_completion_tokens: int = 4096
    prompt_price: float = 0.0
    completion_price: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    architecture: Optional[str] = None
    top_provider: Optional[str] = None
    parameters_b: Optional[float] = None
    quality_score: float = 0.0

    @property
    def is_free(self) -> bool:
        """Модель бесплатна если обе цены == 0."""
        return self.prompt_price == 0.0 and self.completion_price == 0.0

    @property
    def short_name(self) -> str:
        """Имя без автора: 'meta-llama/llama-3:free' → 'llama-3:free'."""
        return self.id.split("/", 1)[1] if "/" in self.id else self.id

    @property
    def author(self) -> str:
        """Автор: 'meta-llama/llama-3:free' → 'meta-llama'."""
        return self.id.split("/", 1)[0] if "/" in self.id else "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь (для JSON-кэша)."""
        return {
            "id": self.id,
            "name": self.name,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "prompt_price": self.prompt_price,
            "completion_price": self.completion_price,
            "is_free": self.is_free,
            "capabilities": self.capabilities,
            "architecture": self.architecture,
            "top_provider": self.top_provider,
            "parameters_b": self.parameters_b,
            "quality_score": self.quality_score,
        }

    def __repr__(self) -> str:
        ctx = f"{self.context_length // 1000}k" if self.context_length >= 1000 else str(self.context_length)
        params = f"{self.parameters_b}B" if self.parameters_b else "?B"
        return f"OpenRouterModel({self.id}, {params}, ctx={ctx}, q={self.quality_score:.1f})"


# =============================================================================
# Динамическое извлечение параметров и вычисление качества
# =============================================================================

def extract_parameters_b(name_or_id: str) -> Optional[float]:
    """
    Извлечь размер модели в миллиардах из названия.

    Примеры:
        'Llama 3.3 70B Instruct' → 70.0
        'qwen2.5-72b-instruct'   → 72.0
        'mistral-small-3.1-24b'  → 24.0
        'deepseek-chat-v3-0324'  → None (не указано)
        'llama-3.2-3b-instruct'  → 3.0
        'Gemini 2.0 Flash'       → None
        'qwen/qwen3-next-80b-a3b-thinking' → 80.0 (не 3.0)
        'baidu/ernie-4.5-vl-28b-a3b' → 28.0 (не 3.0)

    Аргументы:
        name_or_id: название или ID модели

    Возвращает:
        Количество миллиардов параметров или None
    """
    text = name_or_id.lower()

    # Паттерны: '70b', '72B', '8b', '3.5b', '405b', '3b'
    # Но НЕ ловим версии вроде '3.1-24b' (3.1 — версия, 24b — параметры)
    # Ищем число + 'b' в конце слова, но не если это часть 'a3b' или подобных конструкций
    # Используем более строгий паттерн, который исключает 'a3b' и подобные конструкции
    matches = re.findall(r'(?<!a)(\d+(?:\.\d+)?)\s*b(?!a)\b', text)
    if not matches:
        return None

    # Берём самое большое число — чтобы не спутать версию (3.1) с параметрами (24b)
    values = []
    for m in matches:
        try:
            v = float(m)
            # Отсеять нереалистичные: < 0.5B или > 2000B
            if 0.5 <= v <= 2000:
                values.append(v)
        except ValueError:
            continue

    return max(values) if values else None


def compute_quality_score(model: OpenRouterModel) -> float:
    """
    Вычислить рейтинг качества модели из её параметров.

    Формула учитывает:
    - Размер модели в B (самый важный фактор)
    - Размер контекста (log-шкала, дополнительный бонус)
    - Max output (бонус за длинные ответы)
    - Количество capabilities (vision, function_calling — бонус)

    Результат: число от 0 до ~100. Чем выше — тем лучше.

    Аргументы:
        model: распарсенная модель

    Возвращает:
        Числовой рейтинг
    """
    score = 0.0

    # --- Параметры модели: 0-50 баллов ---
    # log-шкала: 3B=5.5, 8B=11, 24B=19, 70B=30, 400B=45
    if model.parameters_b and model.parameters_b > 0:
        score += min(math.log2(model.parameters_b + 1) * 5.0, 50.0)
    else:
        # Неизвестный размер — даём средний балл (вероятно крупная модель)
        score += 20.0

    # --- Контекст: 0-25 баллов ---
    # log-шкала: 4k=3, 32k=7.5, 128k=11, 1M=16
    if model.context_length > 0:
        score += min(math.log2(model.context_length / 1000 + 1) * 3.0, 25.0)

    # --- Max output: 0-15 баллов ---
    # log-шкала: 4k=3, 8k=5, 32k=9, 128k=12
    if model.max_completion_tokens > 0:
        score += min(math.log2(model.max_completion_tokens / 1000 + 1) * 3.0, 15.0)

    # --- Capabilities бонус: 0-10 ---
    caps = set(model.capabilities)
    if "vision" in caps:
        score += 4.0
    if "function_calling" in caps:
        score += 4.0
    if len(caps) > 2:
        score += 2.0

    return round(score, 2)


# =============================================================================
# Кэш моделей
# =============================================================================

@dataclass
class ModelsCache:
    """
    Кэш списка моделей с TTL.

    Атрибуты:
        models: список распарсенных моделей
        fetched_at: когда получены данные
        ttl_minutes: время жизни кэша в минутах
    """
    models: List[OpenRouterModel]
    fetched_at: datetime
    ttl_minutes: int = 60

    @property
    def is_expired(self) -> bool:
        """Истёк ли кэш (возраст > TTL)."""
        return datetime.now() >= self.fetched_at + timedelta(minutes=self.ttl_minutes)


# =============================================================================
# Основной клиент: обнаружение и выбор моделей
# =============================================================================

class OpenRouterModels:
    """
    Клиент для работы с моделями OpenRouter.

    Полностью динамический — никаких захардкоженных списков:
    - Получает модели с API
    - Извлекает размер (B) из названия
    - Вычисляет quality_score из параметров
    - Фильтрует по контексту, output, capabilities
    - Кэширует в памяти (TTL) и на диске (JSON)

    Аргументы:
        api_key: ключ OpenRouter (или из OPENROUTER_API_KEY env)
        cache_ttl: время жизни кэша в минутах (60 по умолчанию)
        cache_file: путь к файловому кэшу (опционально)
    """

    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 60,
        cache_file: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY не задан")

        self.cache_ttl = cache_ttl
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: Optional[ModelsCache] = None

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/news-aggregator",
        }

    # =====================================================================
    # Публичный API
    # =====================================================================

    def get_all_models(self, force_refresh: bool = False) -> List[OpenRouterModel]:
        """
        Получить все модели OpenRouter (с кэшированием).

        Порядок: кэш в памяти → файловый кэш → API запрос.

        Аргументы:
            force_refresh: принудительно обновить, игнорируя кэш

        Возвращает:
            Список всех моделей
        """
        # 1. Кэш в памяти
        if not force_refresh and self._cache and not self._cache.is_expired:
            return self._cache.models

        # 2. Файловый кэш
        if not force_refresh and self.cache_file and self.cache_file.exists():
            loaded = self._load_file_cache()
            if loaded and not loaded.is_expired:
                self._cache = loaded
                return self._cache.models

        # 3. Запрос к API
        models = self._fetch_from_api()
        if not models:
            logger.error("Не удалось получить модели — API недоступен и кэш пуст")
            return []

        # 4. Сохранить в кэш
        self._cache = ModelsCache(
            models=models,
            fetched_at=datetime.now(),
            ttl_minutes=self.cache_ttl,
        )
        if self.cache_file:
            self._save_file_cache()

        return models

    def get_free_models(
        self,
        limit: Optional[int] = None,
        min_context: int = 0,
        min_output: int = 0,
        sort_by: str = "quality",
        capabilities: Optional[List[str]] = None,
        min_parameters_b: Optional[float] = None,
    ) -> List[OpenRouterModel]:
        """
        Получить бесплатные модели с гибкой фильтрацией.

        Аргументы:
            limit: максимальное количество (None = все)
            min_context: минимальный контекст в токенах
            min_output: минимальный max_completion_tokens
            sort_by: 'quality' | 'context' | 'parameters' | 'output' | 'name'
            capabilities: фильтр по возможностям (['vision', 'function_calling'])
            min_parameters_b: минимальный размер модели в миллиардах

        Возвращает:
            Отсортированный и отфильтрованный список
        """
        all_models = self.get_all_models()

        result = [
            m for m in all_models
            if m.is_free
            and m.context_length >= min_context
            and m.max_completion_tokens >= min_output
        ]

        # Фильтр по capabilities
        if capabilities:
            result = [
                m for m in result
                if all(cap in m.capabilities for cap in capabilities)
            ]

        # Фильтр по размеру модели
        if min_parameters_b is not None:
            result = [
                m for m in result
                if m.parameters_b is not None and m.parameters_b >= min_parameters_b
            ]

        # Сортировка
        result = self._sort_models(result, sort_by)

        if limit:
            result = result[:limit]

        return result

    def get_best_free_model(
        self,
        min_context: int = 0,
        min_output: int = 0,
        capabilities: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        min_parameters_b: Optional[float] = None,
    ) -> Optional[OpenRouterModel]:
        """
        Получить лучшую бесплатную модель по quality_score.

        Аргументы:
            min_context: минимальный контекст
            min_output: минимальный max_completion_tokens
            capabilities: обязательные возможности
            exclude_models: ID для исключения
            min_parameters_b: минимальный размер модели

        Возвращает:
            Лучшая модель или None
        """
        exclude = set(exclude_models or [])
        models = self.get_free_models(
            min_context=min_context,
            min_output=min_output,
            sort_by="quality",
            capabilities=capabilities,
            min_parameters_b=min_parameters_b,
        )
        for m in models:
            if m.id not in exclude:
                return m
        return None

    def get_model_by_id(self, model_id: str) -> Optional[OpenRouterModel]:
        """Найти модель по ID. None если не найдена."""
        for m in self.get_all_models():
            if m.id == model_id:
                return m
        return None

    # =====================================================================
    # Сортировка (полностью динамическая)
    # =====================================================================

    @staticmethod
    def _sort_models(models: List[OpenRouterModel], sort_by: str) -> List[OpenRouterModel]:
        """
        Отсортировать модели динамически.

        'quality': по вычисленному quality_score (убывание)
        'context': по context_length (убывание)
        'parameters': по parameters_b (убывание, None — в конец)
        'output': по max_completion_tokens (убывание)
        'name': по алфавиту
        """
        result = list(models)

        if sort_by == "quality":
            result.sort(key=lambda m: m.quality_score, reverse=True)

        elif sort_by == "context":
            result.sort(key=lambda m: m.context_length, reverse=True)

        elif sort_by == "parameters":
            result.sort(key=lambda m: (m.parameters_b or 0), reverse=True)

        elif sort_by == "output":
            result.sort(key=lambda m: m.max_completion_tokens, reverse=True)

        elif sort_by == "name":
            result.sort(key=lambda m: m.name)

        return result

    # =====================================================================
    # Парсинг ответа API
    # =====================================================================

    @staticmethod
    def parse_model(data: Dict[str, Any]) -> Optional[OpenRouterModel]:
        """
        Парсинг одной модели из ответа OpenRouter API.

        Динамически извлекает все параметры:
        - Цены из pricing
        - Контекст из context_length
        - Max output из top_provider
        - Capabilities из architecture.modality и supported_parameters
        - Размер в B из названия модели
        - quality_score вычисляется из всех параметров выше

        Аргументы:
            data: элемент из data[] ответа /api/v1/models

        Возвращает:
            OpenRouterModel или None при ошибке
        """
        try:
            model_id = data.get("id", "")
            name = data.get("name", model_id)

            # Цены
            pricing = data.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0) or 0)
            completion_price = float(pricing.get("completion", 0) or 0)

            # Контекст
            context_length = data.get("context_length", 4096) or 4096

            # Провайдер / max output
            top_provider = data.get("top_provider", {})
            max_completion = top_provider.get("max_completion_tokens") or context_length // 2

            # Модальность
            architecture = data.get("architecture", {})
            modality = architecture.get("modality", "text->text")

            # Capabilities — динамически из данных API
            capabilities = ["chat"]
            if "image" in modality.lower():
                capabilities.append("vision")
            supported_params = data.get("supported_parameters", [])
            if supported_params:
                if "tools" in supported_params or "functions" in supported_params:
                    capabilities.append("function_calling")
                if "json_mode" in supported_params or "response_format" in supported_params:
                    capabilities.append("structured_output")

            # Размер модели — извлечь из ID и имени
            params_b = extract_parameters_b(model_id) or extract_parameters_b(name)

            model = OpenRouterModel(
                id=model_id,
                name=name,
                context_length=context_length,
                max_completion_tokens=max_completion,
                prompt_price=prompt_price,
                completion_price=completion_price,
                capabilities=capabilities,
                architecture=modality,
                top_provider=top_provider.get("name"),
                parameters_b=params_b,
            )

            # Вычислить quality_score
            model.quality_score = compute_quality_score(model)

            return model
        except Exception as e:
            logger.warning(f"Ошибка парсинга модели {data.get('id', '?')}: {e}")
            return None

    # =====================================================================
    # API запрос
    # =====================================================================

    def _fetch_from_api(self) -> List[OpenRouterModel]:
        """Запросить модели с OpenRouter API."""
        try:
            response = requests.get(
                f"{self.API_BASE}/models",
                headers=self.headers,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(f"OpenRouter API ошибка: {response.status_code}")
                return []

            data = response.json()
            models = []
            for item in data.get("data", []):
                model = self.parse_model(item)
                if model:
                    models.append(model)

            free_count = sum(1 for m in models if m.is_free)
            logger.info(f"Получено {len(models)} моделей ({free_count} бесплатных) с OpenRouter API")
            return models

        except requests.exceptions.RequestException as e:
            logger.error(f"Сетевая ошибка OpenRouter: {e}")
            return []

    # =====================================================================
    # Файловый кэш
    # =====================================================================

    def _save_file_cache(self) -> None:
        """Сохранить кэш в JSON-файл."""
        if not self.cache_file or not self._cache:
            return
        try:
            cache_data = {
                "fetched_at": self._cache.fetched_at.isoformat(),
                "ttl_minutes": self._cache.ttl_minutes,
                "models": [m.to_dict() for m in self._cache.models],
            }
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Кэш сохранён: {len(self._cache.models)} моделей → {self.cache_file}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")

    def _load_file_cache(self) -> Optional[ModelsCache]:
        """Загрузить кэш из JSON-файла."""
        if not self.cache_file or not self.cache_file.exists():
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            models = []
            for m in cache_data.get("models", []):
                models.append(OpenRouterModel(
                    id=m["id"],
                    name=m["name"],
                    context_length=m.get("context_length", 4096),
                    max_completion_tokens=m.get("max_completion_tokens", 4096),
                    prompt_price=m.get("prompt_price", 0),
                    completion_price=m.get("completion_price", 0),
                    capabilities=m.get("capabilities", []),
                    architecture=m.get("architecture"),
                    top_provider=m.get("top_provider"),
                    parameters_b=m.get("parameters_b"),
                    quality_score=m.get("quality_score", 0),
                ))

            return ModelsCache(
                models=models,
                fetched_at=datetime.fromisoformat(cache_data["fetched_at"]),
                ttl_minutes=cache_data.get("ttl_minutes", 60),
            )
        except Exception as e:
            logger.warning(f"Не удалось загрузить кэш: {e}")
            return None

    # =====================================================================
    # Вывод в консоль
    # =====================================================================

    def print_free_models(self, limit: int = 15) -> None:
        """Вывести топ бесплатных моделей в консоль."""
        models = self.get_free_models(limit=limit, sort_by="quality")

        print(f"\n{'=' * 90}")
        print(f"🆓 ТОП-{len(models)} БЕСПЛАТНЫХ МОДЕЛЕЙ OPENROUTER (динамический рейтинг)")
        print(f"{'=' * 90}")
        print(f"{'#':>3}  {'Модель':<55} {'Params':>7} {'Ctx':>7} {'Score':>6}")
        print(f"{'-' * 90}")

        for i, m in enumerate(models, 1):
            ctx = f"{m.context_length // 1000}k"
            params = f"{m.parameters_b:.0f}B" if m.parameters_b else "  ?B"
            caps = ""
            if "vision" in m.capabilities:
                caps += "👁"
            if "function_calling" in m.capabilities:
                caps += "🔧"
            print(f"{i:3}  {m.id:<55} {params:>7} {ctx:>7} {m.quality_score:>6.1f} {caps}")

        print(f"{'=' * 90}\n")


# =============================================================================
# Rate-limit трекер (синглтон)
# =============================================================================

@dataclass
class ModelRateLimitInfo:
    """
    Состояние rate-limit для одной модели.

    Атрибуты:
        model_id: ID модели
        last_error_at: время последней ошибки
        consecutive_errors: счётчик подряд ошибок
        cooldown_until: до какого момента модель заблокирована
        requests_this_minute: запросов за текущую минуту
        minute_started: начало текущей минуты
    """
    model_id: str
    last_error_at: Optional[datetime] = None
    consecutive_errors: int = 0
    cooldown_until: Optional[datetime] = None
    requests_this_minute: int = 0
    minute_started: Optional[datetime] = None

    @property
    def is_in_cooldown(self) -> bool:
        """В cooldown ли модель прямо сейчас."""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until

    def record_error(self, error_code: int) -> None:
        """
        Зафиксировать ошибку — поставить в cooldown.

        Экспоненциальный backoff: 30с → 60с → 120с → max 600с.
        Jitter ±20% против thundering herd.
        """
        self.last_error_at = datetime.now()
        self.consecutive_errors += 1

        cooldown_seconds = min(30 * (2 ** (self.consecutive_errors - 1)), 600)
        jitter = cooldown_seconds * 0.2 * (random.random() * 2 - 1)
        cooldown_seconds += jitter

        self.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
        logger.warning(
            f"Модель {self.model_id}: cooldown {cooldown_seconds:.0f}с "
            f"(ошибок подряд: {self.consecutive_errors})"
        )

    def record_success(self) -> None:
        """Зафиксировать успех — сбросить cooldown."""
        self.consecutive_errors = 0
        self.cooldown_until = None

        now = datetime.now()
        if self.minute_started is None or (now - self.minute_started).seconds >= 60:
            self.minute_started = now
            self.requests_this_minute = 0
        self.requests_this_minute += 1


class RateLimitTracker:
    """
    Потокобезопасный синглтон для отслеживания rate-limit всех моделей.

    Общий для всех модулей — если один компонент поймал 429,
    остальные тоже знают что модель в cooldown.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._models: Dict[str, ModelRateLimitInfo] = {}
                    inst._lock = Lock()
                    cls._instance = inst
        return cls._instance

    def get_info(self, model_id: str) -> ModelRateLimitInfo:
        """Получить состояние модели."""
        with self._lock:
            if model_id not in self._models:
                self._models[model_id] = ModelRateLimitInfo(model_id=model_id)
            return self._models[model_id]

    def record_error(self, model_id: str, error_code: int) -> None:
        self.get_info(model_id).record_error(error_code)

    def record_success(self, model_id: str) -> None:
        self.get_info(model_id).record_success()

    def is_available(self, model_id: str) -> bool:
        return not self.get_info(model_id).is_in_cooldown

    def get_available(self, model_ids: List[str]) -> List[str]:
        return [mid for mid in model_ids if self.is_available(mid)]

    def get_cooldown_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                mid: {
                    "in_cooldown": info.is_in_cooldown,
                    "cooldown_until": info.cooldown_until.isoformat() if info.cooldown_until else None,
                    "consecutive_errors": info.consecutive_errors,
                }
                for mid, info in self._models.items()
            }

    @classmethod
    def reset(cls) -> None:
        """Сбросить синглтон (для тестов)."""
        cls._instance = None


# =============================================================================
# Умный выбор модели с fallback
# =============================================================================

class SmartModelSelector:
    """
    Умный выбор модели с учётом cooldown и параметров.

    Не привязан к конкретным задачам — фильтрует по
    контексту, output, capabilities, параметрам.

    Аргументы:
        models_client: клиент OpenRouterModels
        min_context: минимальный контекст по умолчанию
        min_output: минимальный output по умолчанию
    """

    def __init__(
        self,
        models_client: OpenRouterModels,
        min_context: int = 0,
        min_output: int = 0,
    ):
        self._client = models_client
        self._tracker = RateLimitTracker()
        self._min_context = min_context
        self._min_output = min_output

    def select(
        self,
        exclude: Optional[List[str]] = None,
        min_context: Optional[int] = None,
        min_output: Optional[int] = None,
        capabilities: Optional[List[str]] = None,
        min_parameters_b: Optional[float] = None,
    ) -> Optional[str]:
        """
        Выбрать лучшую доступную модель по quality_score.

        Аргументы:
            exclude: ID для исключения
            min_context: минимальный контекст
            min_output: минимальный output
            capabilities: обязательные возможности
            min_parameters_b: минимальный размер

        Возвращает:
            ID модели или None если нет подходящих
        """
        excluded = set(exclude or [])
        ctx = min_context if min_context is not None else self._min_context
        out = min_output if min_output is not None else self._min_output

        models = self._client.get_free_models(
            min_context=ctx, min_output=out,
            sort_by="quality", capabilities=capabilities,
            min_parameters_b=min_parameters_b,
        )

        for m in models:
            if m.id not in excluded and self._tracker.is_available(m.id):
                return m.id

        return None

    def get_models_to_try(
        self,
        primary_model: Optional[str] = None,
        min_context: Optional[int] = None,
        min_output: Optional[int] = None,
    ) -> List[str]:
        """
        Собрать упорядоченный список моделей для последовательных попыток.

        Порядок: primary (если доступна) → остальные по quality_score.

        Аргументы:
            primary_model: предпочтительная модель
            min_context: минимальный контекст
            min_output: минимальный output

        Возвращает:
            Список ID без дубликатов
        """
        ctx = min_context if min_context is not None else self._min_context
        out = min_output if min_output is not None else self._min_output
        result: List[str] = []

        if primary_model and self._tracker.is_available(primary_model):
            result.append(primary_model)

        for m in self._client.get_free_models(min_context=ctx, min_output=out, sort_by="quality"):
            if m.id not in result and self._tracker.is_available(m.id):
                result.append(m.id)

        return result


# =============================================================================
# Утилиты: извлечение JSON из ответов
# =============================================================================

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Извлечь JSON из текстового ответа модели.

    5 стратегий: прямой парсинг → ```json → ``` → скобки { } → очистка.

    Аргументы:
        text: ответ модели

    Возвращает:
        Распарсенный словарь/список или None
    """
    if not text:
        return None
    text = text.strip()

    # 1: прямой парсинг
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2: ```json ... ```
    for block in re.findall(r'```json\s*([\s\S]*?)\s*```', text, re.IGNORECASE):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # 3: ``` ... ```
    for block in re.findall(r'```\s*([\s\S]*?)\s*```', text):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # 4: сбалансированные { }
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # 5: очистка
    cleaned = re.sub(r'^[^{]*', '', text)
    cleaned = re.sub(r'}[^}]*$', '}', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


# =============================================================================
# Удобные функции
# =============================================================================

_default_client: Optional[OpenRouterModels] = None


def get_openrouter_models(
    api_key: Optional[str] = None,
    force_new: bool = False,
) -> OpenRouterModels:
    """Получить глобальный клиент (ленивая инициализация)."""
    global _default_client
    if _default_client is None or force_new:
        _default_client = OpenRouterModels(
            api_key=api_key,
            cache_file="cache/openrouter_models.json",
        )
    return _default_client


def get_best_free_model_id(
    min_context: int = 0,
    min_output: int = 0,
    exclude: Optional[List[str]] = None,
) -> Optional[str]:
    """Получить ID лучшей бесплатной модели."""
    client = get_openrouter_models()
    model = client.get_best_free_model(
        min_context=min_context, min_output=min_output,
        exclude_models=exclude,
    )
    return model.id if model else None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    parser = argparse.ArgumentParser(description="OpenRouter — динамическое обнаружение моделей")
    parser.add_argument("--limit", "-l", type=int, default=15, help="Количество моделей")
    parser.add_argument("--min-context", "-c", type=int, default=0, help="Минимальный контекст")
    parser.add_argument("--min-output", "-o", type=int, default=0, help="Минимальный output")
    parser.add_argument("--min-params", "-p", type=float, default=None, help="Минимальный размер (B)")
    parser.add_argument("--refresh", "-r", action="store_true", help="Обновить кэш")
    parser.add_argument("--json", "-j", action="store_true", help="Вывод в JSON")
    args = parser.parse_args()

    client = OpenRouterModels(cache_file="cache/openrouter_models.json")

    if args.refresh:
        client.get_all_models(force_refresh=True)

    models = client.get_free_models(
        limit=args.limit, min_context=args.min_context,
        min_output=args.min_output, sort_by="quality",
        min_parameters_b=args.min_params,
    )

    if args.json:
        print(json.dumps([m.to_dict() for m in models], indent=2, ensure_ascii=False))
    else:
        client.print_free_models(limit=args.limit)